"""
Retrain the loop-closure classifier on REAL labels derived from ground truth,
instead of the original synthetic/fabricated training data.

Label generation strategy (grounded in real data, not hand-labeling by eye):
  - POSITIVE (loop closure): frame pairs whose ground-truth camera positions
    are spatially close (< POS_THRESH) but temporally far apart
    (> TIME_GAP), i.e. the camera genuinely revisited the same place.
  - NEGATIVE (no loop closure): frame pairs that are spatially far apart
    (> NEG_THRESH), regardless of time gap.

Features are the SAME feature set the original code used (SIFT match count,
match ratio, distance statistics, spatial distance statistics) -- MINUS the
CNN similarity feature, since a CNN feature extractor (ResNet18/PyTorch) is
not available in this environment. This is a deliberate, documented
simplification: it does not affect the validity of the real-vs-synthetic
label comparison, since both classifiers use the same feature set.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

DATASET = os.path.join(os.path.dirname(__file__), "..", "..",
                        "Loop_Closure_and_Algorithms", "data", "rgbd_dataset_freiburg1_xyz")
POS_THRESH = 0.08   # meters -- "same place"
NEG_THRESH = 0.5    # meters -- "clearly different place"
TIME_GAP = 5.0       # seconds -- must be temporally separated to count as a "revisit"
N_PAIRS_PER_CLASS = 400
RNG_SEED = 42


def load_tum_list(path):
    out = []
    with open(path) as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            p = l.split()
            out.append((float(p[0]), p[1]))
    return out


def load_gt(path):
    out = []
    with open(path) as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            p = l.split()
            out.append(tuple(float(v) for v in p[:8]))
    return out


def extract_sift_features(img1_path, img2_path, sift, matcher):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return None

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.zeros(8)

    raw = matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
    matches = [m for m, n in raw if len(raw) and m.distance < 0.7 * n.distance] if raw and len(raw[0]) == 2 else []

    if not matches or len(matches) < 2:
        return np.zeros(8)

    match_distances = [m.distance for m in matches]
    match_ratio = len(matches) / ((len(kp1) + len(kp2)) / 2)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    spatial = np.linalg.norm(pts1 - pts2, axis=1)
    dstats = np.percentile(match_distances, [25, 50, 75])
    sstats = np.percentile(spatial, [25, 50, 75]) if len(spatial) else [float("inf")] * 3

    return np.array([
        len(matches),
        match_ratio,
        np.mean(match_distances),
        np.std(match_distances),
        np.min(spatial) if len(spatial) else float("inf"),
        dstats[1],
        sstats[1],
        match_ratio * np.exp(-np.mean(match_distances) / 100),
    ])


def main():
    rng = np.random.default_rng(RNG_SEED)

    rgb_list = load_tum_list(os.path.join(DATASET, "rgb.txt"))
    gt = load_gt(os.path.join(DATASET, "groundtruth.txt"))
    gt_ts = np.array([g[0] for g in gt])
    gt_xyz = np.array([g[1:4] for g in gt])

    # Match each rgb frame to nearest ground-truth pose
    rgb_ts = np.array([r[0] for r in rgb_list])
    frame_positions = []
    for ts, fname in rgb_list:
        idx = np.argmin(np.abs(gt_ts - ts))
        frame_positions.append(gt_xyz[idx])
    frame_positions = np.array(frame_positions)

    n = len(rgb_list)
    print(f"Loaded {n} frames with ground-truth positions.")

    # Build candidate pairs
    pos_pairs, neg_pairs = [], []
    idx_pool = np.arange(0, n, 3)  # subsample for tractability
    for a in range(len(idx_pool)):
        i = idx_pool[a]
        for b in range(a + 1, len(idx_pool)):
            j = idx_pool[b]
            if rgb_ts[j] - rgb_ts[i] < TIME_GAP:
                continue
            d = np.linalg.norm(frame_positions[j] - frame_positions[i])
            if d < POS_THRESH:
                pos_pairs.append((i, j))
            elif d > NEG_THRESH:
                neg_pairs.append((i, j))

    print(f"Candidate positives: {len(pos_pairs)}, candidate negatives: {len(neg_pairs)}")

    rng.shuffle(pos_pairs)
    rng.shuffle(neg_pairs)
    pos_pairs = pos_pairs[:N_PAIRS_PER_CLASS]
    neg_pairs = neg_pairs[:N_PAIRS_PER_CLASS]
    print(f"Sampled {len(pos_pairs)} positive / {len(neg_pairs)} negative pairs for feature extraction.")

    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher()

    X, y = [], []
    all_pairs = [(p, 1) for p in pos_pairs] + [(p, 0) for p in neg_pairs]
    for k, ((i, j), label) in enumerate(all_pairs):
        p1 = os.path.join(DATASET, rgb_list[i][1])
        p2 = os.path.join(DATASET, rgb_list[j][1])
        feats = extract_sift_features(p1, p2, sift, matcher)
        if feats is not None:
            X.append(feats)
            y.append(label)
        if (k + 1) % 100 == 0:
            print(f"  Extracted features for {k+1}/{len(all_pairs)} pairs")

    X = np.array(X)
    y = np.array(y)
    print(f"\nFinal real-labeled dataset: {len(X)} samples ({y.sum()} positive, {len(y)-y.sum()} negative)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RNG_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RNG_SEED)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 55)
    print("  Real-Labeled Loop-Closure Classifier -- Results")
    print("=" * 55)
    print(f"  Train/test split: {len(X_train)}/{len(X_test)}")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Test ROC-AUC:  {auc:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["no_loop", "loop"]))
    print("=" * 55)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    joblib.dump(clf, os.path.join(out_dir, "rf_loop_closure_REAL_labels.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "rf_scaler_REAL_labels.joblib"))
    np.savez(os.path.join(out_dir, "real_label_eval.npz"),
             X_test=X_test, y_test=y_test, y_pred=y_pred, y_proba=y_proba,
             accuracy=acc, auc=auc)
    print(f"\nModel + eval results saved to {out_dir}/")


if __name__ == "__main__":
    main()
