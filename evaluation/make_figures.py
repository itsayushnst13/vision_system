"""
Generate every figure in results/figures/ from the saved evaluation
artifacts (results/*.npz, results/*.joblib).

Run the three evaluation scripts first -- this script only reads what they
produced, it does not re-run any SLAM or training. That separation is
deliberate: figures are cheap to regenerate and should never silently
depend on a fresh (possibly different) run.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
FIGDIR = os.path.join(RESULTS, "figures")


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, os.path.join(HERE, '..'))}")


def fig_trajectory():
    """Estimated vs ground-truth trajectory, and error over time."""
    path = os.path.join(RESULTS, "tum_eval.npz")
    if not os.path.exists(path):
        print("  [skip] tum_eval.npz not found -- run evaluate_tum.py first")
        return
    d = np.load(path)
    gt = d["gt_xyz"]
    aligned = d["aligned_scaled"]
    aligned_ns = d["aligned_noscale"]
    err = d["errors_scaled"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(gt[:, 0], gt[:, 1], label="Ground truth", lw=2, color="#2b6cb0")
    ax.plot(aligned[:, 0], aligned[:, 1], label="Estimated (Sim(3) aligned)",
            lw=1.6, color="#c53030", alpha=0.85)
    ax.set_title("Trajectory, scale-corrected (top-down)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=8); ax.axis("equal"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(gt[:, 0], gt[:, 1], label="Ground truth", lw=2, color="#2b6cb0")
    ax.plot(aligned_ns[:, 0], aligned_ns[:, 1], label="Estimated (SE(3) only)",
            lw=1.2, color="#c53030", alpha=0.7)
    ax.set_title("Trajectory, raw drift (no scale correction)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=8); ax.axis("equal"); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(err, lw=1.2, color="#805ad5")
    ax.set_title("ATE per frame (Sim(3) aligned)")
    ax.set_xlabel("frame"); ax.set_ylabel("error (m)")
    ax.grid(alpha=0.3)

    fig.suptitle("Visual odometry on TUM RGB-D freiburg1_xyz", y=1.02, fontsize=12)
    _save(fig, "01_trajectory_vo.png")


def fig_classifier():
    """Confusion matrix + ROC + precision-recall for the current classifier."""
    path = os.path.join(RESULTS, "real_label_eval.npz")
    if not os.path.exists(path):
        print("  [skip] real_label_eval.npz not found -- run retrain script first")
        return
    try:
        from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                                     precision_recall_curve, average_precision_score)
    except ImportError:
        print("  [skip] scikit-learn not available")
        return

    d = np.load(path)
    y_true, y_pred, y_proba = d["y_test"], d["y_pred"], d["y_proba"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    cm = confusion_matrix(y_true, y_pred)
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no loop", "loop"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no loop", "loop"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title("Confusion matrix (held-out real pairs)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax = axes[1]
    ax.plot(fpr, tpr, lw=2, color="#2b6cb0", label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve"); ax.legend(); ax.grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ax = axes[2]
    ax.plot(rec, prec, lw=2, color="#c53030",
            label=f"AP = {average_precision_score(y_true, y_proba):.3f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision-recall curve"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("Loop-closure classifier, trained on real ground-truth-derived labels "
                 "(with hard-negative mining)", y=1.03, fontsize=12)
    _save(fig, "02_classifier_metrics.png")


def fig_synthetic_vs_real():
    """Bar chart: synthetic-trained vs real-trained on the same real test set."""
    real_path = os.path.join(RESULTS, "real_label_eval.npz")
    easy_path = os.path.join(RESULTS, "real_label_eval_v1_easy.npz")
    if not os.path.exists(real_path):
        print("  [skip] real_label_eval.npz not found")
        return

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, recall_score, precision_score
    except ImportError:
        print("  [skip] scikit-learn not available")
        return

    d = np.load(real_path)
    X_test, y_test = d["X_test"], d["y_test"]

    # Rebuild the original synthetic-trained model exactly as the legacy code
    # did, so the comparison is like-for-like on the same real test set.
    rng = np.random.default_rng(42)
    n = 2000
    n_matches = rng.normal(75, 15, n)
    match_ratios = rng.normal(0.7, 0.15, n)
    match_distances = rng.normal(100, 20, n)
    distance_stds = rng.normal(15, 5, n)
    spatial_distances = rng.normal(95, 20, n)
    Xs = np.column_stack([
        np.abs(n_matches), np.clip(match_ratios, 0, 1), np.abs(match_distances),
        np.abs(distance_stds), np.abs(spatial_distances),
        np.abs(match_distances) * 0.8, np.abs(spatial_distances) * 0.9,
        np.clip(match_ratios, 0, 1) * np.exp(-np.abs(match_distances) / 100),
    ])
    quality = (0.3 * np.clip(n_matches / 100, 0, 1) + 0.2 * np.clip(match_ratios, 0, 1)
               + 0.5 * np.exp(-np.abs(match_distances) / 150))
    ys = (quality > 0.6).astype(int)

    sc = StandardScaler().fit(Xs)
    clf_s = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf_s.fit(sc.transform(Xs), ys)
    y_pred_s = clf_s.predict(sc.transform(X_test))

    y_pred_r = d["y_pred"]

    labels = ["Accuracy", "Precision (loop)", "Recall (loop)"]
    synth = [accuracy_score(y_test, y_pred_s),
             precision_score(y_test, y_pred_s, zero_division=0),
             recall_score(y_test, y_pred_s, zero_division=0)]
    real = [accuracy_score(y_test, y_pred_r),
            precision_score(y_test, y_pred_r, zero_division=0),
            recall_score(y_test, y_pred_r, zero_division=0)]

    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    b1 = ax.bar(x - w / 2, synth, w, label="Trained on synthetic labels (original)",
                color="#a0aec0")
    b2 = ax.bar(x + w / 2, real, w, label="Trained on real labels (this work)",
                color="#2b6cb0")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15); ax.set_ylabel("score")
    ax.set_title("Same held-out real test set, two training regimes")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    _save(fig, "03_synthetic_vs_real.png")


def fig_loop_closure():
    """Loop-closure ON vs OFF trajectories and ATE bars."""
    path = os.path.join(RESULTS, "loop_closure_comparison.npz")
    if not os.path.exists(path):
        print("  [skip] loop_closure_comparison.npz not found")
        return
    sys.path.insert(0, HERE)
    from metrics import align

    d = np.load(path)
    est_off, gt_off = d["est_off"], d["gt_off"]
    est_on, gt_on = d["est_on"], d["gt_on"]

    off_ns, _ = align(est_off, gt_off, with_scale=False)
    on_ns, _ = align(est_on, gt_on, with_scale=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(gt_off[:, 0], gt_off[:, 1], label="Ground truth", lw=2.4, color="#2b6cb0")
    ax.plot(off_ns[:, 0], off_ns[:, 1], label="Loop closure OFF", lw=1.5, color="#a0aec0")
    ax.plot(on_ns[:, 0], on_ns[:, 1], label="Loop closure ON", lw=1.5, color="#c53030")
    ax.set_title("Trajectory, SE(3) aligned (raw drift visible)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=9); ax.axis("equal"); ax.grid(alpha=0.3)

    ax = axes[1]
    # Show RPE and path length alongside ATE. Reporting ATE alone is what made
    # the earlier version of this experiment look like a success: a pose graph
    # that drags keyframes together improves ATE while wrecking local accuracy.
    names = ["ATE SE(3)", "ATE Sim(3)", "RPE delta=1"]
    off_vals = [float(d["off_ate_se3"]), float(d["off_ate_sim3"]), float(d["off_rpe1"])]
    on_vals = [float(d["on_ate_se3"]), float(d["on_ate_sim3"]), float(d["on_rpe1"])]
    base_vals = [float(d["off_ate_baseline"]), float(d["off_ate_baseline"]),
                 float(d["off_rpe1_baseline"])]
    x = np.arange(len(names)); w = 0.27
    b1 = ax.bar(x - w, off_vals, w, label="OFF", color="#a0aec0")
    b2 = ax.bar(x, on_vals, w, label="ON", color="#c53030")
    b3 = ax.bar(x + w, base_vals, w, label="trivial baseline", color="#000000", alpha=0.35)
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004,
                    f"{b.get_height():.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("RMSE (m)")
    ax.set_title(f"Effect of loop closure ({int(d['n_loops_on'])} verified closures)\n"
                 f"path length {float(d['off_path_length']):.2f} -> "
                 f"{float(d['on_path_length']):.2f} m (GT {float(d['on_gt_path_length']):.2f})",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    _save(fig, "04_loop_closure_effect.png")


def main():
    print("Generating figures into results/figures/ ...")
    fig_trajectory()
    fig_classifier()
    fig_synthetic_vs_real()
    fig_loop_closure()
    print("Done.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Figures added with the metrics fix: every error metric shown against the
# trivial-estimator baseline it must beat.
# ---------------------------------------------------------------------------
def fig05_baselines(out_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    d = np.load(os.path.join(RESULTS, "tum_eval.npz"))
    est, gt = d["est_xyz"], d["gt_xyz"]
    from metrics import compute_ate, compute_rpe, static_point_ate

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ate = compute_ate(est, gt, with_scale=True)
    ax1.bar(["static-point\nbaseline", "system\n(Sim(3) ATE)"],
            [ate["baseline_static_point"], ate["rmse"]],
            color=["#999999", "#c0392b"])
    ax1.axhline(ate["baseline_static_point"], ls="--", c="k", lw=1)
    ax1.set_ylabel("ATE RMSE (m)")
    ax1.set_title("Global accuracy vs the do-nothing floor")
    for i, v in enumerate([ate["baseline_static_point"], ate["rmse"]]):
        ax1.text(i, v * 1.02, f"{v:.4f}", ha="center")

    deltas = [1, 10, 30]
    sysv, base = [], []
    for dd in deltas:
        r = compute_rpe(est, gt, delta=dd)
        sysv.append(r["rmse"])
        base.append(r["baseline_null_motion"])
    x = np.arange(len(deltas))
    ax2.bar(x - 0.18, base, 0.36, label="null-motion baseline", color="#999999")
    ax2.bar(x + 0.18, sysv, 0.36, label="system", color="#c0392b")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"delta={d}" for d in deltas])
    ax2.set_ylabel("RPE RMSE (m)")
    ax2.set_title("Local accuracy vs assuming the camera never moved")
    ax2.legend()
    ax2.set_yscale("log")

    fig.suptitle("Error metrics are only interpretable against their baselines",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "05_baselines.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  wrote {p}")


def fig06_synthetic_distortion(out_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    f = os.path.join(RESULTS, "synthetic_distortion.npz")
    if not os.path.exists(f):
        print("  (skipping 06: run tests/test_distortion_bug.py first)")
        return
    d = np.load(f)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["ATE Sim(3)", "ATE SE(3)", "RPE delta=1"]
    buggy = [float(d["buggy_ate_sim3"]), float(d["buggy_ate_se3"]), float(d["buggy_rpe1"])]
    fixed = [float(d["fixed_ate_sim3"]), float(d["fixed_ate_se3"]), float(d["fixed_rpe1"])]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, buggy, 0.36, label="distortion mishandled", color="#c0392b")
    ax.bar(x + 0.18, fixed, 0.36, label="distortion fixed", color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("error (m, log scale)")
    ax.set_title("Synthetic sequence, real fr1 intrinsics:\nsame scene, same noise, only distortion handling differs")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "06_synthetic_distortion.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    _out = os.path.join(RESULTS, "figures")
    os.makedirs(_out, exist_ok=True)
    fig05_baselines(_out)
    fig06_synthetic_distortion(_out)
