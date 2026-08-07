"""
Headless evaluation of the ORBSlam3 pipeline against TUM RGB-D ground truth.
Uses depth images (RGB-D mode / PnP tracking) for metric-scale trajectory estimation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import cv2
import yaml
from system import ORBSlam3

DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "rgbd_dataset_freiburg1_xyz")
CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "camera_config.yaml")
DEPTH_SCALE = 5000.0  # TUM freiburg1 depth scale factor
MAX_FRAMES = 400       # subsample for reasonable runtime in sandbox


def load_tum_list(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            entries.append((float(parts[0]), parts[1]))
    return entries


def associate(rgb_list, depth_list, gt_list, max_diff=0.02):
    """Associate rgb, depth, and ground truth entries by nearest timestamp."""
    depth_ts = np.array([d[0] for d in depth_list])
    gt_ts = np.array([g[0] for g in gt_list])
    matched = []
    for ts, fname in rgb_list:
        di = np.argmin(np.abs(depth_ts - ts))
        gi = np.argmin(np.abs(gt_ts - ts))
        if abs(depth_ts[di] - ts) <= max_diff and abs(gt_ts[gi] - ts) <= max_diff:
            matched.append((ts, fname, depth_list[di][1], gt_list[gi]))
    return matched


def quat_to_R(qx, qy, qz, qw):
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def main():
    rgb_list = load_tum_list(os.path.join(DATASET, "rgb.txt"))
    depth_list = load_tum_list(os.path.join(DATASET, "depth.txt"))
    gt_raw = []
    with open(os.path.join(DATASET, "groundtruth.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            gt_raw.append(tuple(float(v) for v in p[:8]))

    matched = associate(rgb_list, depth_list, gt_raw)
    print(f"Associated {len(matched)} rgb-depth-groundtruth triplets")

    if MAX_FRAMES:
        step = max(1, len(matched) // MAX_FRAMES)
        matched = matched[::step]
        print(f"Subsampled to {len(matched)} frames (every {step}th frame)")

    slam = ORBSlam3(CONFIG)
    if not slam.initialize():
        print("Failed to initialize SLAM")
        return

    est_traj = []  # (timestamp, x, y, z)
    gt_traj = []   # (timestamp, x, y, z)

    for i, (ts, rgb_fname, depth_fname, gt_entry) in enumerate(matched):
        frame = cv2.imread(os.path.join(DATASET, rgb_fname))
        depth_raw = cv2.imread(os.path.join(DATASET, depth_fname), cv2.IMREAD_UNCHANGED)
        if frame is None or depth_raw is None:
            continue
        depth = depth_raw.astype(np.float32) / DEPTH_SCALE

        pose, kps = slam.process_frame(frame, depth)
        if pose is not None:
            pos = pose[:3, 3]
            est_traj.append((ts, pos[0], pos[1], pos[2]))
            gt_traj.append((gt_entry[0], gt_entry[1], gt_entry[2], gt_entry[3]))

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(matched)} frames, trajectory length={len(est_traj)}")

    print(f"\nFinal trajectory: {len(est_traj)} poses")

    if len(est_traj) < 10:
        print("Too few poses tracked — evaluation not meaningful.")
        return

    est_xyz = np.array([p[1:4] for p in est_traj])
    gt_xyz = np.array([p[1:4] for p in gt_traj])

    # Umeyama alignment (with scale, since PnP-based tracking should already be metric,
    # but we report both to be transparent about residual scale error)
    def align(est, gt, with_scale=True):
        mu_est = est.mean(axis=0)
        mu_gt = gt.mean(axis=0)
        est_c = est - mu_est
        gt_c = gt - mu_gt
        if with_scale:
            scale = np.sqrt((gt_c**2).sum()) / (np.sqrt((est_c**2).sum()) + 1e-12)
        else:
            scale = 1.0
        H = (est_c * scale).T @ gt_c
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1, 1, d])
        R = Vt.T @ D @ U.T
        t = mu_gt - R @ (mu_est * scale)
        aligned = (R @ (est * scale).T).T + t
        return aligned, scale

    aligned_scaled, scale_factor = align(est_xyz, gt_xyz, with_scale=True)
    aligned_noscale, _ = align(est_xyz, gt_xyz, with_scale=False)

    errors_scaled = np.linalg.norm(aligned_scaled - gt_xyz, axis=1)
    errors_noscale = np.linalg.norm(aligned_noscale - gt_xyz, axis=1)

    print("\n" + "=" * 55)
    print("  TUM RGB-D freiburg1_xyz — Evaluation Results")
    print("=" * 55)
    print(f"  Poses evaluated:            {len(est_traj)}")
    print(f"  Recovered scale factor:     {scale_factor:.4f}  (1.0 = perfectly metric)")
    print(f"\n  --- With scale correction (Sim(3) alignment) ---")
    print(f"  ATE RMSE:  {np.sqrt(np.mean(errors_scaled**2)):.4f} m")
    print(f"  ATE Mean:  {np.mean(errors_scaled):.4f} m")
    print(f"  ATE Max:   {np.max(errors_scaled):.4f} m")
    print(f"\n  --- Without scale correction (SE(3) alignment only) ---")
    print(f"  ATE RMSE:  {np.sqrt(np.mean(errors_noscale**2)):.4f} m")
    print(f"  ATE Mean:  {np.mean(errors_noscale):.4f} m")
    print(f"  ATE Max:   {np.max(errors_noscale):.4f} m")
    print("=" * 55)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    np.savez(os.path.join(os.path.dirname(__file__), "..", "results", "tum_eval.npz"),
             est_xyz=est_xyz, gt_xyz=gt_xyz,
             aligned_scaled=aligned_scaled, aligned_noscale=aligned_noscale,
             errors_scaled=errors_scaled, errors_noscale=errors_noscale,
             scale_factor=scale_factor)
    print("\nRaw results saved to results/tum_eval.npz")


if __name__ == "__main__":
    main()
