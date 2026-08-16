"""
Headless evaluation of the ORBSlam3 pipeline against TUM RGB-D ground truth.
Uses depth images (RGB-D mode / PnP tracking) for metric-scale trajectory estimation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
import yaml
from system import ORBSlam3
from metrics import (align, compute_ate, compute_rpe, trajectory_length,
                     max_step, summarize, format_summary)

DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "rgbd_dataset_freiburg1_xyz")
CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "camera_config.yaml")
DEPTH_SCALE = 5000.0  # TUM freiburg1 depth scale factor
# Set to None to use every associated frame. Note the previous value of 400
# produced step = 798 // 400 = 1, i.e. no subsampling at all, while the script
# still printed "Subsampled to 798 frames (every 1th frame)".
MAX_FRAMES = None


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

    if MAX_FRAMES and len(matched) > MAX_FRAMES:
        step = max(1, len(matched) // MAX_FRAMES)
        matched = matched[::step]
        print(f"Subsampled to {len(matched)} frames (every {step}th frame)")
    else:
        print(f"Using all {len(matched)} associated frames (no subsampling)")

    # Pure VO/tracking benchmark -- disable loop closure so this measures
    # the tracker in isolation (evaluate_tum_with_loop_closure.py is the
    # script that exercises loop closure on/off).
    slam = ORBSlam3(CONFIG, enable_loop_closure=False)
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

    # Metrics come from the shared evaluation/metrics.py module so that every
    # script in this repo reports numbers computed the exact same way.
    ate_scaled = compute_ate(est_xyz, gt_xyz, with_scale=True)
    ate_noscale = compute_ate(est_xyz, gt_xyz, with_scale=False)
    aligned_scaled, scale_factor = align(est_xyz, gt_xyz, with_scale=True)
    aligned_noscale, _ = align(est_xyz, gt_xyz, with_scale=False)
    errors_scaled = np.linalg.norm(aligned_scaled - gt_xyz, axis=1)
    errors_noscale = np.linalg.norm(aligned_noscale - gt_xyz, axis=1)

    # RPE at several deltas: local tracking quality, insensitive to the
    # global drift that dominates ATE.
    rpe = {d: compute_rpe(est_xyz, gt_xyz, delta=d) for d in (1, 10, 30)}

    summary = summarize(est_xyz, gt_xyz, label="TUM fr1/xyz visual odometry")

    print("\n" + "=" * 66)
    print("  TUM RGB-D freiburg1_xyz — Visual Odometry Evaluation")
    print("=" * 66)
    print(format_summary(summary))
    print("=" * 66)
    print("  Tracking path counts:", slam.tracker.stats() if slam.tracker else "n/a")
    print("=" * 66)
    print("""  How to read the ratios above: a value >= 1.00 means the system is no
  better than the trivial estimator for that metric (a fixed point for ATE,
  zero motion for RPE). Raw metre values alone cannot distinguish a working
  tracker from a broken one on a sequence with this little motion.""")

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    save_kwargs = dict(
        est_xyz=est_xyz, gt_xyz=gt_xyz,
        aligned_scaled=aligned_scaled, aligned_noscale=aligned_noscale,
        errors_scaled=errors_scaled, errors_noscale=errors_noscale,
        scale_factor=scale_factor,
    )
    for k, v in ate_scaled.items():
        save_kwargs[f"ate_scaled_{k}"] = v
    for k, v in ate_noscale.items():
        save_kwargs[f"ate_noscale_{k}"] = v
    for d, r in rpe.items():
        if r:
            for k, v in r.items():
                save_kwargs[f"rpe{d}_{k}"] = v

    np.savez(os.path.join(os.path.dirname(__file__), "..", "results", "tum_eval.npz"),
             **save_kwargs)
    print("\nRaw results saved to results/tum_eval.npz")


if __name__ == "__main__":
    main()
