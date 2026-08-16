"""
Compares tracking accuracy WITH vs WITHOUT the real-labeled loop-closure
classifier wired into the live SLAM pipeline, on the TUM RGB-D
freiburg1_xyz benchmark.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2
from system import ORBSlam3
from metrics import summarize, format_summary, compute_ate, compute_rpe, trajectory_length

DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "rgbd_dataset_freiburg1_xyz")
CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "camera_config.yaml")
DEPTH_SCALE = 5000.0
MAX_FRAMES = 120


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
    depth_ts = np.array([d[0] for d in depth_list])
    gt_ts = np.array([g[0] for g in gt_list])
    matched = []
    for ts, fname in rgb_list:
        di = np.argmin(np.abs(depth_ts - ts))
        gi = np.argmin(np.abs(gt_ts - ts))
        if abs(depth_ts[di] - ts) <= max_diff and abs(gt_ts[gi] - ts) <= max_diff:
            matched.append((ts, fname, depth_list[di][1], gt_list[gi]))
    return matched


# NOTE: this script used to carry its own private copy of `align`, with the
# non-optimal RMS-ratio scale term, so it silently reported different numbers
# from evaluate_tum.py for the same trajectory. All metrics now come from the
# shared evaluation/metrics.py.


def run_pipeline(matched, enable_loop_closure):
    slam = ORBSlam3(CONFIG, enable_loop_closure=enable_loop_closure)
    if not slam.initialize():
        raise RuntimeError("SLAM init failed")

    est_traj, gt_traj = [], []
    for i, (ts, rgb_fname, depth_fname, gt_entry) in enumerate(matched):
        frame = cv2.imread(os.path.join(DATASET, rgb_fname))
        depth_raw = cv2.imread(os.path.join(DATASET, depth_fname), cv2.IMREAD_UNCHANGED)
        if frame is None or depth_raw is None:
            continue
        depth = depth_raw.astype(np.float32) / DEPTH_SCALE

        # Only pair a ground-truth entry with a frame when the SLAM system
        # actually appended a new pose. process_frame() returns the previous
        # pose (not None) when tracking fails mid-sequence without appending
        # to slam.trajectory, so keying off the return value alone would
        # desynchronise est_traj from slam.trajectory and from gt_traj.
        n_before = len(slam.trajectory)
        pose, kps = slam.process_frame(frame, depth)
        if len(slam.trajectory) > n_before:
            est_traj.append(slam.trajectory[-1][:3, 3].copy())
            gt_traj.append(gt_entry[1:4])

        if (i + 1) % 50 == 0:
            print(f"  [{'loop-closure ON' if enable_loop_closure else 'loop-closure OFF'}] "
                  f"{i+1}/{len(matched)} frames, {len(slam.loop_pairs)} loop closures found")

    if enable_loop_closure:
        slam.flush_loop_closures()
        # Re-read corrected trajectory positions. Lengths are guaranteed to
        # match gt_traj by the append logic above.
        est_traj = [p[:3, 3].copy() for p in slam.trajectory]

    est_arr, gt_arr = np.array(est_traj), np.array(gt_traj)
    assert len(est_arr) == len(gt_arr), (
        f"est/gt length mismatch: {len(est_arr)} vs {len(gt_arr)}")
    return est_arr, gt_arr, len(slam.loop_pairs)


def evaluate(est_xyz, gt_xyz, label):
    """
    Report ATE *and* RPE for each configuration.

    Reporting ATE alone is what allowed the earlier version of this experiment
    to look like a success. A pose graph that drags keyframes together improves
    global ATE while wrecking local accuracy, and RPE is the metric that
    exposes it. Path length is printed for the same reason: a correction that
    inflates the path far beyond ground truth is compacting the trajectory
    rather than straightening it.
    """
    s = summarize(est_xyz, gt_xyz, label=label)
    print(f"\n--- {label} ---")
    print(format_summary(s))
    return {
        "ate_sim3": s["ate_sim3"]["rmse"],
        "ate_se3": s["ate_se3"]["rmse"],
        "rpe1": s["rpe"][1]["rmse"] if s["rpe"].get(1) else float("nan"),
        "rpe1_baseline": s["rpe"][1]["baseline_null_motion"] if s["rpe"].get(1) else float("nan"),
        "ate_baseline": s["ate_sim3"]["baseline_static_point"],
        "path_length": s["est_path_length"],
        "gt_path_length": s["gt_path_length"],
        "scale": s["ate_sim3"]["scale"],
    }


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
    if MAX_FRAMES:
        step = max(1, len(matched) // MAX_FRAMES)
        matched = matched[::step]
    print(f"Running on {len(matched)} frames.\n")

    print("=" * 60)
    print("Run 1: loop closure DISABLED (baseline)")
    print("=" * 60)
    est_off, gt_off, n_loops_off = run_pipeline(matched, enable_loop_closure=False)
    r_off = evaluate(est_off, gt_off, "Loop closure OFF")

    print("\n" + "=" * 60)
    print("Run 2: loop closure ENABLED")
    print("=" * 60)
    est_on, gt_on, n_loops_on = run_pipeline(matched, enable_loop_closure=True)
    r_on = evaluate(est_on, gt_on, "Loop closure ON")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Loop closures accepted: {n_loops_on}")
    print(f"  {'metric':<28}{'OFF':>12}{'ON':>12}{'baseline':>12}")
    print(f"  {'ATE RMSE, Sim(3)-aligned':<28}{r_off['ate_sim3']:>12.4f}{r_on['ate_sim3']:>12.4f}{r_off['ate_baseline']:>12.4f}")
    print(f"  {'ATE RMSE, SE(3)-aligned':<28}{r_off['ate_se3']:>12.4f}{r_on['ate_se3']:>12.4f}{r_off['ate_baseline']:>12.4f}")
    print(f"  {'RPE RMSE, delta=1':<28}{r_off['rpe1']:>12.4f}{r_on['rpe1']:>12.4f}{r_off['rpe1_baseline']:>12.4f}")
    print(f"  {'path length (m)':<28}{r_off['path_length']:>12.3f}{r_on['path_length']:>12.3f}{r_off['gt_path_length']:>12.3f}")
    d_ate = (r_on['ate_se3'] - r_off['ate_se3']) / r_off['ate_se3'] * 100
    d_rpe = (r_on['rpe1'] - r_off['rpe1']) / r_off['rpe1'] * 100
    print(f"\n  SE(3) ATE change: {d_ate:+.1f}%   |   RPE(1) change: {d_rpe:+.1f}%")
    print("""
  A loop-closure result is only a real improvement if ATE falls WITHOUT RPE
  rising and WITHOUT the path length moving away from ground truth. An ATE
  gain paid for with a large RPE regression means the pose graph is compacting
  the trajectory, not correcting drift.""")
    print("=" * 60)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "loop_closure_comparison.npz"),
             est_off=est_off, gt_off=gt_off, est_on=est_on, gt_on=gt_on,
             n_loops_on=n_loops_on, **{f"off_{k}": v for k, v in r_off.items()},
             **{f"on_{k}": v for k, v in r_on.items()})
    print(f"\nResults saved to {out_dir}/loop_closure_comparison.npz")


if __name__ == "__main__":
    main()
