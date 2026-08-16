"""
Controlled measurement of the distortion-handling bug.

The TUM sequence cannot be shipped with this repo, so this test isolates the
geometry from feature matching entirely: a synthetic 3D scene is projected
through the REAL freiburg1 intrinsics and distortion coefficients, with known
correspondences and known ground-truth camera poses. The only thing under test
is how pixel coordinates are converted to and from 3D.

Two pipelines are run on identical inputs:

  BUGGY  -- back-project raw (distorted) pixels with the pinhole model, and
            pass dist_coeffs to solvePnP (the original code path).
  FIXED  -- undistort pixels once, then treat the camera as an ideal pinhole
            and pass zero distortion to solvePnP.

Because both see the same correspondences and the same noise, any difference
in trajectory error is attributable to distortion handling alone.
"""
import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))

from tracking import undistort_points, backproject  # noqa: E402
from metrics import summarize, format_summary       # noqa: E402

# Real TUM freiburg1 parameters, from config/camera_config.yaml
K = np.array([[517.3, 0, 318.6],
              [0, 516.5, 255.3],
              [0, 0, 1]], dtype=np.float64)
DIST = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633], dtype=np.float64)
ZERO_DIST = np.zeros(5)
W, H = 640, 480

N_FRAMES = 400
N_POINTS = 1200
PIXEL_NOISE = 0.5      # px, comparable to ORB localisation accuracy
DEPTH_NOISE = 0.005    # m, comparable to TUM structured-light noise
RNG = np.random.default_rng(0)


def make_scene():
    """A slab of 3D points 1-4 m in front of the camera's start position."""
    x = RNG.uniform(-2.0, 2.0, N_POINTS)
    y = RNG.uniform(-1.5, 1.5, N_POINTS)
    z = RNG.uniform(1.0, 4.0, N_POINTS)
    return np.stack([x, y, z], axis=1)


def make_trajectory():
    """
    Small oscillating xyz motion with mild rotation -- the same character as
    freiburg1_xyz (sub-metre extent, ~1 cm per frame).
    """
    t = np.linspace(0, 4 * np.pi, N_FRAMES)
    pos = np.stack([0.25 * np.sin(t), 0.15 * np.sin(2 * t), 0.10 * np.sin(0.5 * t)], axis=1)
    poses = []
    for i in range(N_FRAMES):
        ang = 0.05 * np.sin(t[i])
        R, _ = cv2.Rodrigues(np.array([0.0, ang, 0.0]))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos[i]
        poses.append(T)
    return poses


def project(points_world, pose_wc):
    """
    Project world points into the image of a camera at world-from-camera pose
    `pose_wc`, applying the real distortion model. Returns (pixels, depths,
    visibility mask).
    """
    Rcw = pose_wc[:3, :3].T
    tcw = -Rcw @ pose_wc[:3, 3]
    pts_cam = (Rcw @ points_world.T).T + tcw
    in_front = pts_cam[:, 2] > 0.3
    rvec, _ = cv2.Rodrigues(Rcw)
    px, _ = cv2.projectPoints(points_world, rvec, tcw, K, DIST)
    px = px.reshape(-1, 2)
    on_image = (px[:, 0] > 5) & (px[:, 0] < W - 5) & (px[:, 1] > 5) & (px[:, 1] < H - 5)
    vis = in_front & on_image
    return px, pts_cam[:, 2], vis


def run(mode):
    """
    Chain frame-to-frame PnP over the synthetic sequence.

    mode='buggy': back-project raw pixels, pass DIST to solvePnP.
    mode='fixed': undistort first, pass ZERO_DIST to solvePnP.
    """
    scene = make_scene()
    gt_poses = make_trajectory()

    est_pose = np.eye(4)
    est_xyz, gt_xyz = [est_pose[:3, 3].copy()], [gt_poses[0][:3, 3].copy()]

    px_prev, z_prev, vis_prev = project(scene, gt_poses[0])
    px_prev = px_prev + RNG.normal(0, PIXEL_NOISE, px_prev.shape)
    z_prev = z_prev + RNG.normal(0, DEPTH_NOISE, z_prev.shape)

    for i in range(1, N_FRAMES):
        px_cur, z_cur, vis_cur = project(scene, gt_poses[i])
        px_cur = px_cur + RNG.normal(0, PIXEL_NOISE, px_cur.shape)
        z_cur = z_cur + RNG.normal(0, DEPTH_NOISE, z_cur.shape)

        both = vis_prev & vis_cur
        if both.sum() < 30:
            est_xyz.append(est_pose[:3, 3].copy())
            gt_xyz.append(gt_poses[i][:3, 3].copy())
            px_prev, z_prev, vis_prev = px_cur, z_cur, vis_cur
            continue

        p_prev = px_prev[both].astype(np.float32)
        p_cur = px_cur[both].astype(np.float32)
        d_prev = z_prev[both]

        if mode == "buggy":
            # Pinhole back-projection applied to DISTORTED pixels ...
            obj = backproject(p_prev, d_prev, K).astype(np.float32)
            img = p_cur
            dist_for_pnp = DIST          # ... while PnP models distortion.
        else:
            p_prev_u = undistort_points(p_prev, K, DIST)
            p_cur_u = undistort_points(p_cur, K, DIST)
            obj = backproject(p_prev_u, d_prev, K).astype(np.float32)
            img = p_cur_u.astype(np.float32)
            dist_for_pnp = ZERO_DIST

        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, K, dist_for_pnp, flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0, confidence=0.999, iterationsCount=200)

        if ok and inl is not None and len(inl) >= 15:
            idx = inl.reshape(-1)
            rvec, tvec = cv2.solvePnPRefineLM(obj[idx], img[idx], K, dist_for_pnp, rvec, tvec)
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3)
            T = np.eye(4)
            T[:3, :3] = R.T
            T[:3, 3] = -R.T @ t
            est_pose = est_pose @ T

        est_xyz.append(est_pose[:3, 3].copy())
        gt_xyz.append(gt_poses[i][:3, 3].copy())
        px_prev, z_prev, vis_prev = px_cur, z_cur, vis_cur

    return np.array(est_xyz), np.array(gt_xyz)


def main():
    results = {}
    for mode in ("buggy", "fixed"):
        est, gt = run(mode)
        s = summarize(est, gt, label=mode)
        results[mode] = s
        print("=" * 66)
        print(f"  SYNTHETIC SEQUENCE — distortion handling: {mode.upper()}")
        print("=" * 66)
        print(format_summary(s))
        print()

    b, f = results["buggy"], results["fixed"]
    print("=" * 66)
    print("  EFFECT OF THE DISTORTION FIX (same scene, same noise, same seed)")
    print("=" * 66)
    print(f"  ATE Sim(3):   {b['ate_sim3']['rmse']:.4f} m  ->  {f['ate_sim3']['rmse']:.4f} m"
          f"   ({100*(f['ate_sim3']['rmse']-b['ate_sim3']['rmse'])/b['ate_sim3']['rmse']:+.1f}%)")
    print(f"  ATE SE(3):    {b['ate_se3']['rmse']:.4f} m  ->  {f['ate_se3']['rmse']:.4f} m"
          f"   ({100*(f['ate_se3']['rmse']-b['ate_se3']['rmse'])/b['ate_se3']['rmse']:+.1f}%)")
    for d in (1, 10, 30):
        rb, rf = b["rpe"][d], f["rpe"][d]
        print(f"  RPE delta={d:<3} {rb['rmse']:.4f} m  ->  {rf['rmse']:.4f} m"
              f"   (vs null-motion baseline {rb['baseline_null_motion']:.4f} m;"
              f" {rb['ratio_to_baseline']:.2f}x -> {rf['ratio_to_baseline']:.2f}x)")
    print(f"  static-point ATE baseline: {b['ate_sim3']['baseline_static_point']:.4f} m")

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    np.savez(os.path.join(os.path.dirname(__file__), "..", "results", "synthetic_distortion.npz"),
             buggy_ate_sim3=b["ate_sim3"]["rmse"], fixed_ate_sim3=f["ate_sim3"]["rmse"],
             buggy_ate_se3=b["ate_se3"]["rmse"], fixed_ate_se3=f["ate_se3"]["rmse"],
             buggy_rpe1=b["rpe"][1]["rmse"], fixed_rpe1=f["rpe"][1]["rmse"],
             null_rpe1=b["rpe"][1]["baseline_null_motion"],
             static_ate=b["ate_sim3"]["baseline_static_point"])
    return results


if __name__ == "__main__":
    main()
