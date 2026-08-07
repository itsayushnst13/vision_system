"""
Trajectory evaluation metrics (shared by all evaluation scripts).

Implements the two standard metrics used in the TUM RGB-D benchmark:

  ATE (Absolute Trajectory Error) -- global consistency. The estimated
      trajectory is first aligned to ground truth (Umeyama), then the
      per-pose Euclidean distances are summarised. Sensitive to
      accumulated drift, which is what loop closure is meant to fix.

  RPE (Relative Pose Error) -- local accuracy. Compares relative motion
      over a fixed frame delta, so it is largely insensitive to global
      drift and instead measures per-step tracking quality. Reported here
      because ATE alone cannot distinguish "good tracking + drift" from
      "bad tracking".

Reporting both matters for this project: the fixes to tracking.py should
show up in RPE, while the loop-closure work should show up in ATE.
"""
import numpy as np


def align(est, gt, with_scale=True):
    """
    Umeyama alignment of `est` onto `gt`.

    Args:
        est: (N,3) estimated positions
        gt:  (N,3) ground-truth positions
        with_scale: if True, solve for a similarity transform Sim(3)
            (rotation + translation + uniform scale). If False, solve for
            a rigid transform SE(3) only, leaving scale error visible.

    Returns:
        (aligned_est, scale)
    """
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)

    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    est_c = est - mu_est
    gt_c = gt - mu_gt

    if with_scale:
        scale = np.sqrt((gt_c ** 2).sum()) / (np.sqrt((est_c ** 2).sum()) + 1e-12)
    else:
        scale = 1.0

    H = (est_c * scale).T @ gt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = mu_gt - R @ (mu_est * scale)
    aligned = (R @ (est * scale).T).T + t
    return aligned, scale


def compute_ate(est, gt, with_scale=True):
    """
    Absolute Trajectory Error after alignment.

    Returns a dict with rmse / mean / median / std / min / max (metres)
    and the recovered scale factor.
    """
    aligned, scale = align(est, gt, with_scale=with_scale)
    err = np.linalg.norm(aligned - gt, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "std": float(np.std(err)),
        "min": float(np.min(err)),
        "max": float(np.max(err)),
        "scale": float(scale),
        "n": int(len(err)),
    }


def compute_rpe(est, gt, delta=1):
    """
    Relative Pose Error (translation part) over a fixed frame delta.

    For each i, compares the estimated displacement between frames i and
    i+delta against the ground-truth displacement over the same interval.
    Because it only looks at relative motion, RPE is not affected by
    accumulated global drift -- it isolates per-step tracking quality.

    Note: this is the translation-magnitude form of RPE. A full SE(3) RPE
    would also report a rotational component; poses here are evaluated on
    position only, consistent with the rest of this repo (the pose-graph
    correction is translation-only). Documented rather than implied.

    Returns a dict with rmse / mean / median / std (metres per `delta`
    frames), or None if the trajectory is too short.
    """
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    n = min(len(est), len(gt))
    if n <= delta:
        return None

    est_d = est[delta:n] - est[:n - delta]
    gt_d = gt[delta:n] - gt[:n - delta]
    err = np.linalg.norm(est_d - gt_d, axis=1)

    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "std": float(np.std(err)),
        "delta": int(delta),
        "n": int(len(err)),
    }


def trajectory_length(xyz):
    """Total path length travelled (metres)."""
    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))


def max_step(xyz):
    """Largest single-frame displacement (metres). Useful for spotting
    catastrophic pose jumps of the kind the RANSAC/scale fixes removed."""
    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
