"""
Trajectory evaluation metrics (shared by all evaluation scripts).

Implements the two standard TUM RGB-D metrics, plus the naive baselines that
make them interpretable.

  ATE (Absolute Trajectory Error) -- global consistency, after aligning the
      estimate to ground truth.

  RPE (Relative Pose Error) -- local accuracy over a fixed frame gap, largely
      insensitive to accumulated drift.

WHY BASELINES ARE REPORTED ALONGSIDE
------------------------------------
ATE and RPE are error magnitudes in metres. On a sequence with small motion
they can look small while carrying no information at all, because two trivial
estimators already score well:

  * static-point estimator -- output one fixed position for every frame.
    Its ATE equals the RMS spread of the ground truth about its centroid.
    Any real system must beat this or it has learned nothing about motion.

  * null-motion estimator -- output zero displacement between every frame
    pair. Its RPE equals the RMS ground-truth displacement over that gap.
    Any real tracker must beat this or it is worse than assuming the camera
    never moved.

Every metric function below therefore also returns the corresponding baseline
and the ratio to it. A ratio < 1.0 means the system beats the trivial
estimator; >= 1.0 means it does not. Reporting the raw metre value alone is
how a non-functional tracker gets mistaken for a working one.
"""
import numpy as np


def align(est, gt, with_scale=True):
    """
    Umeyama alignment of `est` onto `gt`.

    Args:
        est: (N,3) estimated positions
        gt:  (N,3) ground-truth positions
        with_scale: if True, solve the full similarity transform Sim(3)
            (rotation + translation + uniform scale). If False, solve the
            rigid transform SE(3) only, leaving scale error visible.

    Returns:
        (aligned_est, scale)

    Note on the scale term: the least-squares-optimal similarity scale is
        s = trace(D @ S) / sum(est_centred**2)
    where S holds the singular values of the cross-covariance and D is the
    reflection-correcting diagonal (Umeyama 1991, eq. 41). An earlier version
    of this file used the RMS-ratio ||gt_c|| / ||est_c|| instead, which
    equalises the spread of the two clouds rather than minimising the aligned
    error. That is not the minimiser, so it reports an ATE strictly >= the
    true Sim(3) ATE.
    """
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)

    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    est_c = est - mu_est
    gt_c = gt - mu_gt

    H = est_c.T @ gt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    if with_scale:
        var_est = (est_c ** 2).sum()
        scale = float((S * np.array([1.0, 1.0, d])).sum() / (var_est + 1e-12))
    else:
        scale = 1.0

    t = mu_gt - R @ (mu_est * scale)
    aligned = (R @ (est * scale).T).T + t
    return aligned, scale


def static_point_ate(gt):
    """
    ATE of the best possible constant-position estimator.

    The optimal constant is the ground-truth centroid, so this equals the RMS
    spread of the ground truth. This is the floor any trajectory estimate must
    beat to be carrying information about motion.
    """
    gt = np.asarray(gt, dtype=float)
    return float(np.sqrt((np.linalg.norm(gt - gt.mean(axis=0), axis=1) ** 2).mean()))


def null_motion_rpe(gt, delta=1):
    """
    RPE of the estimator that always reports zero displacement.

    Equals the RMS ground-truth displacement over `delta` frames. Any tracker
    scoring above this is worse than assuming the camera is stationary.
    """
    gt = np.asarray(gt, dtype=float)
    if len(gt) <= delta:
        return None
    d = np.linalg.norm(gt[delta:] - gt[:-delta], axis=1)
    return float(np.sqrt((d ** 2).mean()))


def compute_ate(est, gt, with_scale=True):
    """
    Absolute Trajectory Error after alignment.

    Returns rmse / mean / median / std / min / max (metres), the recovered
    scale, the static-point baseline, and the ratio of RMSE to that baseline.
    """
    aligned, scale = align(est, gt, with_scale=with_scale)
    err = np.linalg.norm(aligned - gt, axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    baseline = static_point_ate(gt)
    return {
        "rmse": rmse,
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "std": float(np.std(err)),
        "min": float(np.min(err)),
        "max": float(np.max(err)),
        "scale": float(scale),
        "n": int(len(err)),
        "baseline_static_point": baseline,
        "ratio_to_baseline": float(rmse / (baseline + 1e-12)),
    }


def compute_rpe(est, gt, delta=1):
    """
    Relative Pose Error (translation) over a fixed frame delta.

    Returns rmse / mean / median / std (metres per `delta` frames), the
    null-motion baseline, and the ratio of RMSE to it. None if too short.

    This is the translation-magnitude form of RPE; a full SE(3) RPE would also
    report a rotational component. Poses are evaluated on position only, which
    is consistent with the translation-only pose-graph correction.
    """
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    n = min(len(est), len(gt))
    if n <= delta:
        return None

    est_d = est[delta:n] - est[:n - delta]
    gt_d = gt[delta:n] - gt[:n - delta]
    err = np.linalg.norm(est_d - gt_d, axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    baseline = null_motion_rpe(gt[:n], delta=delta)

    return {
        "rmse": rmse,
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "std": float(np.std(err)),
        "delta": int(delta),
        "n": int(len(err)),
        "baseline_null_motion": baseline,
        "ratio_to_baseline": float(rmse / (baseline + 1e-12)) if baseline else None,
    }


def drift_decomposition(est, gt):
    """
    Separate per-step error into a constant bias and zero-mean noise, then
    check which one explains how RPE grows with the frame gap.

    Rationale: for genuinely unbiased per-step errors, RPE grows as sqrt(delta)
    (a random walk). For a constant bias it grows linearly. Observed growth
    that is linear while the measured constant bias is too small to explain it
    indicates temporally CORRELATED error -- a slowly-varying systematic fault
    (typically rotation or an uncorrected lens model), not an inherent limit of
    frame-to-frame odometry.

    Returns a dict with the bias vector, noise sigma, and observed vs predicted
    RPE under each model.
    """
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    e = np.diff(est, axis=0) - np.diff(gt, axis=0)
    bias = e.mean(axis=0)
    bias_mag = float(np.linalg.norm(bias))
    sigma = float(np.sqrt(((e - bias) ** 2).sum(axis=1).mean()))

    rows = []
    for delta in (1, 10, 30):
        r = compute_rpe(est, gt, delta=delta)
        if r is None:
            continue
        rows.append({
            "delta": delta,
            "observed": r["rmse"],
            "pred_random_walk": float(sigma * np.sqrt(delta)),
            "pred_constant_bias": float(np.sqrt((delta * bias_mag) ** 2 + delta * sigma ** 2)),
        })
    return {"bias_vector": bias, "bias_magnitude": bias_mag, "noise_sigma": sigma, "growth": rows}


def trajectory_length(xyz):
    """
    Total path length travelled (metres).

    Caution when comparing to ground truth: path length is a POSITIVELY BIASED
    statistic under per-step noise, because summing ||step|| adds noise
    magnitude regardless of sign. A noisy estimate of a short trajectory will
    always look longer than the truth. Do not read an inflated path length as
    evidence of a scale error on its own -- check the per-step ratio and the
    Sim(3) scale factor too.
    """
    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))


def max_step(xyz):
    """Largest single-frame displacement (metres). Spots catastrophic jumps."""
    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))


def summarize(est, gt, label="", deltas=(1, 10, 30)):
    """One-call evaluation returning every metric plus its baseline."""
    out = {
        "label": label,
        "ate_sim3": compute_ate(est, gt, with_scale=True),
        "ate_se3": compute_ate(est, gt, with_scale=False),
        "rpe": {d: compute_rpe(est, gt, delta=d) for d in deltas},
        "est_path_length": trajectory_length(est),
        "gt_path_length": trajectory_length(gt),
        "est_max_step": max_step(est),
        "gt_max_step": max_step(gt),
        "gt_mean_step": float(np.linalg.norm(np.diff(gt, axis=0), axis=1).mean()),
        "est_mean_step": float(np.linalg.norm(np.diff(est, axis=0), axis=1).mean()),
        "drift": drift_decomposition(est, gt),
    }
    return out


def format_summary(s):
    """Render summarize() output as a text block, baselines included."""
    L = []
    a3, a6 = s["ate_sim3"], s["ate_se3"]
    L.append(f"  Poses evaluated:        {a3['n']}")
    L.append(f"  GT mean step:           {s['gt_mean_step']:.4f} m   "
             f"(GT max step {s['gt_max_step']:.4f} m)")
    L.append(f"  Est mean step:          {s['est_mean_step']:.4f} m   "
             f"(Est max step {s['est_max_step']:.4f} m)")
    L.append(f"  Path length est / gt:   {s['est_path_length']:.3f} / {s['gt_path_length']:.3f} m")
    L.append("")
    L.append("  --- ATE ---")
    L.append(f"  static-point baseline:  {a3['baseline_static_point']:.4f} m  <-- must beat this")
    L.append(f"  Sim(3) aligned RMSE:    {a3['rmse']:.4f} m  "
             f"(scale {a3['scale']:.4f}, {a3['ratio_to_baseline']:.2f}x baseline)")
    L.append(f"  SE(3)  aligned RMSE:    {a6['rmse']:.4f} m  "
             f"({a6['ratio_to_baseline']:.2f}x baseline)")
    L.append("")
    L.append("  --- RPE (translation) ---")
    for d, r in s["rpe"].items():
        if r:
            L.append(f"  delta={d:>3}: RMSE {r['rmse']:.4f} m | "
                     f"null-motion baseline {r['baseline_null_motion']:.4f} m | "
                     f"{r['ratio_to_baseline']:.2f}x baseline")
    L.append("")
    L.append("  --- drift model ---")
    dr = s["drift"]
    L.append(f"  per-step bias |b| = {dr['bias_magnitude']:.5f} m, noise sigma = {dr['noise_sigma']:.5f} m")
    for g in dr["growth"]:
        L.append(f"  delta={g['delta']:>3}: observed {g['observed']:.4f} | "
                 f"random-walk predicts {g['pred_random_walk']:.4f} | "
                 f"constant-bias predicts {g['pred_constant_bias']:.4f}")
    return "\n".join(L)
