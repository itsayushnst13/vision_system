"""
Recompute every reported figure from the committed results/*.npz arrays.

These arrays hold the trajectories produced by the ORIGINAL (pre-fix) code, so
this script does not require the TUM dataset. Its purpose is to re-derive the
published numbers with corrected metrics and with baselines attached, making
clear which of the original claims survive.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from metrics import (summarize, format_summary, align, static_point_ate,  # noqa: E402
                     null_motion_rpe, trajectory_length)

R = os.path.join(os.path.dirname(__file__), "..", "results")


def rms_ratio_align(est, gt):
    """The superseded RMS-ratio alignment, kept only to quantify its error."""
    mu_e, mu_g = est.mean(0), gt.mean(0)
    ec, gc = est - mu_e, gt - mu_g
    s = np.sqrt((gc ** 2).sum()) / (np.sqrt((ec ** 2).sum()) + 1e-12)
    H = (ec * s).T @ gc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_g - Rm @ (mu_e * s)
    return (Rm @ (est * s).T).T + t, s


def main():
    print("=" * 70)
    print("  RECOMPUTED FROM COMMITTED RESULTS (original, pre-fix trajectories)")
    print("=" * 70)

    d = np.load(os.path.join(R, "tum_eval.npz"))
    est, gt = d["est_xyz"], d["gt_xyz"]

    s = summarize(est, gt, label="VO (as published)")
    print("\n[1] Visual odometry, 796 frames")
    print(format_summary(s))

    a_old, sc_old = rms_ratio_align(est, gt)
    e_old = float(np.sqrt((np.linalg.norm(a_old - gt, axis=1) ** 2).mean()))
    print(f"\n  alignment comparison:")
    print(f"    RMS-ratio scale (published):  s={sc_old:.4f}  ATE={e_old:.4f} m")
    print(f"    least-squares optimal scale:  s={s['ate_sim3']['scale']:.4f}  "
          f"ATE={s['ate_sim3']['rmse']:.4f} m")
    print(f"    static-point baseline:                  ATE="
          f"{s['ate_sim3']['baseline_static_point']:.4f} m")

    gt_steps = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    print(f"\n  GT step statistics (the published text compared RPE against 0.023 m,")
    print(f"  which is the MAXIMUM step; the mean is what RPE should be judged against):")
    print(f"    mean {gt_steps.mean():.4f} m | median {np.median(gt_steps):.4f} m | "
          f"max {gt_steps.max():.4f} m")

    print("\n" + "-" * 70)
    print("[2] Loop closure ON vs OFF, 133 frames")
    d2 = np.load(os.path.join(R, "loop_closure_comparison.npz"))
    off, on, g = d2["est_off"], d2["est_on"], d2["gt_off"]

    rows = []
    for name, x in (("OFF", off), ("ON", on)):
        ss = summarize(x, g, label=name)
        rows.append((name, ss))
        print(f"\n  --- loop closure {name} ---")
        print(format_summary(ss))

    (_, so), (_, sn) = rows[0], rows[1]
    print("\n  side by side:")
    print(f"    {'metric':<26}{'OFF':>10}{'ON':>10}{'baseline':>12}")
    print(f"    {'ATE Sim(3)':<26}{so['ate_sim3']['rmse']:>10.4f}{sn['ate_sim3']['rmse']:>10.4f}"
          f"{so['ate_sim3']['baseline_static_point']:>12.4f}")
    print(f"    {'ATE SE(3)':<26}{so['ate_se3']['rmse']:>10.4f}{sn['ate_se3']['rmse']:>10.4f}"
          f"{so['ate_se3']['baseline_static_point']:>12.4f}")
    print(f"    {'RPE delta=1':<26}{so['rpe'][1]['rmse']:>10.4f}{sn['rpe'][1]['rmse']:>10.4f}"
          f"{so['rpe'][1]['baseline_null_motion']:>12.4f}")
    print(f"    {'path length (m)':<26}{so['est_path_length']:>10.3f}{sn['est_path_length']:>10.3f}"
          f"{so['gt_path_length']:>12.3f}")

    print("\n" + "-" * 70)
    print("[3] Classifier test sets (unchanged by these fixes)")
    for f, lbl in (("real_label_eval.npz", "hard negatives"),
                   ("real_label_eval_v1_easy.npz", "easy split")):
        dd = np.load(os.path.join(R, f))
        y = dd["y_test"]
        maj = max(np.mean(y == 0), np.mean(y == 1))
        print(f"  {lbl:<16} n={len(y):<5} accuracy {float(dd['accuracy']):.3f} | "
              f"AUC {float(dd['auc']):.3f} | majority-class baseline {maj:.3f}")

    np.savez(os.path.join(R, "recomputed_baselines.npz"),
             vo_ate_sim3=s["ate_sim3"]["rmse"],
             vo_ate_sim3_published=e_old,
             vo_ate_static_baseline=s["ate_sim3"]["baseline_static_point"],
             vo_rpe1=s["rpe"][1]["rmse"],
             vo_rpe1_baseline=s["rpe"][1]["baseline_null_motion"],
             gt_mean_step=float(gt_steps.mean()), gt_max_step=float(gt_steps.max()),
             lc_off_rpe1=so["rpe"][1]["rmse"], lc_on_rpe1=sn["rpe"][1]["rmse"],
             lc_off_path=so["est_path_length"], lc_on_path=sn["est_path_length"],
             lc_gt_path=so["gt_path_length"])
    print("\nSaved to results/recomputed_baselines.npz")


if __name__ == "__main__":
    main()
