"""
Reproduce every number and figure reported in README.md / RESULTS.md.

Usage:
    python run_all.py            # full pipeline
    python run_all.py --skip-train   # reuse existing classifier artifacts

Stages, in dependency order:
  1. Visual odometry evaluation on TUM RGB-D  -> results/tum_eval.npz
  2. Loop-closure classifier retraining        -> results/rf_*REAL_labels*.joblib
  3. Live loop-closure ON vs OFF comparison    -> results/loop_closure_comparison.npz
  4. Figures                                   -> results/figures/*.png

Stage 3 depends on the artifacts from stage 2, and stage 4 on all of the
above, which is why this runs them in sequence rather than independently.

Requires the TUM RGB-D freiburg1_xyz sequence under data/ -- see README.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "evaluation")
DATA = os.path.join(HERE, "data", "rgbd_dataset_freiburg1_xyz")

STAGES = [
    ("Visual odometry evaluation", "evaluate_tum.py", False),
    ("Loop-closure classifier retraining", "retrain_loop_closure_real_labels.py", True),
    ("Live loop-closure ON vs OFF comparison", "evaluate_tum_with_loop_closure.py", False),
    ("Figure generation", "make_figures.py", False),
]


def check_dataset():
    if not os.path.isdir(DATA):
        print(f"ERROR: dataset not found at {os.path.relpath(DATA, HERE)}")
        print("\nDownload the TUM RGB-D freiburg1_xyz sequence:")
        print("  https://vision.in.tum.de/data/datasets/rgbd-dataset/download")
        print(f"and extract it so that this path exists:\n  {DATA}")
        return False
    required = ["rgb.txt", "depth.txt", "groundtruth.txt", "rgb", "depth"]
    missing = [r for r in required if not os.path.exists(os.path.join(DATA, r))]
    if missing:
        print(f"ERROR: dataset at {DATA} is missing: {', '.join(missing)}")
        return False
    return True


def run_stage(title, script):
    path = os.path.join(EVAL, script)
    print("\n" + "=" * 70)
    print(f"  {title}")
    print(f"  ({os.path.relpath(path, HERE)})")
    print("=" * 70)
    t0 = time.time()
    result = subprocess.run([sys.executable, path], cwd=HERE)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  STAGE FAILED (exit {result.returncode}) after {elapsed:.1f}s")
        return False
    print(f"\n  stage completed in {elapsed:.1f}s")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse existing classifier artifacts instead of retraining")
    args = ap.parse_args()

    if not check_dataset():
        return 1

    t0 = time.time()
    for title, script, is_training in STAGES:
        if is_training and args.skip_train:
            print(f"\n[skipped] {title} (--skip-train)")
            continue
        if not run_stage(title, script):
            print("\nAborting: a stage failed. Fix the error above and re-run.")
            return 1

    print("\n" + "=" * 70)
    print(f"  All stages completed in {time.time() - t0:.1f}s")
    print("=" * 70)
    print("\n  Metrics:  results/*.npz  (and printed above)")
    print("  Figures:  results/figures/*.png")
    print("  Summary:  RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
