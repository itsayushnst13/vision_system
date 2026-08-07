# vision_system

Monocular / RGB-D Visual SLAM pipeline (ORB features + PnP/Essential-Matrix
tracking), with a CNN-assisted loop closure module. Consolidated from earlier
prototypes (`DRDO`, `Loop_Closure`, `Loop_Closure_and_Algorithms`, `LC_BA`)
into a single, evaluated codebase.

## Status (honest, as of this consolidation)

This is a research prototype, not a production SLAM system. Below is exactly
what has been verified and what has not.

### Verified on real data
- Core tracking pipeline (`src/tracking.py`, `src/system.py`) runs end-to-end
  on the **TUM RGB-D `freiburg1_xyz`** benchmark (796 real frames, real
  ground truth trajectory).

- **Bug 1 (fixed): no outlier rejection in PnP.** The original tracker used
  `cv2.solvePnP` with no RANSAC. A single bad feature correspondence could
  produce a wild one-frame pose jump that then propagated through the whole
  chained trajectory. Observed effect: ATE >10^6 m (unbounded divergence).
  Fixed by switching to `cv2.solvePnPRansac`.

- **Bug 2 (fixed): pose composed in the wrong direction.** `solvePnP` /
  `recoverPose` return the transform FROM the previous camera frame TO the
  current one (`X_curr = R @ X_prev + t`), but the code composed
  `prev_pose @ transform` directly without inverting first — the wrong
  direction for accumulating a world-frame trajectory. Fixed by inverting
  `(R, t)` before composing.

- **Bug 3 (fixed): unit-scale fallback jumps.** When PnP doesn't have enough
  valid depth points, the tracker falls back to the Essential-Matrix method
  (`cv2.recoverPose`), which only recovers translation *direction*, not
  magnitude — OpenCV returns a unit-normalized vector. That was being used
  as-is, injecting an arbitrary ~1.0 m jump into a trajectory whose real
  per-frame motion is centimeter-scale, corrupting the trajectory's overall
  scale for the rest of the sequence. Fixed by rescaling the fallback
  direction using a rolling median of recent PnP-derived (metric) step
  sizes — this also wires up `recent_scales` / `median_scale`, which existed
  as unused dead code in the original `Tracker.__init__`.

- **Post-fix baseline (measured, not estimated), after all 3 fixes above:**
  - No more single-frame catastrophic jumps (max per-frame step: 0.043 m,
    consistent with the dataset's real motion).
  - **Residual issue — not a bug, an architectural limitation:** the
    trajectory still accumulates ~10x drift over the 796-frame sequence
    (estimated spatial range ~7 m vs. ground-truth range ~0.7 m). This is
    the expected behavior of frame-to-frame VO with no global back-end
    (no pose-graph optimization / bundle adjustment / loop closure
    correction). It is *not* fixable by patching `tracking.py` alone — it's
    the reason the loop-closure module exists in this repo in the first
    place (see below).
  - ATE RMSE with Sim(3) alignment (post-hoc scale+rotation correction):
    **0.235 m**
  - ATE RMSE without scale correction (raw accumulated drift):
    **2.63 m**
  - See `evaluation/evaluate_tum.py` and `results/tum_eval.npz` for the
    exact reproducible run.

### Known unresolved issues
- **Trajectory drift without loop closure (see above)** — this is the main
  remaining item, and it's architectural rather than a single bug.
- `src/cnn_loop_closure.py` + `src/random_forest_loop_closure.py`: the
  Random Forest loop-closure classifier included here was originally
  trained on **synthetically generated features/labels**, not on real
  annotated loop closures. Its outputs should **not** be treated as a
  validated loop-closure detector until retrained on real labeled pairs.
  This is flagged explicitly so it is not accidentally cited as a working
  result.
- No formal loop-closure pose-graph correction is integrated into the main
  tracking loop yet — this is the natural next step now that the tracker
  itself is bug-fixed and produces a clean (if drifting) trajectory.

## Repository layout

```
vision_system/
├── src/
│   ├── system.py                     # Main SLAM system (ORBSlam3 class)
│   ├── tracking.py                   # Feature tracking + pose estimation (RANSAC-PnP fix applied)
│   ├── mapping.py                    # Basic mapping
│   ├── run_slam.py                   # Interactive runner (webcam/video/image seq)
│   ├── cnn_loop_closure.py           # CNN+RF loop closure — UNVALIDATED, see above
│   └── random_forest_loop_closure.py
├── evaluation/
│   └── evaluate_tum.py               # Headless TUM RGB-D evaluator (source of the numbers above)
├── config/
│   └── camera_config.yaml
├── results/
│   └── tum_eval.npz                  # Raw arrays from the last verified evaluation run
├── docs/                             # Background literature / notes
└── data/                             # (not checked in — see Data section)
```

## Data

Evaluation uses the TUM RGB-D `freiburg1_xyz` sequence. Not included in this
repo due to size; download from:
https://vision.in.tum.de/data/datasets/rgbd-dataset/download
and place under `data/rgbd_dataset_freiburg1_xyz/`.

## Reproducing the evaluation numbers

```bash
python -m venv venv && source venv/bin/activate
pip install opencv-python-headless numpy scipy matplotlib PyYAML
python evaluation/evaluate_tum.py
```

## Next steps (tracked honestly, not yet done)

1. Implement pose-graph optimization / loop-closure correction to address
   the residual ~10x drift documented above (the tracker itself is now
   bug-fixed; this is the remaining structural gap).
2. Re-label a real loop-closure training set from recorded video and
   retrain the RF classifier on real (not synthetic) labels.
3. Compare before/after loop-closure integration as a quantitative result
   (this is the strongest candidate for the paper's core empirical section).
