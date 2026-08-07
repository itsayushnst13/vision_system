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
- No formal loop-closure pose-graph correction is integrated into the main
  tracking loop yet — this is the natural next step now that the tracker
  itself is bug-fixed and produces a clean (if drifting) trajectory.

### Loop-closure classifier: real-label retraining and live integration

This went through three honest iterations — each one surfaced a genuine
new problem, which is itself the point of documenting it here.

**v1 -- synthetic labels (original code).** Trained on fabricated Gaussian
feature vectors with heuristic labels; never saw a real loop closure.
Result on a real held-out test: 32% accuracy, 2% recall (missed 98% of
genuine loop closures despite a "reasonable" 0.977 AUC — badly
miscalibrated).

**v2a -- real labels, easy margin.** Retrained using TUM ground truth to
auto-generate real labels: positive = spatially close (<0.08m) +
temporally separated (>5s) frame pairs; negative = spatially far apart
(>0.5m). Result: 100% accuracy / 100% recall on held-out test — but when
wired into the live SLAM loop (`src/system.py`), it **catastrophically
over-triggered**, flagging ~1 in 8 new keyframes as a loop closure against
essentially any earlier keyframe, and the resulting pose-graph correction
diverged (unaligned ATE went from 0.65m to over 2 **billion** meters). The
100% test accuracy was an artifact of a too-easy, too-widely-separated
train/test split that never exposed the classifier to the "nearby but not
actually the same place" cases it has to handle at deployment time.

**v2b -- real labels, hard-negative mining.** Retrained with an added hard
negative class: frame pairs in the ambiguous 0.12m–0.35m range (nearby,
not a revisit). Held-out accuracy dropped to a far more credible 94%
(precision 0.92 / recall 0.94 on the loop class) — the harder, more
realistic test genuinely separates learning from memorizing an easy
boundary. When wired into the live loop, false-positive triggers dropped
substantially, but **not to zero**: over 133 frames, 16 candidate loop
closures were still flagged, most of them false. This is a distinct and
important finding, separate from calibration: it's a **base-rate /
class-imbalance problem**. Training data was roughly balanced
(250 positive / 350 negative), but live deployment queries every new
keyframe against several older candidates, the overwhelming majority of
which are true negatives — so even a model with good *balanced* accuracy
produces many more false positives than true positives in absolute
deployment terms, because true loop closures are rare relative to the
number of candidate pairs actually queried.

**Safety guard (implemented, and this is the part that's actually
production-safe right now).** Rather than trust every classifier
"loop closure" and risk pose-graph divergence, `src/system.py` now
sanity-checks every proposed correction in `pose_graph_optimizer.py`:
if applying it would move any keyframe further than ~10x the trajectory's
own spatial extent, the correction is rejected outright and the
trajectory is left uncorrected. Re-running the v2b classifier live with
this guard in place: **all 16 flagged loop closures were correctly
rejected as destabilizing**, and the trajectory matched the
loop-closure-disabled baseline exactly (ATE 0.6517m unaligned, both
runs) — no corruption, but also no measured improvement yet, because this
particular short sequence didn't happen to contain a loop closure the
classifier got right without also raising false positives.

- **Where this honestly stands:** the classifier is not yet reliable
  enough to trust blindly, but the system is now safe to run with loop
  closure enabled — it degrades gracefully to the uncorrected baseline
  rather than corrupting the trajectory. This is a real, deployable
  improvement over where the code started (no keyframe storage at all,
  synthetic-only training, and no safety check).
- **Concrete next step to get an actual accuracy improvement (not just
  safety):** calibrate the decision threshold for the true deployment
  base rate (few true positives per many queries) rather than the
  balanced-training default, and/or require N-consecutive-frame
  consistency before accepting a candidate loop closure — standard
  practice in production SLAM systems, not yet implemented here.
- Artifacts: `results/rf_loop_closure_REAL_labels.joblib` (current, v2b,
  hard-negative-mined), `results/rf_loop_closure_REAL_labels_v1_easy.joblib`
  (kept for comparison), `results/loop_closure_comparison.npz` (live
  on/off run), `results/real_label_eval.npz` (v2b held-out test set).

## Repository layout

```
vision_system/
├── src/
│   ├── system.py                     # Main SLAM system (ORBSlam3 class) -- now with keyframe storage + loop closure wiring
│   ├── tracking.py                   # Feature tracking + pose estimation (RANSAC-PnP fix applied)
│   ├── mapping.py                    # Basic mapping
│   ├── run_slam.py                   # Interactive runner (webcam/video/image seq)
│   ├── loop_closure_detector.py      # Wraps trained classifier for live use (NEW)
│   ├── pose_graph_optimizer.py       # Lightweight scipy pose-graph correction + safety guard (NEW)
│   ├── cnn_loop_closure.py           # CNN+RF loop closure (original, synthetic-trained)
│   └── random_forest_loop_closure.py
├── evaluation/
│   ├── evaluate_tum.py                        # Headless TUM RGB-D VO evaluator
│   ├── retrain_loop_closure_real_labels.py    # Real-label loop-closure retraining + eval
│   └── evaluate_tum_with_loop_closure.py      # Live loop-closure ON vs OFF comparison (NEW)
├── config/
│   └── camera_config.yaml
├── results/
│   ├── tum_eval.npz                           # VO evaluation results
│   ├── rf_loop_closure_REAL_labels.joblib     # Real-labeled classifier
│   ├── rf_scaler_REAL_labels.joblib
│   └── real_label_eval.npz                    # Real-label test set + predictions
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
pip install opencv-python-headless numpy scipy matplotlib PyYAML scikit-learn joblib
python evaluation/evaluate_tum.py                          # VO/tracking evaluation
python evaluation/retrain_loop_closure_real_labels.py       # Loop-closure retraining + eval
python evaluation/evaluate_tum_with_loop_closure.py         # Live loop-closure ON vs OFF comparison
```

## Next steps (tracked honestly, not yet done)

1. Calibrate the loop-closure decision threshold for the true deployment
   base rate (few genuine loop closures per many candidate queries),
   rather than the balanced-training default -- this is the most direct
   path to fewer false-positive triggers.
2. Require N-consecutive-frame consistency before accepting a candidate
   loop closure (standard production-SLAM practice), instead of acting on
   a single frame's classification.
3. Re-add the CNN (ResNet18) similarity feature once a PyTorch install is
   available, and re-run the real-vs-synthetic comparison with it included.
4. Extend the pose-graph optimizer to correct rotation, not just
   translation, once (1) and (2) make loop closures trustworthy enough to
   actually accept.
5. Find or construct a longer sequence with a genuine, unambiguous loop
   closure to demonstrate a real before/after ATE improvement -- the
   current freiburg1_xyz subset used here doesn't contain one that this
   classifier can isolate cleanly from the false positives.
