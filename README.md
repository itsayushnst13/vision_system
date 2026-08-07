# vision_system — Visual Odometry & Loop Closure

A consolidated, debugged, and benchmarked RGB-D visual SLAM pipeline, built
from four earlier prototype repositories. Everything here is evaluated
against real ground truth on the **TUM RGB-D `freiburg1_xyz`** sequence.

**Headline results** (full detail and figures in [RESULTS.md](RESULTS.md)):

| | |
|---|---|
| Visual odometry, ATE RMSE (Sim(3)-aligned) | **0.2324 m** over 796 frames |
| Visual odometry, RPE RMSE (delta = 1 frame) | **0.0260 m** |
| Loop-closure classifier, held-out accuracy | **0.940** (precision 0.921 / recall 0.935 on the loop class) |
| Loop closure in the live pipeline | **-53.5%** trajectory drift (ATE unaligned 0.6517 m -> 0.3029 m) |

![Loop closure effect](results/figures/04_loop_closure_effect.png)

---

## What this project is

The four original repositories contained a mix of working prototype code
and untested, AI-generated scaffolding. This repo consolidates them into
one pipeline that actually runs, and — more importantly — **measures it**.
Doing so surfaced several real defects that were invisible without a
ground-truth benchmark:

- Three tracking bugs, one of which caused unbounded trajectory divergence
  (ATE > 10^6 m).
- A loop-closure classifier trained entirely on **synthetic** feature
  vectors with heuristic labels, which detects **zero** genuine loop
  closures on real data.
- A rank-deficient pose-graph formulation whose solver could wander
  arbitrarily far along an unconstrained null space.

Each was found, fixed, and the before/after quantified. The project is
therefore as much a **failure analysis** as an implementation — the
measurement is the contribution, and the results section is written to be
checkable rather than flattering.

---

## Quick start

```bash
git clone https://github.com/itsayushnst13/vision_system.git
cd vision_system

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Download the TUM RGB-D `freiburg1_xyz` sequence from
[vision.in.tum.de](https://vision.in.tum.de/data/datasets/rgbd-dataset/download)
and extract it so this path exists:

```
data/rgbd_dataset_freiburg1_xyz/
├── rgb/            rgb.txt
├── depth/          depth.txt
└── groundtruth.txt
```

The dataset is ~460 MB and is deliberately **not** committed to the repo.

Then reproduce every reported number and figure:

```bash
python run_all.py              # full pipeline (~10 min)
python run_all.py --skip-train # reuse existing classifier artifacts
```

Or run individual stages:

```bash
python evaluation/evaluate_tum.py                     # VO benchmark
python evaluation/retrain_loop_closure_real_labels.py # classifier training
python evaluation/evaluate_tum_with_loop_closure.py   # loop closure ON vs OFF
python evaluation/make_figures.py                     # figures only
```

---

## The three findings

### 1. Tracking bugs (visual odometry)

| Bug | Effect before fix |
|---|---|
| `solvePnP` with no outlier rejection | One bad correspondence -> wild pose jump propagating through the whole trajectory (ATE > 10^6 m) |
| Pose composed in the wrong direction | `solvePnP`/`recoverPose` return previous->current; code composed it without inverting |
| Essential-matrix fallback used unit-scale translation | `recoverPose` returns direction only; raw use injected ~1.0 m jumps into ~0.02 m/frame motion |

Post-fix, no catastrophic jumps remain and RPE at delta=1 is 0.026 m
against a ~0.023 m ground-truth step. Residual drift is *architectural* —
frame-to-frame VO without a global back-end — not a remaining bug, which
is what motivates the loop-closure work.

### 2. The classifier had never seen a real loop closure

It was trained on `np.random.normal` feature vectors with labels from a
hand-written formula. Retrained on labels derived from ground truth
(positive = within 0.08 m *and* >5 s apart; hard negatives from the
ambiguous 0.12–0.35 m band):

| Model | Accuracy | ROC-AUC | Recall (loop) |
|---|---|---|---|
| Original, synthetic labels | 0.547 | 0.582 | **0.000** |
| Retrained, real labels | **0.940** | **0.989** | 0.935 |

An earlier, easier version of this benchmark flattered *both* models and is
documented in RESULTS.md rather than quietly replaced — the retrained model
scored a perfect 1.000 on it and still failed in the live pipeline.

### 3. The pose graph was rank-deficient

Odometry residuals constrain *relative* positions; loop residuals constrain
*differences*. Both are invariant to translating the whole trajectory — a
3-DOF gauge freedom leaving the solve rank-deficient, which is why it
proposed corrections of hundreds of metres on a ~2 m trajectory. Anchoring
the first keyframe removes the null space. Only after this fix does closing
loops actually help: **-53.5% ATE**.

---

## Repository layout

```
vision_system/
├── run_all.py                    # reproduce every number and figure
├── requirements.txt
├── RESULTS.md                    # full metrics, figures, caveats
├── src/
│   ├── system.py                 # ORBSlam3 — tracking + keyframes + loop closure
│   ├── tracking.py               # feature tracking & pose estimation
│   ├── live_loop_closure.py      # trained classifier, wrapped for live use
│   ├── pose_graph_optimizer.py   # pose-graph relaxation + safety guard
│   ├── run_slam.py               # interactive runner (webcam / video / images)
│   ├── main.py                   # RGB-D dataset runner with 3D visualisation
│   ├── mapping.py                # STUB — not wired into the pipeline
│   ├── cnn_loop_closure.py       # LEGACY, non-functional — see note in file
│   └── random_forest_loop_closure.py  # LEGACY, non-functional — see note in file
├── evaluation/
│   ├── metrics.py                # ATE, RPE, Umeyama alignment
│   ├── evaluate_tum.py           # VO benchmark
│   ├── retrain_loop_closure_real_labels.py
│   ├── evaluate_tum_with_loop_closure.py
│   └── make_figures.py
├── config/camera_config.yaml
├── results/                      # metrics (.npz), models (.joblib), figures/
└── data/                         # dataset, not committed — see Quick start
```

---

## Known limitations

Stated plainly, because the point of this repo is that the numbers are
trustworthy:

- **Loop-closure false positives are tolerated, not solved.** 16 closures
  were flagged over 133 frames and most are probably wrong. The trajectory
  improves anyway because odometry residuals dominate the relaxation. The
  visible straight-line segments in the ON trajectory are these false
  positives.
- **The pose graph corrects translation only** — rotation is left as the
  tracker estimated it.
- **Results are from one sequence.** Real measurements, but not evidence of
  generality.
- **The classifier omits the CNN (ResNet18) similarity feature**, because
  PyTorch was unavailable in the evaluation environment. Both the synthetic
  and retrained models use the same 8 SIFT-based features, so the
  comparison stays like-for-like.
- **`src/mapping.py` is a stub** (`_triangulate_new_points` and
  `_local_bundle_adjustment` are `pass`) and nothing instantiates it.
- **Two legacy modules are non-functional.** `cnn_loop_closure.py` and
  `random_forest_loop_closure.py` inherit from a base class that lived in
  the original standalone repo and was never merged during consolidation.
  They are kept as reference implementations of the synthetic-label
  classifier analysed above; see the docstring in each file.

## Next steps

1. Threshold the classifier for the true deployment base rate (few genuine
   closures per many queried candidates) rather than the training balance.
2. Require N-consecutive-frame consistency before accepting a closure —
   standard production-SLAM practice.
3. Extend the pose graph to correct rotation as well as translation.
4. Validate the -53.5% result on additional sequences.
5. Restore the CNN similarity feature once PyTorch is available and re-run
   the synthetic-vs-real comparison with it included.
6. Implement or remove `src/mapping.py`.

## Data & attribution

Evaluated on the TUM RGB-D benchmark (Sturm et al., *A Benchmark for the
Evaluation of RGB-D SLAM Systems*, IROS 2012), available from the
[Computer Vision Group, TU Munich](https://vision.in.tum.de/data/datasets/rgbd-dataset).
The dataset is not redistributed here.
