"""
Lightweight pose-graph relaxation (translation only).

GTSAM is not a dependency here, so this is a minimal least-squares relaxation
over 3D keyframe positions using scipy. Rotation is not optimised.

Model
-----
  odometry edge (k, k+1): preserve the relative translation the tracker
      measured between consecutive keyframes.
  loop edge (i, j):       place keyframe j at the position implied by
      keyframe i plus the MEASURED relative translation between them.
  anchor:                 pin keyframe 0 to its original position.

WHY LOOP EDGES CARRY A MEASUREMENT
----------------------------------
An earlier version used  residual = w * (p[j] - p[i]),  i.e. it forced the two
keyframes to occupy the SAME position. That is not what a loop closure means.
A loop closure says the two frames observe the same place; the camera is
nearby, not identical, and typically differs by tens of centimetres and a
rotation. Forcing coincidence at high weight compacts the trajectory
artificially: it improves global ATE (which rewards a trajectory that stays
near the ground-truth centroid) while destroying local accuracy, and the
"straight line segments" it produces are the optimiser dragging keyframes on
top of each other.

Loop edges therefore now take a measured relative translation, obtained by
geometric verification between the two keyframes (see
live_loop_closure.estimate_relative_pose). If a measurement is unavailable the
edge is dropped rather than silently degraded into a coincidence constraint.

WHY THE ANCHOR IS NEEDED
------------------------
Odometry residuals constrain relative positions and loop residuals constrain
differences of positions. Both are invariant to translating the whole
trajectory by a constant vector: a 3-DOF gauge freedom that leaves the normal
equations rank-deficient, letting the solver wander arbitrarily far along the
null space at almost no cost. Pinning keyframe 0 removes it.
"""
import numpy as np
from scipy.optimize import least_squares


class PoseGraphOptimizer:
    def __init__(self, odometry_weight: float = 1.0, loop_weight: float = 3.0,
                 anchor_weight: float = 10.0):
        """
        Args:
            odometry_weight: weight on preserving measured consecutive deltas.
            loop_weight: weight on loop-closure edges. Kept modest (3x, not the
                previous 15x) because a loop measurement derived from a small
                number of feature matches is less reliable than the odometry it
                is correcting, and because an over-weighted loop edge is what
                produces the compaction artefact described above.
            anchor_weight: weight pinning keyframe 0.
        """
        self.odometry_weight = odometry_weight
        self.loop_weight = loop_weight
        self.anchor_weight = anchor_weight

    def optimize(self, positions: np.ndarray, loop_pairs: list,
                 loop_measurements: list = None):
        """
        Args:
            positions: (N,3) keyframe positions from raw tracking.
            loop_pairs: list of (i, j) index pairs flagged as loop closures.
            loop_measurements: list of (3,) measured translations from
                keyframe i to keyframe j, in world frame, one per pair. If
                None, all loop edges are dropped and the input is returned
                unchanged -- an unmeasured loop closure carries no usable
                geometric information.

        Returns:
            (N,3) corrected positions.
        """
        positions = np.asarray(positions, dtype=float)
        n = len(positions)
        if n < 2 or not loop_pairs:
            return positions.copy()

        if loop_measurements is None or len(loop_measurements) != len(loop_pairs):
            # Refuse to fabricate a coincidence constraint.
            return positions.copy()

        pairs, meas = [], []
        for (i, j), m in zip(loop_pairs, loop_measurements):
            if m is None:
                continue
            pairs.append((i, j))
            meas.append(np.asarray(m, dtype=float))
        if not pairs:
            return positions.copy()

        idx_i = np.array([i for i, _ in pairs])
        idx_j = np.array([j for _, j in pairs])
        meas = np.array(meas)

        odom_targets = np.diff(positions, axis=0)
        anchor = positions[0].copy()
        x0 = positions.flatten()

        def residuals(x):
            p = x.reshape(n, 3)
            odom_res = self.odometry_weight * (np.diff(p, axis=0) - odom_targets)
            loop_res = self.loop_weight * ((p[idx_j] - p[idx_i]) - meas)
            anchor_res = self.anchor_weight * (p[0] - anchor)
            return np.concatenate([odom_res.ravel(), loop_res.ravel(), anchor_res.ravel()])

        # 'trf' rather than 'lm': robust when residual count differs from
        # parameter count, and better behaved on near-degenerate problems.
        result = least_squares(residuals, x0, method="trf", max_nfev=2000)
        return result.x.reshape(n, 3)
