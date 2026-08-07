"""
Lightweight Pose Graph Optimizer (translation-only)
====================================================
No GTSAM available in this environment, so this implements a minimal
least-squares pose graph relaxation using scipy, operating on 3D keyframe
positions only (rotation is not optimized).

Model:
  - Odometry edges: consecutive keyframes should preserve their original
    relative translation (soft constraint, moderate weight).
  - Loop-closure edges: revisited keyframe pairs should end up at (nearly)
    the same 3D position (strong constraint, high weight).

This is a deliberately simple technique -- a full SE(3) pose graph with
proper information matrices (as GTSAM would give) is out of scope here.
It is enough to demonstrate, quantitatively, whether closing a detected
loop reduces trajectory drift.
"""
import numpy as np
from scipy.optimize import least_squares


class PoseGraphOptimizer:
    def __init__(self, odometry_weight: float = 1.0, loop_weight: float = 15.0,
                 anchor_weight: float = 10.0):
        self.odometry_weight = odometry_weight
        self.loop_weight = loop_weight
        self.anchor_weight = anchor_weight

    def optimize(self, positions: np.ndarray, loop_pairs: list):
        """
        Args:
            positions: (N, 3) array of keyframe positions from raw VO/tracking.
            loop_pairs: list of (i, j) index pairs flagged as loop closures
                        (i, j index into `positions`).

        Returns:
            (N, 3) array of corrected positions.
        """
        n = len(positions)
        if n < 2:
            return positions.copy()

        if not loop_pairs:
            # Nothing to correct
            return positions.copy()

        odom_targets = np.diff(positions, axis=0)  # relative translations to preserve
        anchor = positions[0].copy()

        x0 = positions.flatten()

        def residuals(x):
            p = x.reshape(n, 3)

            # Odometry residuals: preserve original consecutive deltas.
            # Vectorized -- the previous implementation built these in a
            # Python loop, which made the numerically-differentiated LM solve
            # very slow on longer trajectories.
            odom_res = self.odometry_weight * (np.diff(p, axis=0) - odom_targets)

            # Loop-closure residuals: matched keyframes should coincide.
            if loop_pairs:
                idx_i = np.array([i for i, _ in loop_pairs])
                idx_j = np.array([j for _, j in loop_pairs])
                loop_res = self.loop_weight * (p[idx_j] - p[idx_i])
            else:
                loop_res = np.zeros((0, 3))

            # ANCHOR residual -- this was missing and it was the root cause of
            # the solver blow-ups seen earlier (proposed corrections of
            # hundreds of metres on a ~2m trajectory).
            #
            # Odometry residuals only constrain RELATIVE positions, and loop
            # residuals only constrain DIFFERENCES between positions. With
            # both, the objective is completely invariant to translating the
            # entire trajectory by any constant vector -- a 3-DOF gauge
            # freedom that leaves the problem rank-deficient. Levenberg-
            # Marquardt on a rank-deficient system can wander arbitrarily far
            # along the null space while barely changing the cost, which is
            # exactly the pathology we observed. Pinning the first keyframe to
            # its original position removes that null space and makes the
            # solve well-posed.
            anchor_res = self.anchor_weight * (p[0] - anchor)

            return np.concatenate([
                odom_res.ravel(), loop_res.ravel(), anchor_res.ravel()
            ])

        # 'trf' rather than 'lm': it is robust when the residual count differs
        # from the parameter count and is better behaved on near-degenerate
        # problems than LM.
        result = least_squares(residuals, x0, method="trf", max_nfev=2000)
        return result.x.reshape(n, 3)
