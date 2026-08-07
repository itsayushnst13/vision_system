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
    def __init__(self, odometry_weight: float = 1.0, loop_weight: float = 15.0):
        self.odometry_weight = odometry_weight
        self.loop_weight = loop_weight

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

        odom_targets = np.diff(positions, axis=0)  # relative translations to preserve

        x0 = positions.flatten()

        def residuals(x):
            p = x.reshape(n, 3)
            res = []

            # Odometry residuals: preserve original consecutive deltas
            for k in range(n - 1):
                delta = p[k + 1] - p[k]
                res.append(self.odometry_weight * (delta - odom_targets[k]))

            # Loop-closure residuals: matched keyframes should coincide
            for (i, j) in loop_pairs:
                res.append(self.loop_weight * (p[j] - p[i]))

            return np.concatenate(res) if res else np.zeros(3)

        if not loop_pairs:
            # Nothing to correct
            return positions.copy()

        result = least_squares(residuals, x0, method="lm", max_nfev=2000)
        return result.x.reshape(n, 3)
