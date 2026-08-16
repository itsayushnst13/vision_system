"""
Live loop-closure detection with geometric verification.

Named LiveLoopClosureDetector to avoid colliding with the LoopClosureDetector
base class that random_forest_loop_closure.py / cnn_loop_closure.py inherit
from. An earlier version of this file was called loop_closure_detector.py and
shadowed that module on sys.path, silently breaking those two originals.

Two stages, in order:

  1. APPEARANCE -- the real-labeled Random Forest classifier (trained in
     evaluation/retrain_loop_closure_real_labels.py) scores a pair of frames
     on 8 SIFT-based features and returns a probability.

  2. GEOMETRY -- accepted candidates are verified by solving PnP between the
     older keyframe's 3D points and the newer keyframe's 2D observations. This
     returns the relative translation between the two, which the pose graph
     needs, and rejects appearance matches that admit no consistent rigid
     motion.

Stage 2 is not optional polish. The classifier is scored on a roughly balanced
test set, but at run time nearly every queried candidate is a true negative, so
good balanced precision still yields many more false positives than true
positives in absolute terms. Geometric verification is the standard remedy: an
appearance false positive almost never survives a PnP consensus test, and a
closure with no measured relative translation cannot be fed to the pose graph
anyway.
"""
import os
import numpy as np
import cv2
import joblib

from tracking import undistort_points, backproject, sample_depth


class LiveLoopClosureDetector:
    def __init__(self, model_path: str, scaler_path: str, threshold: float = 0.5,
                 camera_matrix: np.ndarray = None, dist_coeffs: np.ndarray = None,
                 min_geometric_inliers: int = 25):
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Loop closure model/scaler not found at {model_path} / {scaler_path}. "
                "Run evaluation/retrain_loop_closure_real_labels.py first."
            )
        self.clf = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = threshold
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.camera_matrix = camera_matrix
        self.dist_coeffs = (np.zeros(5) if dist_coeffs is None
                            else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1))
        self.min_geometric_inliers = min_geometric_inliers

        # Diagnostics: how many appearance candidates survive geometry.
        self.n_appearance_accepted = 0
        self.n_geometry_accepted = 0

    # ------------------------------------------------------------------
    def _sift_matches(self, gray1, gray2):
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return None, None, None
        raw = self.matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        # Guard each pair: knnMatch can return fewer than 2 neighbours, which
        # would make tuple-unpacking crash.
        matches = [p[0] for p in raw if len(p) == 2 and p[0].distance < 0.7 * p[1].distance]
        return kp1, kp2, matches

    def _extract_features(self, gray1, gray2):
        kp1, kp2, matches = self._sift_matches(gray1, gray2)
        if matches is None or len(matches) < 2:
            return None

        match_distances = [m.distance for m in matches]
        match_ratio = len(matches) / ((len(kp1) + len(kp2)) / 2)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        spatial = np.linalg.norm(pts1 - pts2, axis=1)
        dstats = np.percentile(match_distances, [25, 50, 75])
        sstats = np.percentile(spatial, [25, 50, 75]) if len(spatial) else [float("inf")] * 3

        return np.array([
            len(matches),
            match_ratio,
            np.mean(match_distances),
            np.std(match_distances),
            np.min(spatial) if len(spatial) else float("inf"),
            dstats[1],
            sstats[1],
            match_ratio * np.exp(-np.mean(match_distances) / 100),
        ])

    def check(self, gray1: np.ndarray, gray2: np.ndarray):
        """Appearance stage only. Returns (is_candidate, confidence)."""
        feats = self._extract_features(gray1, gray2)
        if feats is None:
            return False, 0.0
        feats_scaled = self.scaler.transform(feats.reshape(1, -1))
        proba = self.clf.predict_proba(feats_scaled)[0, 1]
        return bool(proba > self.threshold), float(proba)

    # ------------------------------------------------------------------
    def estimate_relative_pose(self, gray_old, gray_new, depth_old,
                               depth_min=0.5, depth_max=5.0):
        """
        Geometrically verify a candidate and measure the motion between the two
        keyframes.

        Solves PnP from the OLD keyframe's 3D points (built from its depth map)
        to the NEW keyframe's 2D observations.

        Returns (R, t, n_inliers) with X_new = R @ X_old + t in the old
        keyframe's camera frame, or (None, None, 0) if verification fails.
        """
        if self.camera_matrix is None or depth_old is None:
            return None, None, 0

        kp1, kp2, matches = self._sift_matches(gray_old, gray_new)
        if matches is None or len(matches) < self.min_geometric_inliers:
            return None, None, 0

        pts_old = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts_new = np.float32([kp2[m.trainIdx].pt for m in matches])

        d = sample_depth(depth_old, pts_old, depth_min, depth_max)
        valid = d > 0
        if valid.sum() < self.min_geometric_inliers:
            return None, None, 0

        pts_old_u = undistort_points(pts_old[valid], self.camera_matrix, self.dist_coeffs)
        pts_new_u = undistort_points(pts_new[valid], self.camera_matrix, self.dist_coeffs)
        obj = backproject(pts_old_u, d[valid], self.camera_matrix).astype(np.float32)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, pts_new_u.astype(np.float32), self.camera_matrix, np.zeros(5),
            flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=3.0,
            confidence=0.999, iterationsCount=300)

        if not ok or inliers is None or len(inliers) < self.min_geometric_inliers:
            return None, None, 0

        idx = inliers.reshape(-1)
        rvec, tvec = cv2.solvePnPRefineLM(obj[idx], pts_new_u[idx].astype(np.float32),
                                          self.camera_matrix, np.zeros(5), rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.reshape(3), int(len(idx))

    def verify(self, gray_old, gray_new, depth_old, pose_old,
               depth_min=0.5, depth_max=5.0):
        """
        Full verification. Returns (accepted, world_frame_translation, n_inliers).

        The measured translation is expressed in the world frame so it can be
        used directly as a pose-graph edge measurement: the displacement of the
        new keyframe relative to the old one.
        """
        R, t, n = self.estimate_relative_pose(gray_old, gray_new, depth_old,
                                              depth_min, depth_max)
        if R is None:
            return False, None, 0
        # t maps old-camera coords into new-camera coords; the new camera's
        # position expressed in the old camera's frame is -R^T t.
        delta_cam = -R.T @ t
        delta_world = pose_old[:3, :3] @ delta_cam
        self.n_geometry_accepted += 1
        return True, delta_world, n
