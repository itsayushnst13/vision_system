"""
Live Loop Closure Detector
===========================
NOTE: named LiveLoopClosureDetector (in live_loop_closure.py) to avoid
colliding with the pre-existing LoopClosureDetector class that
random_forest_loop_closure.py / cnn_loop_closure.py inherit from. An
earlier version of this file was called loop_closure_detector.py and
shadowed that module on sys.path, silently breaking those two originals.

Wraps the real-labeled Random Forest classifier (trained in
evaluation/retrain_loop_closure_real_labels.py) for use inside the live
SLAM tracking loop.

Given two grayscale images, extracts the same 8 SIFT-based features used
during training and returns a loop-closure probability.
"""
import os
import numpy as np
import cv2
import joblib


class LiveLoopClosureDetector:
    def __init__(self, model_path: str, scaler_path: str, threshold: float = 0.5):
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

    def _extract_features(self, gray1: np.ndarray, gray2: np.ndarray):
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return None

        raw = self.matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        # Guard each pair: knnMatch can return fewer than 2 neighbours for a
        # query point, which would make tuple-unpacking `for m, n in raw` crash.
        matches = [p[0] for p in raw if len(p) == 2 and p[0].distance < 0.7 * p[1].distance]
        if not matches or len(matches) < 2:
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
        """
        Returns (is_loop_closure: bool, confidence: float).
        confidence is 0.0 if features could not be extracted.
        """
        feats = self._extract_features(gray1, gray2)
        if feats is None:
            return False, 0.0
        feats_scaled = self.scaler.transform(feats.reshape(1, -1))
        proba = self.clf.predict_proba(feats_scaled)[0, 1]
        return bool(proba > self.threshold), float(proba)
