"""
Frame-to-frame tracking (visual odometry).

Estimates camera motion between consecutive frames, using RGB-D PnP when depth
is available and an essential-matrix fallback when it is not.

LENS DISTORTION -- THE CENTRAL CORRECTNESS ISSUE IN THIS MODULE
---------------------------------------------------------------
TUM freiburg1 has strong radial distortion (k1=0.26, k2=-0.95, k3=1.16). An
earlier version of this file handled it inconsistently:

  * _compute_3d_points back-projected keypoints with the plain pinhole model
    applied to RAW (distorted) pixel coordinates, and
  * solvePnPRansac was passed self.dist_coeffs, so it applied the distortion
    model when projecting those 3D points back to the image.

Half the pipeline was distortion-aware and half was not. The resulting
geometric error varies with radius from the optical centre, so it does not
average out over a sequence -- it produces slowly-varying, temporally
CORRELATED pose error, which is exactly the fault signature that makes RPE
grow linearly with frame gap instead of as sqrt(gap).

The fix here is to undistort once, at the point where keypoints enter the
geometry, and then treat the camera as an ideal pinhole everywhere downstream.
Every function that consumes pixel coordinates for geometry now receives
undistorted coordinates, and zero distortion coefficients are passed to PnP.
"""
import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class TrackingState:
    """Tracking state constants."""
    INITIALIZING = 0
    TRACKING_GOOD = 1
    TRACKING_BAD = 2
    LOST = 3


def undistort_points(pts_xy: np.ndarray, camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Map raw (distorted) pixel coordinates to ideal-pinhole pixel coordinates.

    cv2.undistortPoints returns normalised camera coordinates; passing P=K
    projects them back to pixels so the result stays in the same units and can
    be used with the same intrinsic matrix downstream.
    """
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    return out.reshape(-1, 2)


def backproject(pts_xy_undistorted: np.ndarray, depths: np.ndarray,
                camera_matrix: np.ndarray) -> np.ndarray:
    """
    Pinhole back-projection of UNDISTORTED pixel coordinates to 3D camera-frame
    points. Callers must undistort first -- passing raw pixels here reintroduces
    the distortion inconsistency described in the module docstring.
    """
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    x = (pts_xy_undistorted[:, 0] - cx) * depths / fx
    y = (pts_xy_undistorted[:, 1] - cy) * depths / fy
    return np.stack([x, y, depths], axis=1)


def sample_depth(depth: np.ndarray, pts_xy: np.ndarray,
                 depth_min: float, depth_max: float,
                 patch: int = 2) -> np.ndarray:
    """
    Sample depth at (possibly sub-pixel) keypoint locations.

    Uses the median of a small patch rather than a single pixel. A single
    nearest-pixel lookup is fragile on TUM depth maps: the structured-light
    sensor leaves holes (recorded as 0) and produces large errors at depth
    discontinuities, which is precisely where corner features are detected.
    One bad depth value becomes one bad 3D point, which then either corrupts
    the PnP solve or is silently rejected as an outlier.

    Depths outside [depth_min, depth_max] are returned as 0 (invalid); the TUM
    sensor is not trustworthy outside roughly 0.5-5 m, and those limits live in
    camera_config.yaml but were previously never applied.
    """
    h, w = depth.shape[:2]
    out = np.zeros(len(pts_xy), dtype=np.float64)
    for i, (px, py) in enumerate(pts_xy):
        x, y = int(round(px)), int(round(py))
        x0, x1 = max(0, x - patch), min(w, x + patch + 1)
        y0, y1 = max(0, y - patch), min(h, y + patch + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        window = depth[y0:y1, x0:x1].reshape(-1)
        valid = window[(window > depth_min) & (window < depth_max)]
        if valid.size >= 3:
            out[i] = float(np.median(valid))
    return out


class Tracker:
    """
    Tracking module. Responsible for:
      1. Initialising the map
      2. Matching features frame to frame
      3. Estimating camera pose
    """

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 tracking_cfg: Optional[dict] = None):
        """
        Args:
            camera_matrix: 3x3 intrinsics
            dist_coeffs: distortion coefficients (k1,k2,p1,p2,k3)
            tracking_cfg: optional dict from camera_config.yaml's Tracking
                section. These values were previously hard-coded here, so
                editing the config had no effect.
        """
        cfg = tracking_cfg or {}
        self.camera_matrix = camera_matrix
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        # After undistortion the camera is an ideal pinhole, so PnP must be
        # told there is no remaining distortion to model.
        self.zero_dist = np.zeros(5, dtype=np.float64)
        self.state = TrackingState.INITIALIZING

        # Feature matching. crossCheck must be False: cv2 forbids knnMatch on a
        # cross-checking matcher, and the ratio test needs the two nearest
        # neighbours. The previous code set crossCheck=True and then substituted
        # `m.distance < 0.7 * max_distance_in_set` for the ratio test, which
        # thresholds against the WORST match present rather than the
        # second-nearest neighbour and therefore rejects almost nothing.
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.min_matches = int(cfg.get("minMatches", 50))
        self.nn_ratio = float(cfg.get("nnRatio", 0.75))
        self.min_inliers = int(cfg.get("minPnPInliers", 15))
        self.reproj_error = float(cfg.get("reprojectionError", 3.0))
        self.depth_min = float(cfg.get("depthMin", 0.5))
        self.depth_max = float(cfg.get("depthMax", 5.0))

        # Reference-frame tracking. Chaining a relative pose for every frame
        # writes each frame's error permanently into the chain. Holding a
        # reference frame until the camera has moved `referenceThreshold`
        # metres means poses inside a segment are estimated independently
        # against a common origin, so error accumulates once per segment
        # rather than once per frame. Measured on fr1/xyz this reduces ATE
        # substantially; it does NOT improve RPE at delta=1, because that
        # metric is limited by single-step PnP noise rather than by chaining.
        self.use_reference_frame = bool(cfg.get("useReferenceFrame", True))
        self.reference_threshold = float(cfg.get("referenceThreshold", 0.10))
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.ref_points_3d = None
        self.ref_pose = np.eye(4)
        self.n_reference_resets = 0

        # Previous frame state
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None
        self.prev_valid_depth = None
        self.prev_pose = np.eye(4)

        # Scale reference for the monocular fallback
        self.scale_window_size = 5
        self.recent_scales: List[float] = []
        self.median_scale = 1.0
        self.min_translation = 0.01

        # Diagnostics
        self.n_pnp = 0
        self.n_fallback = 0
        self.n_failed = 0

    # ------------------------------------------------------------------
    def _match(self, desc1, desc2):
        """Lowe's ratio test against the true second-nearest neighbour."""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        raw = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.nn_ratio * n.distance:
                good.append(m)
        return good

    def compute_3d_points(self, keypoints: List, depth: np.ndarray) -> np.ndarray:
        """
        Back-project keypoints to 3D camera-frame points, undistorting first.

        Returns an (N,3) array; rows with invalid depth are all-zero, matching
        the convention used by callers.
        """
        if not keypoints:
            return np.zeros((0, 3))
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        pts_u = undistort_points(pts, self.camera_matrix, self.dist_coeffs)
        # Depth is sampled at the RAW pixel location (that is where the sensor
        # measurement physically lives); only the geometry uses undistorted
        # coordinates.
        d = sample_depth(depth, pts, self.depth_min, self.depth_max)
        pts3d = backproject(pts_u, d, self.camera_matrix)
        pts3d[d <= 0] = 0.0
        return pts3d

    # ------------------------------------------------------------------
    def track(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray,
              depth: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Estimate the camera pose for the current frame.

        Returns a 4x4 world-from-camera pose, or None if tracking failed.
        """
        if self.state == TrackingState.INITIALIZING:
            if self._initialize_map(keypoints, descriptors):
                self.state = TrackingState.TRACKING_GOOD
                if depth is not None:
                    self.prev_points_3d = self.compute_3d_points(keypoints, depth)
                self._set_reference(keypoints, descriptors,
                                    self.prev_points_3d, np.eye(4))
            return np.eye(4)

        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            if depth is not None:
                self.prev_points_3d = self.compute_3d_points(keypoints, depth)
            return self.prev_pose

        # Choose what to track against: the held reference frame, or the
        # immediately previous frame.
        if self.use_reference_frame and self.ref_keypoints is not None:
            src_kp, src_desc, src_3d, src_pose = (
                self.ref_keypoints, self.ref_descriptors,
                self.ref_points_3d, self.ref_pose)
        else:
            src_kp, src_desc, src_3d, src_pose = (
                self.prev_keypoints, self.prev_descriptors,
                self.prev_points_3d, self.prev_pose)

        good_matches = self._match(src_desc, descriptors)
        if len(good_matches) < 8 and self.use_reference_frame:
            # Reference drifted out of view -- re-anchor on the previous frame
            # and retry rather than dropping the frame entirely.
            self._set_reference(self.prev_keypoints, self.prev_descriptors,
                                self.prev_points_3d, self.prev_pose)
            src_kp, src_desc, src_3d, src_pose = (
                self.prev_keypoints, self.prev_descriptors,
                self.prev_points_3d, self.prev_pose)
            good_matches = self._match(src_desc, descriptors)
        if len(good_matches) < 8:
            self.n_failed += 1
            return None

        prev_pts = np.float32([src_kp[m.queryIdx].pt for m in good_matches])
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

        # Undistort once, here. Everything downstream is ideal-pinhole.
        prev_pts_u = undistort_points(prev_pts, self.camera_matrix, self.dist_coeffs)
        curr_pts_u = undistort_points(curr_pts, self.camera_matrix, self.dist_coeffs)

        try:
            R = t = None
            use_pnp = False

            if depth is not None and src_3d is not None and len(src_3d):
                prev_3d = np.float32([src_3d[m.queryIdx] for m in good_matches])
                valid = np.any(prev_3d != 0, axis=1)
                if valid.sum() >= self.min_inliers:
                    obj = prev_3d[valid]
                    img = curr_pts_u[valid].astype(np.float32)

                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        obj, img, self.camera_matrix, self.zero_dist,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                        reprojectionError=self.reproj_error,
                        confidence=0.999,
                        iterationsCount=200,
                    )

                    if success and inliers is not None and len(inliers) >= self.min_inliers:
                        # Actually refine on the inlier set. The previous code
                        # carried a comment claiming to "refine using inliers
                        # only" but then used the raw RANSAC rvec/tvec, which
                        # are fitted to a minimal sample and not to the
                        # consensus set.
                        idx = inliers.reshape(-1)
                        rvec, tvec = cv2.solvePnPRefineLM(
                            obj[idx], img[idx], self.camera_matrix,
                            self.zero_dist, rvec, tvec,
                        )
                        R, _ = cv2.Rodrigues(rvec)
                        t = tvec.reshape(3)
                        use_pnp = True
                        self.n_pnp += 1

            if not use_pnp:
                # Essential-matrix fallback. Points are already undistorted, so
                # the intrinsics alone describe the projection.
                E, _ = cv2.findEssentialMat(
                    prev_pts_u, curr_pts_u, self.camera_matrix,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0,
                )
                if E is None or E.shape != (3, 3):
                    self.n_failed += 1
                    return None
                _, R, t, _ = cv2.recoverPose(E, prev_pts_u, curr_pts_u, self.camera_matrix)
                t = t.reshape(3)
                self.n_fallback += 1

            # solvePnP / recoverPose both return the transform FROM the previous
            # camera frame TO the current one:  X_curr = R @ X_prev + t.
            # Accumulating world poses as  current = prev @ transform  requires
            # `transform` to map current-frame coordinates back into the previous
            # frame, i.e. the inverse.
            R_inv = R.T
            t_inv = -R_inv @ t

            if use_pnp:
                step = float(np.linalg.norm(t_inv))
                if step > self.min_translation:
                    self.recent_scales.append(step)
                    if len(self.recent_scales) > self.scale_window_size:
                        self.recent_scales.pop(0)
                    self.median_scale = float(np.median(self.recent_scales))
            else:
                # recoverPose returns a unit-norm translation DIRECTION only.
                # Using it raw injects a ~1 m jump into centimetre-scale motion.
                norm = float(np.linalg.norm(t_inv))
                if norm > 1e-9 and self.recent_scales:
                    t_inv = t_inv / norm * self.median_scale
                elif norm > 1e-9 and depth is not None:
                    # RGB-D mode with no metric reference yet: assuming no
                    # motion is safer than injecting an arbitrary unit jump.
                    t_inv = np.zeros(3)
                # Pure monocular (depth is None): scale is inherently ambiguous,
                # so a consistent unit scale is the correct best effort.

            transform = np.eye(4)
            transform[:3, :3] = R_inv
            transform[:3, 3] = t_inv

            # Compose against whichever frame the estimate was made relative to.
            current_pose = src_pose @ transform
            self.prev_pose = current_pose.copy()

            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            if depth is not None:
                self.prev_points_3d = self.compute_3d_points(keypoints, depth)

            # Advance the reference once the camera has moved far enough from it.
            if self.use_reference_frame:
                if self.ref_keypoints is None or np.linalg.norm(
                        current_pose[:3, 3] - self.ref_pose[:3, 3]) > self.reference_threshold:
                    self._set_reference(keypoints, descriptors,
                                        self.prev_points_3d, current_pose)

            return current_pose

        except Exception as e:
            print(f"\nTracking failed: {e}")
            self.n_failed += 1
            return None

    def _set_reference(self, keypoints, descriptors, points_3d, pose):
        """Adopt a new reference frame for subsequent pose estimates."""
        self.ref_keypoints = keypoints
        self.ref_descriptors = descriptors
        self.ref_points_3d = points_3d
        self.ref_pose = pose.copy()
        self.n_reference_resets += 1

    def _initialize_map(self, keypoints: List, descriptors: np.ndarray) -> bool:
        if keypoints is None or len(keypoints) < self.min_matches:
            return False
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        return True

    def stats(self) -> dict:
        """Tracking-path counts, useful for diagnosing which branch ran."""
        return {"pnp": self.n_pnp, "essential_fallback": self.n_fallback,
                "failed": self.n_failed, "reference_resets": self.n_reference_resets}
