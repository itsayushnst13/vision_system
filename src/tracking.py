import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class TrackingState:
    """Data class to store tracking information"""
    INITIALIZING = 0
    TRACKING_GOOD = 1
    TRACKING_BAD = 2
    LOST = 3

class Tracker:
    """
    Tracking module for ORB-SLAM3. Responsible for:
    1. Initializing the map
    2. Tracking features frame to frame
    3. Estimating camera pose
    """
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        Initialize the tracker.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.state = TrackingState.INITIALIZING
        
        # Feature matching parameters
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = 50
        self.nn_ratio = 0.7  # Ratio test threshold
        
        # Previous frame information
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None
        self.prev_pose = np.eye(4)
        
        # Scale consistency parameters
        self.scale_window_size = 5
        self.recent_scales = []
        self.median_scale = 1.0
        self.scale_change_threshold = 0.3  # Maximum allowed scale change ratio
        self.min_translation = 0.01  # Minimum translation to update scale
        
    def track(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray,
              depth: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Track camera motion between current and previous frame.
        
        Args:
            frame: Current frame
            keypoints: Current frame keypoints
            descriptors: Current frame descriptors
            depth: Optional depth image (in meters)
            
        Returns:
            Optional[np.ndarray]: 4x4 camera pose matrix or None if tracking failed
        """
        if self.state == TrackingState.INITIALIZING:
            if self._initialize_map(keypoints, descriptors):
                self.state = TrackingState.TRACKING_GOOD
            return np.eye(4)
        
        if self.prev_keypoints is None:
            # First frame
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            if depth is not None:
                self.prev_points_3d = self._compute_3d_points(keypoints, depth)
            return self.prev_pose
        
        # Match features
        if self.prev_descriptors is None or descriptors is None:
            print("\nNo descriptors to match")
            return None
        matches = self.matcher.match(self.prev_descriptors, descriptors)

        if len(matches) == 0:
            print("\nNo feature matches")
            return None

        # Filter matches by distance
        max_dist = max(m.distance for m in matches)
        good_matches = [m for m in matches if m.distance < 0.7 * max_dist]
        
        if len(good_matches) < 8:
            print("\nNot enough good matches")
            return None
        
        # Get matched point pairs
        prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
        
        try:
            if depth is not None and self.prev_points_3d is not None:
                # Use 3D-2D matching with depth information
                # Get 3D points from previous frame
                prev_3d_pts = np.float32([self.prev_points_3d[m.queryIdx] for m in good_matches])
                
                # Filter out points with invalid depth
                valid_mask = np.all(prev_3d_pts != 0, axis=1)
                if np.sum(valid_mask) < 8:
                    # Not enough valid 3D points, fallback to essential matrix method
                    use_pnp = False
                else:
                    # NOTE: use SEPARATE filtered arrays here. Previously
                    # `curr_pts` itself was reassigned to curr_pts[valid_mask];
                    # if PnP was then attempted but failed, the essential-matrix
                    # fallback received a filtered curr_pts but a full-length
                    # prev_pts -> mismatched array lengths -> crash. Keep the
                    # originals intact for the fallback path.
                    prev_3d_pts_valid = prev_3d_pts[valid_mask]
                    curr_pts_valid = curr_pts[valid_mask]
                    use_pnp = True
                
                if use_pnp:
                    # Estimate pose using PnP with RANSAC outlier rejection.
                    # (Plain solvePnP has no outlier rejection: a single bad
                    # correspondence can produce a wild pose estimate that then
                    # propagates through the whole chained trajectory.)
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        prev_3d_pts_valid,
                        curr_pts_valid,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                        reprojectionError=4.0,
                        confidence=0.999,
                        iterationsCount=200
                    )

                    if success and inliers is not None and len(inliers) >= 8:
                        # Refine using inliers only
                        R, _ = cv2.Rodrigues(rvec)
                        t = tvec
                    else:
                        use_pnp = False
            else:
                use_pnp = False
            
            if not use_pnp:
                # Fallback to essential matrix method
                E, mask = cv2.findEssentialMat(
                    prev_pts,
                    curr_pts,
                    self.camera_matrix,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )
                
                if E is None:
                    return None
                    
                # Recover pose from essential matrix
                _, R, t, mask = cv2.recoverPose(E, prev_pts, curr_pts, self.camera_matrix)
            
            # solvePnP / recoverPose both return (R, t) as the transform
            # FROM the previous camera frame TO the current camera frame:
            #   X_curr = R @ X_prev + t
            # To accumulate a world-frame camera pose via
            #   current_pose = prev_pose @ transform
            # `transform` must instead map CURRENT-frame coordinates back into
            # the PREVIOUS frame (i.e. its inverse), otherwise every step
            # composes the wrong direction and pose/scale drifts
            # systematically. Invert before composing:
            R_inv = R.T
            t_inv = -R_inv @ t.reshape(3)

            if use_pnp:
                # PnP translation is metric (derived from real depth), so it
                # is a trustworthy scale reference. Track it in a rolling
                # window so we have a sane scale estimate available for the
                # essential-matrix fallback below.
                step_norm = np.linalg.norm(t_inv)
                if step_norm > self.min_translation:
                    self.recent_scales.append(step_norm)
                    if len(self.recent_scales) > self.scale_window_size:
                        self.recent_scales.pop(0)
                    self.median_scale = float(np.median(self.recent_scales))
            else:
                # Essential-matrix recoverPose() only determines translation
                # DIRECTION, not magnitude — t is unit-normalized by OpenCV.
                # Using it as-is injects an arbitrary ~1.0 (unit) jump into a
                # trajectory whose real per-frame motion is typically
                # centimeter-scale, which corrupts the entire recovered
                # trajectory scale for the rest of the sequence. Rescale the
                # unit direction by the most recent trustworthy (PnP-derived)
                # step size instead.
                current_norm = np.linalg.norm(t_inv)
                if current_norm > 1e-9 and self.recent_scales:
                    # We have a metric (PnP-derived) scale reference from
                    # earlier RGB-D frames -> rescale the unit direction to it.
                    t_inv = t_inv / current_norm * self.median_scale
                elif current_norm > 1e-9 and depth is not None:
                    # RGB-D mode but no metric scale reference yet (e.g. the
                    # very start of the sequence, all frames so far fell back):
                    # better to assume no motion than to inject an arbitrary
                    # unit-scale jump into a would-be-metric trajectory.
                    t_inv = np.zeros(3)
                # else: pure monocular mode (depth is None, no metric reference
                # possible) -- keep the unit-scale essential-matrix translation
                # as-is. Monocular SLAM is inherently scale-ambiguous, so an
                # arbitrary consistent unit scale is the correct best effort;
                # zeroing it would freeze the trajectory at the origin.

            # Create transformation matrix (previous_from_current)
            transform = np.eye(4)
            transform[:3, :3] = R_inv
            transform[:3, 3] = t_inv
            
            # Update pose
            current_pose = self.prev_pose @ transform
            self.prev_pose = current_pose.copy()
            
            # Update previous frame info
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            if depth is not None:
                self.prev_points_3d = self._compute_3d_points(keypoints, depth)
            
            return current_pose
            
        except Exception as e:
            print(f"\nTracking failed: {str(e)}")
            return None
        
    def _compute_3d_points(self, keypoints: List, depth: np.ndarray) -> np.ndarray:
        """
        Compute 3D points from keypoints and depth image.
        
        Args:
            keypoints: List of keypoints
            depth: Depth image in meters
            
        Returns:
            np.ndarray: Nx3 array of 3D points
        """
        points_3d = np.zeros((len(keypoints), 3))
        
        for i, kp in enumerate(keypoints):
            x, y = map(int, kp.pt)
            if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                z = depth[y, x]
                if z > 0:  # Valid depth
                    # Back-project to 3D using depth
                    x_3d = (x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                    y_3d = (y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
                    points_3d[i] = [x_3d, y_3d, z]
                    
        return points_3d
        
    def _initialize_map(self, keypoints: List, descriptors: np.ndarray) -> bool:
        """
        Initialize the map using the first frame.
        
        Args:
            keypoints: Keypoints from first frame
            descriptors: Descriptors from first frame
            
        Returns:
            bool: True if initialization successful
        """
        if len(keypoints) < self.min_matches:
            return False
            
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        return True 