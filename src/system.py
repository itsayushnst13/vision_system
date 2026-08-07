import numpy as np
import cv2
import threading
import os
from typing import Optional, Dict, List, Tuple
from tracking import Tracker
from pose_graph_optimizer import PoseGraphOptimizer

try:
    from loop_closure_detector import LoopClosureDetector
    _HAS_LOOP_CLOSURE_MODULE = True
except ImportError:
    _HAS_LOOP_CLOSURE_MODULE = False

class ORBSlam3:
    """
    Main class for ORB-SLAM3 system that coordinates all the SLAM components.
    """
    def __init__(self, config_path: str, enable_loop_closure: bool = True):
        """
        Initialize ORB-SLAM3 system.
        
        Args:
            config_path: Path to the configuration file
            enable_loop_closure: Whether to attempt loop-closure detection and
                trajectory correction. Silently disabled (with a warning) if
                the trained classifier files aren't found, consistent with
                this codebase's graceful-degradation approach elsewhere.
        """
        self.config_path = config_path
        self.is_running = False
        
        # Load configuration
        self.load_config()
        
        # Initialize components
        self.feature_extractor = None
        self.tracker = None
        self.mapper = None
        
        # Initialize system state
        self.current_frame = None
        self.keyframes: List = []
        self.map_points: Dict = {}
        self.trajectory: List[np.ndarray] = []
        self.current_pose = np.eye(4)
        
        # Keyframe parameters
        self.min_keyframe_rotation = 0.15  # radians
        self.min_keyframe_translation = 0.1  # meters
        self.last_keyframe_pose = np.eye(4)
        self.keyframe_counter = 0

        # Loop closure state
        self.enable_loop_closure = enable_loop_closure
        self.loop_detector = None
        self.loop_pairs: List[Tuple[int, int]] = []  # (keyframe_idx_i, keyframe_idx_j)
        self.loop_check_min_keyframe_gap = 15  # only check keyframes this far apart
        self.loop_confidence_threshold = 0.9   # stricter than the classifier's own 0.5
                                                # default -- see README: the real-labeled
                                                # classifier was trained on an easily-
                                                # separable split and over-triggers on
                                                # visually similar (but not truly revisited)
                                                # frames at its default threshold.
        self.loop_max_candidates_per_check = 8   # cap comparisons for tractability
        self.pose_graph_optimizer = PoseGraphOptimizer()
        
        # Threading locks
        self.map_update_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        
    def load_config(self):
        """Load camera configuration."""
        import yaml
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        camera_config = config['Camera']
        self.camera_matrix = np.array([
            [camera_config['fx'], 0, camera_config['cx']],
            [0, camera_config['fy'], camera_config['cy']],
            [0, 0, 1]
        ])
        
        self.dist_coeffs = np.array([
            camera_config['k1'],
            camera_config['k2'],
            camera_config['p1'],
            camera_config['p2'],
            camera_config['k3']
        ])

    def initialize(self) -> bool:
        """
        Initialize all SLAM system components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize feature extraction
            self.feature_extractor = cv2.ORB_create(
                nfeatures=2000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=19,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=20
            )
            
            # Initialize tracker
            self.tracker = Tracker(self.camera_matrix, self.dist_coeffs)

            # Initialize loop closure detector (optional, graceful degradation)
            if self.enable_loop_closure and _HAS_LOOP_CLOSURE_MODULE:
                model_path = os.path.join(os.path.dirname(__file__), "..", "results",
                                           "rf_loop_closure_REAL_labels.joblib")
                scaler_path = os.path.join(os.path.dirname(__file__), "..", "results",
                                            "rf_scaler_REAL_labels.joblib")
                try:
                    self.loop_detector = LoopClosureDetector(model_path, scaler_path)
                    print("Loop closure detector loaded (real-labeled classifier).")
                except FileNotFoundError as e:
                    print(f"Loop closure disabled: {e}")
                    self.loop_detector = None
            
            self.is_running = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize ORB-SLAM3: {str(e)}")
            return False

    def process_frame(self, frame: np.ndarray, depth: np.ndarray = None) -> Tuple[Optional[np.ndarray], List]:
        """
        Process a new frame through the SLAM system.
        
        Args:
            frame: Input RGB image frame
            depth: Optional depth image (in meters)
            
        Returns:
            Tuple[Optional[np.ndarray], List]: (Estimated camera pose matrix (4x4), Keypoints) or (None, []) if tracking failed
        """
        if not self.is_running:
            return None, []
            
        with self.pose_lock:
            self.current_frame = frame.copy()
            
            # 1. Extract features
            keypoints, descriptors = self.feature_extractor.detectAndCompute(frame, None)
            
            if self.tracker is not None:
                # 2. Track camera motion
                estimated_pose = self.tracker.track(frame, keypoints, descriptors, depth)
                
                if estimated_pose is not None:
                    # Update current pose
                    self.current_pose = estimated_pose
                    self.trajectory.append(self.current_pose)
                    
                    # Check if we should create a new keyframe
                    if self._need_new_keyframe(estimated_pose):
                        self._create_keyframe(frame, keypoints, descriptors, estimated_pose, depth)
                    
                    return self.current_pose, keypoints
            
            # If tracking failed or no tracker
            if len(self.trajectory) == 0:
                # First frame, initialize pose
                self.trajectory.append(self.current_pose)
                self._create_keyframe(frame, keypoints, descriptors, self.current_pose, depth)
            
            return self.current_pose, keypoints

    def _need_new_keyframe(self, current_pose: np.ndarray) -> bool:
        """
        Check if we need to create a new keyframe.
        
        Args:
            current_pose: Current camera pose
            
        Returns:
            bool: True if new keyframe needed
        """
        # Extract rotation and translation from relative transformation
        relative_transform = np.linalg.inv(self.last_keyframe_pose) @ current_pose
        R = relative_transform[:3, :3]
        t = relative_transform[:3, 3]
        
        # Compute rotation angle
        angle = np.arccos((np.trace(R) - 1) / 2)
        
        # Compute translation distance
        distance = np.linalg.norm(t)
        
        return angle > self.min_keyframe_rotation or distance > self.min_keyframe_translation
        
    def _create_keyframe(self, frame: np.ndarray, keypoints: List, 
                        descriptors: np.ndarray, pose: np.ndarray,
                        depth: np.ndarray = None):
        """
        Create a new keyframe and add it to the system.
        
        Args:
            frame: Image frame
            keypoints: Detected keypoints
            descriptors: Feature descriptors
            pose: Camera pose
            depth: Optional depth image (in meters)
        """
        # Triangulate 3D points
        points_3d = np.zeros((len(keypoints), 3))
        
        if depth is not None:
            # Use depth information when available
            for i, kp in enumerate(keypoints):
                x, y = map(int, kp.pt)
                if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                    z = depth[y, x]
                    if z > 0:  # Valid depth
                        # Back-project to 3D using depth
                        x_3d = (x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                        y_3d = (y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
                        points_3d[i] = [x_3d, y_3d, z]
        else:
            # Fallback to simple projection when no depth available
            for i, kp in enumerate(keypoints):
                pt = np.array([kp.pt[0], kp.pt[1], 1.0])
                points_3d[i] = (np.linalg.inv(pose)[:3, :3] @ pt) + pose[:3, 3]

        # Store the keyframe -- previously this method computed points_3d and
        # discarded it without ever appending to self.keyframes, so
        # get_map_points()/loop closure had nothing to work with. Fixed here.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self.keyframes.append({
            "image_gray": gray,
            "pose": pose.copy(),
            "trajectory_index": len(self.trajectory) - 1,  # index into self.trajectory
            "points_3d": points_3d,
        })

        # Update state
        self.last_keyframe_pose = pose.copy()
        self.keyframe_counter += 1

        # Loop closure check: compare this new keyframe against sufficiently
        # older keyframes only (mirrors the temporal-separation criterion
        # used when the classifier was trained on real labels).
        if self.loop_detector is not None and len(self.keyframes) > self.loop_check_min_keyframe_gap:
            new_idx = len(self.keyframes) - 1
            all_candidates = list(range(0, new_idx - self.loop_check_min_keyframe_gap + 1))
            # Cap the number of comparisons for tractability: evenly
            # subsample candidates rather than checking every older keyframe.
            if len(all_candidates) > self.loop_max_candidates_per_check:
                step = len(all_candidates) / self.loop_max_candidates_per_check
                candidates = [all_candidates[int(i * step)] for i in range(self.loop_max_candidates_per_check)]
            else:
                candidates = all_candidates
            best_j, best_conf = None, 0.0
            for j in candidates:
                is_loop, conf = self.loop_detector.check(
                    self.keyframes[j]["image_gray"], gray
                )
                if conf > self.loop_confidence_threshold and conf > best_conf:
                    best_j, best_conf = j, conf
            if best_j is not None:
                print(f"Loop closure detected: keyframe {best_j} <-> {new_idx} "
                      f"(confidence={best_conf:.3f})")
                self.loop_pairs.append((best_j, new_idx))
                # Re-optimizing on every single detection is expensive
                # (O(n) least-squares solve each time); batch corrections
                # instead of triggering one per detection.
                if len(self.loop_pairs) % 5 == 0:
                    self._correct_trajectory_with_loop_closures()

    def _correct_trajectory_with_loop_closures(self):
        """
        Run pose-graph correction using all detected loop closures so far.

        Simplification: this optimizes and corrects the TRANSLATION
        component of keyframe poses only (rotation is left as estimated by
        the tracker). The resulting per-keyframe correction offset is then
        linearly interpolated onto the full (non-keyframe) trajectory
        between keyframes. This is a deliberately lightweight approach in
        the absence of GTSAM -- see pose_graph_optimizer.py.
        """
        if not self.loop_pairs or len(self.keyframes) < 2:
            return

        kf_positions = np.array([kf["pose"][:3, 3] for kf in self.keyframes])
        corrected = self.pose_graph_optimizer.optimize(kf_positions, self.loop_pairs)

        # Safety guard: a genuinely mistaken loop-closure constraint (false
        # positive) can make the least-squares solve diverge wildly. Reject
        # the correction outright if it would move any keyframe further
        # than a few times the trajectory's own spatial extent -- that is
        # never a legitimate correction, only a solver blow-up.
        traj_extent = np.linalg.norm(kf_positions.max(axis=0) - kf_positions.min(axis=0))
        max_shift = np.max(np.linalg.norm(corrected - kf_positions, axis=1))
        sanity_limit = max(10.0 * traj_extent, 1.0)  # generous but bounded
        if max_shift > sanity_limit:
            print(f"Loop closure correction REJECTED: proposed shift "
                  f"{max_shift:.2f}m exceeds sanity limit {sanity_limit:.2f}m "
                  f"(likely a false-positive loop closure destabilizing the "
                  f"pose graph). Trajectory left uncorrected.")
            return

        with self.map_update_lock:
            for k, kf in enumerate(self.keyframes):
                offset = corrected[k] - kf_positions[k]
                kf["pose"][:3, 3] = corrected[k]

                # Apply this keyframe's correction as a piecewise-constant
                # offset to all trajectory entries up to the next keyframe.
                # (A true SE(3) pose-graph would interpolate rotations too;
                # here we only correct translation, documented above.)
                start = kf["trajectory_index"]
                end = (self.keyframes[k + 1]["trajectory_index"]
                       if k + 1 < len(self.keyframes) else len(self.trajectory))
                for t in range(start, min(end, len(self.trajectory))):
                    self.trajectory[t][:3, 3] += offset



    def flush_loop_closures(self):
        """Force a pose-graph correction pass using all loop closures found
        so far (in case the last batch didn't reach the auto-trigger count)."""
        self._correct_trajectory_with_loop_closures()

    def shutdown(self):
        """
        Properly shutdown the SLAM system.
        """
        self.is_running = False
        
    def get_trajectory(self) -> List[np.ndarray]:
        """
        Get the camera trajectory.
        
        Returns:
            List[np.ndarray]: List of 4x4 pose matrices
        """
        return self.trajectory
        
    def get_map_points(self) -> Dict:
        """
        Get the current map points.
        
        Returns:
            Dict: Dictionary containing map points and their descriptors
        """
        with self.map_update_lock:
            return self.map_points.copy() 