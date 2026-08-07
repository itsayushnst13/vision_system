import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MapPoint:
    """Class representing a 3D map point"""
    position: np.ndarray  # 3D position in world coordinates
    descriptor: np.ndarray  # Feature descriptor
    observations: Dict  # Frame IDs where this point was observed
    last_seen: int  # Frame ID when this point was last observed
    created_in_frame: int  # Frame ID where this point was created

class Mapper:
    """
    Mapping module for ORB-SLAM3. Responsible for:
    1. Creating and managing map points
    2. Local mapping
    3. Local Bundle Adjustment
    """
    def __init__(self):
        """Initialize the mapper"""
        self.map_points: Dict[int, MapPoint] = {}
        self.keyframes: List = []
        self.current_frame_id = 0
        self.min_observations = 3
        
    def create_map_point(self, position: np.ndarray, descriptor: np.ndarray) -> int:
        """
        Create a new map point.
        
        Args:
            position: 3D position of the point
            descriptor: Feature descriptor
            
        Returns:
            int: ID of the created map point
        """
        point_id = len(self.map_points)
        self.map_points[point_id] = MapPoint(
            position=position,
            descriptor=descriptor,
            observations={},
            last_seen=self.current_frame_id,
            created_in_frame=self.current_frame_id
        )
        return point_id
        
    def update(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray, 
              pose: np.ndarray, is_keyframe: bool = False) -> None:
        """
        Update the map with new observations.
        
        Args:
            frame: Current image frame
            keypoints: Detected keypoints
            descriptors: Feature descriptors
            pose: Current camera pose (4x4 matrix)
            is_keyframe: Whether this frame should be a keyframe
        """
        self.current_frame_id += 1
        
        if is_keyframe:
            self._process_keyframe(frame, keypoints, descriptors, pose)
        else:
            self._update_map_points(keypoints, descriptors, pose)
            
    def _process_keyframe(self, frame: np.ndarray, keypoints: List, 
                         descriptors: np.ndarray, pose: np.ndarray) -> None:
        """
        Process a new keyframe.
        
        Args:
            frame: Keyframe image
            keypoints: Detected keypoints
            descriptors: Feature descriptors
            pose: Camera pose
        """
        # Store keyframe
        self.keyframes.append({
            'frame_id': self.current_frame_id,
            'pose': pose,
            'keypoints': keypoints,
            'descriptors': descriptors
        })
        
        # Triangulate new points
        if len(self.keyframes) > 1:
            self._triangulate_new_points(self.keyframes[-2], self.keyframes[-1])
            
        # Local Bundle Adjustment
        if len(self.keyframes) > 2:
            self._local_bundle_adjustment()
            
    def _update_map_points(self, keypoints: List, descriptors: np.ndarray, 
                          pose: np.ndarray) -> None:
        """
        Update map points with new observations.
        
        Args:
            keypoints: Detected keypoints
            descriptors: Feature descriptors
            pose: Camera pose
        """
        # Update observations for existing map points
        for point_id, map_point in self.map_points.items():
            # Project point into current frame
            point_3d = map_point.position
            point_2d = self._project_point(point_3d, pose)
            
            if point_2d is not None:
                # Find closest keypoint
                closest_kp_idx = self._find_closest_keypoint(point_2d, keypoints)
                
                if closest_kp_idx is not None:
                    # Update observation
                    map_point.observations[self.current_frame_id] = closest_kp_idx
                    map_point.last_seen = self.current_frame_id
                    
    def _triangulate_new_points(self, kf1: Dict, kf2: Dict) -> None:
        """
        Triangulate new points between two keyframes.
        
        Args:
            kf1: First keyframe
            kf2: Second keyframe
        """
        # TODO: Implement point triangulation
        pass
        
    def _local_bundle_adjustment(self) -> None:
        """
        Perform local bundle adjustment.
        """
        # TODO: Implement local bundle adjustment
        pass
        
    def _project_point(self, point_3d: np.ndarray, pose: np.ndarray) -> Optional[np.ndarray]:
        """
        Project 3D point into image plane.
        
        Args:
            point_3d: 3D point in world coordinates
            pose: Camera pose
            
        Returns:
            Optional[np.ndarray]: 2D point in image coordinates, None if behind camera
        """
        # Transform point to camera coordinates
        point_cam = pose @ np.append(point_3d, 1)
        
        if point_cam[2] <= 0:  # Point is behind camera
            return None
            
        # Project to image plane
        point_2d = point_cam[:2] / point_cam[2]
        return point_2d
        
    def _find_closest_keypoint(self, point_2d: np.ndarray, keypoints: List, 
                              threshold: float = 10.0) -> Optional[int]:
        """
        Find the closest keypoint to a projected point.
        
        Args:
            point_2d: Projected point
            keypoints: List of keypoints
            threshold: Maximum distance threshold
            
        Returns:
            Optional[int]: Index of closest keypoint, None if none found
        """
        min_dist = float('inf')
        closest_idx = None
        
        for idx, kp in enumerate(keypoints):
            dist = np.linalg.norm(point_2d - np.array(kp.pt))
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_idx = idx
                
        return closest_idx 