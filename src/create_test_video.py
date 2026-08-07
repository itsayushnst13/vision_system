import cv2
import numpy as np
import os

def create_test_video():
    """Create a test video with moving patterns for SLAM testing"""
    # Video parameters
    width = 640
    height = 480
    fps = 30
    duration = 10  # seconds
    n_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))
    
    # Create pattern of circles that move in 3D
    n_points = 50
    points_3d = np.random.rand(n_points, 3)
    points_3d = points_3d * np.array([10, 10, 5])  # Scale points
    
    print("Creating test video...")
    for i in range(n_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Camera motion (circular path)
        t = i / n_frames * 2 * np.pi
        camera_x = 5 * np.cos(t)
        camera_z = 5 * np.sin(t)
        
        # Create camera pose matrix
        camera_pos = np.array([camera_x, 0, camera_z])
        look_at = np.array([0, 0, 0])
        up = np.array([0, 1, 0])
        
        # Compute camera rotation matrix
        z_axis = look_at - camera_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        t = -R @ camera_pos
        
        # Project 3D points to 2D
        points_2d = []
        for point in points_3d:
            # Transform point to camera frame
            p_cam = R @ point + t
            
            # Perspective projection (assuming focal length = 500)
            if p_cam[2] > 0:  # Point is in front of camera
                x = int(500 * p_cam[0] / p_cam[2] + width/2)
                y = int(500 * p_cam[1] / p_cam[2] + height/2)
                if 0 <= x < width and 0 <= y < height:
                    points_2d.append((x, y))
        
        # Draw points
        for x, y in points_2d:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
        # Add some static features
        for j in range(10):
            x = int(width * (j + 1) / 11)
            cv2.line(frame, (x, 0), (x, height), (50, 50, 50), 1)
            cv2.line(frame, (0, x * height//width), (width, x * height//width), (50, 50, 50), 1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}/{n_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Show progress
        if i % fps == 0:
            print(f"Progress: {i/n_frames*100:.1f}%")
    
    # Cleanup
    out.release()
    print("Test video created successfully!")

if __name__ == '__main__':
    create_test_video() 