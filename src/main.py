import cv2
import numpy as np
import os
from system import ORBSlam3
import plotly.graph_objects as go
from glob import glob

def visualize_trajectory_3d(trajectory, save_path='trajectory_3d.html'):
    """
    Create an interactive 3D visualization of the camera trajectory using Plotly
    
    Args:
        trajectory: List of 4x4 pose matrices
        save_path: Path to save the HTML visualization
    """
    # Extract camera positions
    positions = np.array([pose[:3, 3] for pose in trajectory])
    
    # Create the main trajectory trace
    trace_path = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='lines+markers',
        name='Camera Path',
        line=dict(color='blue', width=2),
        marker=dict(size=2, color=np.arange(len(positions)), colorscale='Viridis'),
    )
    
    data = [trace_path]
    
    # Add start and end points
    data.append(go.Scatter3d(
        x=[positions[0, 0]],
        y=[positions[0, 1]],
        z=[positions[0, 2]],
        mode='markers',
        name='Start',
        marker=dict(size=8, color='green'),
    ))
    
    data.append(go.Scatter3d(
        x=[positions[-1, 0]],
        y=[positions[-1, 1]],
        z=[positions[-1, 2]],
        mode='markers',
        name='End',
        marker=dict(size=8, color='red'),
    ))
    
    # Create the layout
    layout = go.Layout(
        title='3D Camera Trajectory (RGB-D)',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        showlegend=True
    )
    
    # Create and save the figure
    fig = go.Figure(data=data, layout=layout)
    fig.write_html(save_path)
    print(f"\nInteractive 3D trajectory visualization saved as '{save_path}'")

def process_rgbd_dataset(rgb_path, depth_path, config_path):
    """
    Process RGB-D dataset
    
    Args:
        rgb_path: Directory containing RGB images
        depth_path: Directory containing depth images
        config_path: Path to camera configuration file
    """
    # Get sorted lists of RGB and depth images
    rgb_files = sorted(glob(os.path.join(rgb_path, '*.png')))
    depth_files = sorted(glob(os.path.join(depth_path, '*.png')))
    
    if len(rgb_files) != len(depth_files):
        print("Error: Number of RGB and depth images don't match")
        return
    
    if not rgb_files:
        print("Error: No images found in dataset")
        return
    
    # Initialize SLAM system
    slam = ORBSlam3(config_path)
    if not slam.initialize():
        print("Failed to initialize SLAM system")
        return
    
    total_frames = len(rgb_files)
    print(f"\nProcessing {total_frames} frames...")
    
    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        # Show progress
        print(f"Processing frame {i+1}/{total_frames} ({(i+1)*100/total_frames:.1f}%)", end='\r')
        
        # Read RGB and depth images
        rgb = cv2.imread(rgb_file)
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)  # Read as-is for 16-bit depth
        
        if rgb is None or depth is None:
            print(f"\nFailed to read frame {i}")
            continue
        
        # Convert depth to meters (assuming depth is in millimeters)
        depth = depth.astype(float) / 1000.0
        
        # Process frame
        pose, keypoints = slam.process_frame(rgb, depth)
        
        # Visualize current frame
        frame_vis = rgb.copy()
        
        # Draw keypoints
        for kp in keypoints:
            pt = tuple(map(int, kp.pt))
            cv2.circle(frame_vis, pt, 3, (0, 255, 0), -1)
        
        # Draw trajectory overlay
        if len(slam.trajectory) > 1:
            # Project 3D trajectory onto image plane
            traj_points = np.array([(pose[0, 3], pose[2, 3]) for pose in slam.trajectory])
            traj_points = (traj_points * 50 + np.array([rgb.shape[1]/2, rgb.shape[0]-100])).astype(np.int32)
            cv2.polylines(frame_vis, [traj_points], False, (0, 255, 255), 2)
        
        # Add frame info
        cv2.putText(frame_vis, f"Frame: {i+1}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('ORB-SLAM3 RGB-D', frame_vis)
        
        # Show depth visualization
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imshow('Depth', depth_vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    slam.shutdown()
    
    print("\nProcessing complete!")
    
    # Create interactive 3D visualization
    print("\nGenerating 3D visualization...")
    visualize_trajectory_3d(slam.get_trajectory(), 'rgbd_trajectory_3d.html')

def main():
    """Main function to run the SLAM system"""
    # Set paths for TUM RGB-D dataset
    dataset_dir = "data/rgbd_dataset_freiburg1_xyz"
    rgb_path = os.path.join(dataset_dir, "rgb")
    depth_path = os.path.join(dataset_dir, "depth")
    
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print("Error: RGB-D dataset not found. Please download the TUM RGB-D dataset.")
        print("Dataset should be organized as:")
        print("  data/rgbd_dataset_freiburg1_xyz/")
        print("    ├── rgb/")
        print("    └── depth/")
        return
    
    # Initialize SLAM system
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), 'config', 'camera_config.yaml')
    
    # Process RGB-D dataset
    process_rgbd_dataset(rgb_path, depth_path, config_path)

if __name__ == '__main__':
    main() 