#!/usr/bin/env python3
"""
Camera Viewer with RL Training Overlay

Visualizes:
- ArUco board detection with PBVS monitor
- RL target position (green circle)
- Drawing pen trajectory (purple line)
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge


class CameraViewer(Node):
    """Camera viewer with RL training overlay visualization."""
    
    def __init__(self):
        super().__init__('camera_viewer')
        
        self.bridge = CvBridge()
        
        # Camera intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # ArUco board state
        self.board_pose = None
        self.board_detected = False
        
        # RL Training state
        self.current_target = None
        self.pen_trajectory = []
        self.max_trajectory_points = 200  # Last 200 points
        
        # FPS Calculation
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0.0
        
        # Subscribers - Board detection
        self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10)
        self.create_subscription(
            PoseStamped, '/vision/board_pose', self.board_pose_callback, 10)
        self.create_subscription(
            Bool, '/vision/board_detected', self.board_detected_callback, 10)
        
        # Subscribers - RL Training
        self.create_subscription(
            Point, '/rl/current_target', self.target_callback, 10)
        self.create_subscription(
            PointStamped, '/rl/pen_position', self.pen_callback, 10)
        
        self.get_logger().info("Camera Viewer with RL Overlay started")
    
    def info_callback(self, msg):
        """Get camera intrinsics for drawing axes."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera intrinsics received")
    
    def board_pose_callback(self, msg):
        self.board_pose = msg
    
    def board_detected_callback(self, msg):
        self.board_detected = msg.data
    
    def target_callback(self, msg):
        """Update current RL target position."""
        self.current_target = msg
    
    def pen_callback(self, msg):
        """Add pen position to trajectory."""
        self.pen_trajectory.append(msg.point)
        if len(self.pen_trajectory) > self.max_trajectory_points:
            self.pen_trajectory.pop(0)
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        
        # FPS Calculation
        self.frame_count += 1
        now = time.time()
        if now - self.fps_start_time >= 1.0:
            self.fps = self.frame_count / (now - self.fps_start_time)
            self.frame_count = 0
            self.fps_start_time = now
        
        # Latency Calculation: Use wall-clock time for actual processing delay
        # This measures the time from when frame arrived to when it's displayed
        # Store arrival time on first access for this message
        frame_arrival_time = time.time()
        
        # Calculate processing latency (time spent in this callback)
        # For display, we just show how long since last frame was processed
        if not hasattr(self, '_last_frame_time'):
            self._last_frame_time = frame_arrival_time
        
        frame_delta_ms = (frame_arrival_time - self._last_frame_time) * 1000
        self._last_frame_time = frame_arrival_time
        
        # Show frame interval as "latency" (time between frames)
        latency_ms = frame_delta_ms if frame_delta_ms < 1000 else 0.0
        
        # Initialize status (default: searching)
        status_text = f"Robust: SEARCHING... FPS:{self.fps:.1f}"
        color = (0, 0, 255)  # Red
        
        # Draw Overlay (Matching original style)
        if self.board_detected and self.board_pose:
            # "Robust: LOCKED"
            status_text = f"Robust: LOCKED FPS:{self.fps:.1f}"
            color = (0, 255, 0)  # Green
            
            # Extract position (needed for display)
            pos = self.board_pose.pose.position
            
            # Draw Axes if intrinsics available
            if self.camera_matrix is not None:
                ori = self.board_pose.pose.orientation
                
                tvec = np.array([[pos.x], [pos.y], [pos.z]])
                rmat = self.quaternion_to_rotation_matrix(ori)
                rvec, _ = cv2.Rodrigues(rmat)
                
                try:
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, 
                                     rvec, tvec, 0.05)  # 5cm axes
                except Exception:
                    pass
            
            # Additional Monitor Info
            y0 = 55
            dy = 22
            cv2.putText(cv_image, "PBVS MONITOR", (10, y0), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y0 += dy
            cv2.putText(cv_image, f"pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f})", 
                       (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw RL Training Overlays (if camera intrinsics available)
        if self.camera_matrix is not None:
            # Draw target (green circle)
            if self.current_target is not None:
                self._draw_target(cv_image, self.current_target)
            
            # Draw pen trajectory (purple line)
            if len(self.pen_trajectory) > 1:
                self._draw_trajectory(cv_image, self.pen_trajectory)
        
        # Status Text (Top Left)
        cv2.putText(cv_image, f"{status_text} | Latency: {latency_ms:.1f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Camera View', cv_image)
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            rclpy.shutdown()
        elif key == ord('s'):
            filename = f"/tmp/camera_snapshot_{time.time()}.png"
            cv2.imwrite(filename, cv_image)
            self.get_logger().info(f"Saved: {filename}")
        elif key == ord('c'):  # Clear trajectory
            self.pen_trajectory.clear()
            self.get_logger().info("Trajectory cleared")
    
    def _draw_target(self, image, target):
        """Project and draw target as green circle."""
        try:
            # 3D point in base_link frame
            point_3d = np.array([[target.x], [target.y], [target.z]], dtype=np.float64)
            
            # Project to image plane (assuming camera frame aligned with base_link)
            # For proper projection, we'd need camera extrinsics (base_link -> camera)
            # Simplified: project assuming identity transform
            rvec = np.zeros(3, dtype=np.float64)
            tvec = np.zeros(3, dtype=np.float64)
            
            points_2d, _ = cv2.projectPoints(
                point_3d.T, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
            
            # Draw target
            pt = tuple(points_2d[0][0].astype(int))
            if 0 <= pt[0] < image.shape[1] and 0 <= pt[1] < image.shape[0]:
                cv2.circle(image, pt, 20, (0, 255, 0), 3)  # Green outer circle
                cv2.circle(image, pt, 5, (0, 255, 0), -1)  # Green center
                # Label
                cv2.putText(image, "TARGET", (pt[0] + 25, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            pass  # Silently ignore projection errors
    
    def _draw_trajectory(self, image, trajectory):
        """Project and draw pen trajectory as purple line."""
        try:
            # Convert trajectory to numpy array
            points_3d = np.array([[p.x, p.y, p.z] for p in trajectory], dtype=np.float64)
            
            # Project all points
            rvec = np.zeros(3, dtype=np.float64)
            tvec = np.zeros(3, dtype=np.float64)
            
            points_2d, _ = cv2.projectPoints(
                points_3d, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
            
            # Draw polyline
            pts = points_2d.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(image, [pts], False, (255, 0, 255), 3)  # Purple line
        except Exception as e:
            pass  # Silently ignore projection errors
            
def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
