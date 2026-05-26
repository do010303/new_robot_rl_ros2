#!/usr/bin/env python3
"""
Board Transform Utility for Visual Servoing

Builds the transform pipeline from board-local coordinates to base_link:
  board-local (x, y, 0, 1) → T_vision (board→camera) → TF2 (camera→base_link)

The T_vision matrix comes from ArUco solvePnP (rotation + translation).
The TF2 transform comes from the robot URDF (camera_link → base_link).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import tf2_ros
import rclpy
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped


class BoardTransform:
    """Transforms points from board-local 2D to base_link 3D frame."""
    
    def __init__(
        self,
        tf_buffer: tf2_ros.Buffer,
        lock_transform: bool = True,
        use_ideal_rotation: bool = True,
    ):
        self.tf_buffer = tf_buffer
        self.lock_transform = lock_transform
        self.use_ideal_rotation = use_ideal_rotation
        
        # T_vision: 4×4 board→camera (from ArUco PoseStamped)
        self.T_vision = None
        # T_tf2: 4×4 camera_link→base_link (from TF2)
        self.T_tf2 = None
        # Combined: T_combined = T_tf2 @ T_vision
        self.T_combined = None
        
        self.locked = False
        self.is_ready = False
    
    def update_from_pose(self, pose_msg: PoseStamped) -> bool:
        """
        Build transform from ArUco PoseStamped position + TF2.
        
        When lock_transform=True, the first valid transform is latched and reused.
        When lock_transform=False, every valid pose updates the transform so the
        rest of the stack is ready for future continuous board tracking.

        If use_ideal_rotation=True, the measured board position is kept but the
        board axes are aligned to an ideal vertical board. If False, the full
        measured board rotation from solvePnP is used.
        """
        if self.lock_transform and self.locked:
            return True
        
        # Build T_vision (4×4 board→camera) from pose quaternion + translation
        p = pose_msg.pose.position
        q = pose_msg.pose.orientation
        
        # Use scipy for robust quaternion to matrix conversion
        q_list = [q.x, q.y, q.z, q.w]
        R = R_scipy.from_quat(q_list).as_matrix()
        
        # Build T_vision (4×4 board -> camera_optical_link)
        self.T_vision = np.eye(4)
        self.T_vision[:3, :3] = R
        self.T_vision[:3, 3] = [p.x, p.y, p.z]
        
        # Build T_tf2 (4×4 camera_optical_link→base_link) from TF2
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', pose_msg.header.frame_id,
                rclpy.time.Time(seconds=0), timeout=Duration(seconds=0.2)
            )
            
            t = tf.transform.translation
            r = tf.transform.rotation
            q2_list = [r.x, r.y, r.z, r.w]
            R2 = R_scipy.from_quat(q2_list).as_matrix()
            
            self.T_tf2 = np.eye(4)
            self.T_tf2[:3, :3] = R2
            self.T_tf2[:3, 3] = [t.x, t.y, t.z]
            
        except Exception:
            return False
        
        T_full = self.T_tf2 @ self.T_vision

        if self.use_ideal_rotation:
            board_center_base = T_full[:3, 3]
            R_ideal = np.array([
                [0, 0, -1],
                [-1, 0, 0],
                [0, 1, 0],
            ], dtype=np.float64)

            self.T_combined = np.eye(4)
            self.T_combined[:3, :3] = R_ideal
            self.T_combined[:3, 3] = board_center_base
        else:
            self.T_combined = T_full

        self.is_ready = True
        self.locked = self.lock_transform
        return True
    
    def board_to_base(self, points_board: np.ndarray) -> np.ndarray:
        """
        Transform points from board-local to base_link frame.
        
        Args:
            points_board: (N, 4) array of [x, y, 0, 1] in board-local coords
                          OR (N, 3) array of [x, y, z] (will add homogeneous coord)
        
        Returns:
            (N, 3) array of [x, y, z] in base_link frame
        """
        if self.T_combined is None:
            raise RuntimeError("Board transform not initialized — wait for ArUco detection")
        
        pts = np.atleast_2d(points_board)
        
        # Add homogeneous coordinate if needed
        if pts.shape[1] == 3:
            pts = np.hstack([pts, np.ones((pts.shape[0], 1))])
        
        # Transform: (4×4) @ (4×N).T → (4×N).T → take [:, :3]
        transformed = (self.T_combined @ pts.T).T
        return transformed[:, :3]
    
    def board_to_camera(self, points_board: np.ndarray) -> np.ndarray:
        """Transform points from board-local to camera_link frame."""
        if self.T_vision is None:
            raise RuntimeError("T_vision not initialized")
        
        pts = np.atleast_2d(points_board)
        if pts.shape[1] == 3:
            pts = np.hstack([pts, np.ones((pts.shape[0], 1))])
        
        transformed = (self.T_vision @ pts.T).T
        return transformed[:, :3]
    
    def get_board_center_base(self) -> np.ndarray:
        """Get board center position in base_link frame."""
        origin = np.array([[0, 0, 0, 1]])
        return self.board_to_base(origin)[0]
    
    def reset(self):
        """Reset transform (unlock for re-detection)."""
        self.T_vision = None
        self.T_tf2 = None
        self.T_combined = None
        self.locked = False
        self.is_ready = False
