"""
Forward and Inverse Kinematics for 6-DOF Robot Arm
Uses actual URDF joint transforms for accurate FK
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional

# Joint transforms from URDF (New.xacro)
# Each entry is the transform from the previous link frame to the joint frame
JOINT_TRANSFORMS = [
    # Joint 1: base_link -> Link1_1 (Z-axis)
    {'xyz': np.array([-0.003394, -0.003955, 0.068502]), 'axis': np.array([0, 0, 1])},
    # Joint 2: Link1_1 -> Link2_1 (-X-axis)
    {'xyz': np.array([0.041821, -0.019984, 0.053522]), 'axis': np.array([-1, 0, 0])},
    # Joint 3: Link2_1 → Link3_1 (-X-axis aligned with J2)
    {'xyz': np.array([-0.075886, -7.0e-06, 0.116723]), 'axis': np.array([-1, 0, 0])},
    # Joint 4: Link3_1 -> Link4_1 (-Y-axis)
    {'xyz': np.array([0.032204, 0.031535, 0.062164]), 'axis': np.array([0, -1, 0])},
    # Joint 5: Link4_1 → Link5_1 (-X-axis aligned with J2)
    {'xyz': np.array([-0.032579, -0.033100, 0.077214]), 'axis': np.array([-1, 0, 0])},
    # Joint 6: Link5_1 -> Link6_1 (-Y-axis)
    {'xyz': np.array([0.031600, 0.015300, 0.063800]), 'axis': np.array([0, -1, 0])},
]

END_EFFECTOR_OFFSET = np.array([0.000079, -0.016091, 0.046444])


JOINT_LIMITS_LOW = np.array([-np.pi/2] * 6)
JOINT_LIMITS_HIGH = np.array([np.pi/2] * 6)


def fk(joint_angles):
    """Forward Kinematics using URDF transforms"""
    if len(joint_angles) != 6:
        raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")
    
    def rot_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def rot_y(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    def rot_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
    # 1. Base to Joint 1
    pos = JOINT_TRANSFORMS[0]['xyz']
    R = rot_z(joint_angles[0])
    
    # 2. Joint 1 to Joint 2
    pos = pos + R @ JOINT_TRANSFORMS[1]['xyz']
    R = R @ rot_x(-joint_angles[1]) # -X axis
    
    # 3. Joint 2 to Joint 3
    pos = pos + R @ JOINT_TRANSFORMS[2]['xyz']
    R = R @ rot_x(-joint_angles[2]) # -X axis
    
    # 4. Joint 3 to Joint 4
    pos = pos + R @ JOINT_TRANSFORMS[3]['xyz']
    R = R @ rot_y(-joint_angles[3]) # -Y axis
    
    # 5. Joint 4 to Joint 5
    pos = pos + R @ JOINT_TRANSFORMS[4]['xyz']
    R = R @ rot_x(-joint_angles[4]) # -X axis
    
    # 6. Joint 5 to Joint 6
    pos = pos + R @ JOINT_TRANSFORMS[5]['xyz']
    R = R @ rot_y(-joint_angles[5]) # -Y axis
    
    # 7. Joint 6 to End-effector
    pos = pos + R @ END_EFFECTOR_OFFSET
    
    return (pos[0], pos[1], pos[2])
