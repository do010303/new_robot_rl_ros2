#!/usr/bin/env python3
"""
ROS2 Humble RL Environment for 6-DOF Robot Arm
Adapted from ROS1 Noetic main_rl_environment_noetic.py

This provides:
1. State space: end-effector position + 6 joint states + target position + distances
2. Action space: 2D target position (Y, Z) on drawing surface
3. Reward calculation: distance-based with goal achievement
4. Episode management with reset and step functions
"""

import rclpy
from rclpy.node import Node
from rclpy.clock import JumpThreshold
from rclpy.duration import Duration
import numpy as np
import random
import time
from typing import Tuple, Optional

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, PoseStamped
from gazebo_msgs.msg import ModelStates

# TF2 for end-effector tracking
import tf2_ros

from rl.control_backends import (
    create_motion_backend,
    resolve_control_backend,
)

# Gym for RL spaces
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces


# ============================================================================
# WORKSPACE CONFIGURATION  
# ============================================================================

# DEFAULT workspace bounds for +Y (robot_arm2 compatibility)
# These will be OVERRIDDEN by board detection in visual_servoing
SURFACE_X_MIN = -0.06  # -6cm
SURFACE_X_MAX = 0.06   # +6cm
SURFACE_Y_MIN = 0.15   # +15cm (DEFAULT - overridden by board Y)
SURFACE_Y_MAX = 0.30   # +30cm (DEFAULT - overridden by board Y)  
SURFACE_Z_MIN = 0.16   # 16cm
SURFACE_Z_MAX = 0.28   # 28cm

# Target sphere radius for collision detection
TARGET_RADIUS = 0.01  # 1cm

# DEFAULT workspace boundaries (for robot_arm2 compatibility)
# Will be updated dynamically in visual_servoing via board detection
WORKSPACE_BOUNDS = {
    'x_min': SURFACE_X_MIN + TARGET_RADIUS,
    'x_max': SURFACE_X_MAX - TARGET_RADIUS,
    'y_min': SURFACE_Y_MIN + TARGET_RADIUS,
    'y_max': SURFACE_Y_MAX - TARGET_RADIUS,
    'z_min': SURFACE_Z_MIN + TARGET_RADIUS,
    'z_max': SURFACE_Z_MAX -TARGET_RADIUS
}

# ============================================================================
# RL ENVIRONMENT CLASS
# ============================================================================

class RLEnvironment(Node):
    """
    ROS2 RL Environment for 6-DOF Robot Arm
    
    Provides Gym-compatible interface for reinforcement learning training.
    """
    
    def __init__(
        self,
        max_episode_steps=200,
        goal_tolerance=0.01,
        control_backend: Optional[str] = None,
        mirror_safe_moves: bool = True,
    ):
        """
        Initialize RL Environment
        
        Args:
            max_episode_steps: Maximum steps per episode (default: 200)
            goal_tolerance: Distance threshold for goal achievement (default: 1cm = sphere radius)
        """
        resolved_backend = resolve_control_backend(control_backend)
        use_sim_time = resolved_backend != 'real_replay'
        super().__init__('rl_environment', parameter_overrides=[
            rclpy.parameter.Parameter(
                'use_sim_time',
                rclpy.parameter.Parameter.Type.BOOL,
                use_sim_time,
            )
        ])
        
        self.control_backend_name = resolved_backend
        self.get_logger().info(
            f"🤖 Initializing RL Environment for 6-DOF Robot "
            f"(backend={self.control_backend_name})..."
        )
        
        # Configuration
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = goal_tolerance
        self.current_step = 0
        
        # Robot state variables (6-DOF)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.joint_positions = np.zeros(6)
        self.joint_velocities = np.zeros(6)
        
        # Target sphere state (initial position at center of workspace)  
        self.target_x = 0.0
        self.target_y = (WORKSPACE_BOUNDS['y_min'] + WORKSPACE_BOUNDS['y_max']) / 2  # Center of Y workspace
        self.target_z = 0.30  # Center of Z workspace
        
        # Board-relative workspace (visual_servoing mode)
        self.use_board_tracking = False  # Set to True in visual_servoing
        self.board_detected = False
        self.board_pose: Optional[PoseStamped] = None
        self.workspace_center = np.array([0.0, 0.22, 0.22])  # Default center
        self.board_transform_util = None  # Initialized in enable_board_tracking
        self.last_board_status_log_time = 0.0
        
        # State readiness flag
        self.data_ready = False
        
        # Joint limits for Gazebo physics (from new_arm.xacro)
        self.gazebo_limits_low = np.array([
            -3.1415, -3.1415, -3.1415, -3.1415, -3.1415, -3.1415
        ])
        self.gazebo_limits_high = np.array([
            3.1415, 3.1415, 3.1415, 3.1415, 3.1415, 3.1415
        ])
        
        # RL Agent bounds strictly in [0, 180°] mapped positive space
        self.joint_offsets = np.array([1.570796, 1.570796, 1.570796, 3.141592, 1.570796, 1.570796])
        self.joint_limits_low = self.gazebo_limits_low + self.joint_offsets
        self.joint_limits_high = self.gazebo_limits_high + self.joint_offsets
        
        # IK success tracking (legacy, not used with direct joint control)
        self.last_ik_success = 1.0
        
        # Action space: 6D absolute joint angles (radians) [Positive-only]
        self.action_space = spaces.Box(
            low=self.joint_limits_low,
            high=self.joint_limits_high,
            dtype=np.float32
        )
        
        # Observation space: 16D state
        # [joints(6), robot_xyz(3), target_xyz(3), dist_xyz(3), dist_3d(1)]
        self.observation_space = spaces.Box(
            low=np.array([
                -3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159,  # joint limits min
                0.20, -0.30, 0.0,                                  # robot_xyz min (X=0.2 to 0.8)
                0.20, -0.30, 0.0,                                  # target_xyz min
                -0.60, -0.60, -0.60,                                # dist_xyz min
                0.0                                                  # dist_3d min
            ]),
            high=np.array([
                3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159,  # joint limits max
                0.80, 0.30, 0.60,                                    # robot_xyz max
                0.80, 0.30, 0.60,                                    # target_xyz max
                0.60, 0.60, 0.60,                                    # dist_xyz max
                1.0                                                  # dist_3d max
            ]),
            dtype=np.float32
        )
        
        self.get_logger().info(f"📊 Action space: 6D absolute joint angles (0° to 180° mapping)")
        self.get_logger().info(f"📊 Observation space: 16D state")
        
        # Target sphere state (static sphere in world file)
        self.target_spawned = True
        
        # Initialize ROS2 interfaces
        self._setup_tf_listener()
        self.motion_backend = create_motion_backend(
            self,
            backend_name=self.control_backend_name,
            mirror_safe_moves=mirror_safe_moves,
        )
        self.digital_twin_enabled = (self.control_backend_name == 'sim_to_real_shadow')
        self.digital_twin_mode = 'sim_to_real' if self.digital_twin_enabled else 'none'
        self._setup_publishers()
        self._setup_subscribers()
        
        self.get_logger().info("✅ RL Environment initialized!")
    
    def _setup_tf_listener(self):
        """Initialize TF2 listener for end-effector position tracking"""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_clock_jump_callback = self.get_clock().create_jump_callback(
            JumpThreshold(
                min_forward=Duration(seconds=3600.0),
                min_backward=Duration(seconds=-1.0),
                on_clock_change=True,
            ),
            post_callback=self._handle_tf_time_jump,
        )
        self.get_logger().info("✅ TF2 listener initialized")

    def _handle_tf_time_jump(self, _jump_info):
        """Clear TF history when sim time jumps backward or clock source changes."""
        try:
            self.tf_buffer.clear()
        except Exception:
            return

        self.get_logger().warn("⏱️ Sim-time jump detected — cleared RL TF buffer")

        if self.board_transform_util is not None:
            self.board_transform_util.reset()
        self.board_detected = False
        self.board_pose = None
    
    def _setup_publishers(self):
        """Initialize publishers for target visualization and overlays."""
        self.get_logger().info("⏳ Setting up publishers...")
        
        # Optional target visualization topic for external visualizers/sim helpers
        self.target_position_pub = self.create_publisher(
            Point,
            '/target_position',
            10
        )
        
        # Publishers for camera overlay (visual_servoing mode)
        self.rl_target_pub = self.create_publisher(Point, '/rl/current_target', 10)
        
        self.get_logger().info("✅ Publishers created")
    
    def _setup_subscribers(self):
        """Setup ROS2 subscribers for robot and environment state"""
        self.get_logger().info("⏳ Setting up state subscribers...")
        
        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.motion_backend.joint_state_topic,
            self._joint_state_callback,
            10
        )
        
        if self.motion_backend.uses_gazebo_model_states:
            self.model_state_sub = self.create_subscription(
                ModelStates,
                '/gazebo/model_states',
                self._model_state_callback,
                10
            )
        else:
            self.model_state_sub = None
        
        self.get_logger().info("✅ State subscribers initialized!")
    
    def enable_board_tracking(self, lock_transform: bool = True, use_ideal_rotation: bool = True):
        """Enable board-relative workspace for visual_servoing."""
        self.use_board_tracking = True
        
        # Initialize BoardTransform
        from rl.board_transform import BoardTransform
        self.board_transform_util = BoardTransform(
            self.tf_buffer,
            lock_transform=lock_transform,
            use_ideal_rotation=use_ideal_rotation,
        )
        
        # Subscribe to board pose
        self.board_sub = self.create_subscription(
            PoseStamped, '/vision/board_pose',
            self._board_callback, 10
        )
        
        tracking_mode = 'static_locked' if lock_transform else 'continuous'
        self.get_logger().info(
            f"Board tracking enabled - subscribing to /vision/board_pose "
            f"(mode={tracking_mode}, ideal_rotation={use_ideal_rotation})"
        )
    
    def _board_callback(self, msg: PoseStamped):
        """Build board-to-base_link transform from ArUco detection.
        
        Uses BoardTransform to build full 4x4 matrix for converting
        board-local coordinates to base_link frame.
        """
        if self.board_transform_util is None:
            return
        
        if self.board_transform_util.lock_transform and self.board_transform_util.locked and self.board_detected:
            return
        
        success = self.board_transform_util.update_from_pose(msg)
        
        if not success:
            self.get_logger().debug("TF2 not ready for board transform")
            return
        
        self.board_detected = True
        self.board_pose = msg
        
        # Get board center in base_link
        center = self.board_transform_util.get_board_center_base()
        self.workspace_center = center
        
        if self.board_transform_util.lock_transform:
            self.get_logger().info(
                f"Board LOCKED (board->base_link transform ready)\n"
                f"   Center at base_link: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]"
            )
        else:
            now = time.time()
            if now - self.last_board_status_log_time >= 5.0:
                self.last_board_status_log_time = now
                self.get_logger().info(
                    f"Board updated (continuous tracking)\n"
                    f"   Center at base_link: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]"
                )
    
    def wait_for_initial_detection(self, timeout=10.0):
        """Wait for initial board detection (visual_servoing mode)."""
        if not self.use_board_tracking:
            return True
        
        if self.board_detected:
            return True
        
        self.get_logger().info(f"⏳ Waiting for initial board detection...")
        start = time.time()
        
        while not self.board_detected and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.board_detected:
            self.get_logger().info(
                f"✅ Board detected at [{self.workspace_center[0]:.3f}, "
                f"{self.workspace_center[1]:.3f}, {self.workspace_center[2]:.3f}]")
            return True
        else:
            self.get_logger().warn("⚠️  No board detected! Using default workspace.")
            return False
    
    def _joint_state_callback(self, msg: JointState):
        """Update joint positions and velocities for 6-DOF robot"""
        try:
            positions, velocities, found_all = self.motion_backend.extract_joint_state(msg)
        except Exception as e:
            self.get_logger().warn(f"Error reading joint state: {e}", throttle_duration_sec=5.0)
            return
        
        self.joint_positions = positions
        self.joint_velocities = velocities
        
        if found_all:
            self.data_ready = True
        
        # Update end-effector position
        self._update_end_effector_position()
    
    def _model_state_callback(self, msg: ModelStates):
        """Update target sphere position"""
        try:
            if 'my_sphere' in msg.name:
                sphere_index = msg.name.index('my_sphere')
                sphere_pose = msg.pose[sphere_index]
                
                self.target_x = sphere_pose.position.x
                self.target_y = sphere_pose.position.y
                self.target_z = sphere_pose.position.z
                
                if len(self.joint_positions) == 6:
                    self.data_ready = True
        except Exception as e:
            self.get_logger().warn(f"Error processing model states: {e}", throttle_duration_sec=5.0)
    
    def _update_end_effector_position(self):
        """
        Update end-effector position using TF2, with FK fallback.
        
        Reads transform from base_link to bibut_1 (pen tip).
        Falls back to FK calculation if TF is unavailable (e.g. sim-time mismatch).
        """
        tf_ok = False
        try:
            # Use Time(seconds=0) to always get the LATEST transform regardless of clock
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'bibut_1',
                rclpy.time.Time(seconds=0),
                timeout=rclpy.duration.Duration(seconds=0, nanoseconds=50000000)  # 50ms
            )
            
            self.robot_x = transform.transform.translation.x
            self.robot_y = transform.transform.translation.y
            self.robot_z = transform.transform.translation.z
            tf_ok = True
            
        except Exception as e:
            # TF failed — fall back to FK calculation
            try:
                from rl.fk_ik_utils import fk
                fk_pos = fk(self.joint_positions)
                self.robot_x = fk_pos[0]
                self.robot_y = fk_pos[1]
                self.robot_z = fk_pos[2]
            except Exception as fk_err:
                self.get_logger().warn(
                    f"Both TF and FK failed: TF={e}, FK={fk_err}",
                    throttle_duration_sec=5.0
                )
    
    def get_state(self) -> Optional[np.ndarray]:
        """Get current 16D state vector for RL agent."""
        if not self.data_ready:
            return None
        
        try:
            # Calculate distances
            dist_x = self.target_x - self.robot_x
            dist_y = self.target_y - self.robot_y
            dist_z = self.target_z - self.robot_z
            dist_3d = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
            
            # Map [-pi/2, pi/2] Gazebo joints space to [0, pi] positive agent space
            rl_joints = self.joint_positions + self.joint_offsets

            state = np.array([
                # Joint positions (6)
                *rl_joints,
                # End-effector position (3)
                self.robot_x, self.robot_y, self.robot_z,
                # Target position (3)
                self.target_x, self.target_y, self.target_z,
                # Distance vector (3)
                dist_x, dist_y, dist_z,
                # Euclidean distance (1)
                dist_3d
            ], dtype=np.float32)  # Total: 16D
            
            return state
            
        except Exception as e:
            self.get_logger().error(f"Error creating state vector: {e}")
            return None
    
    def reset_environment(self) -> Optional[np.ndarray]:
        """
        Reset environment for new episode
        
        1. Move robot to home position [0,0,0,0,0,0]
        2. Randomize target sphere position
        3. Wait for robot to settle
        4. Return initial state
        
        Returns:
            Initial state observation (18D)
        """
        self.get_logger().info("🔄 Resetting environment...")
        self.current_step = 0
        
        # 1. Move robot to home position
        home_joints = np.zeros(6)
        self.get_logger().info("   Moving to home position...")
        success = self._move_to_joint_positions(home_joints, duration=2.0)
        
        if not success:
            self.get_logger().warn("⚠️ Failed to reach home position")
        
        # Wait for robot to settle
        time.sleep(0.5)
        
        # 2. Randomize target sphere position
        self._randomize_target()
        
        # 3. Wait for state to update
        time.sleep(0.2)
        
        self.get_logger().info(f"✅ Environment reset! Target: ({self.target_y:.3f}, {self.target_z:.3f})")
        
        return self.get_state()
    
    def _randomize_target(self):
        """Randomize target sphere position within 3D workspace."""
        
        if self.use_board_tracking and self.board_detected:
            # BOARD-RELATIVE: Generate random point ON the board surface
            # Using board-local 2D coords, then transform to base_link
            WORKSPACE_RADIUS = 0.06  # 6cm from board center (12x12cm board)
            
            # Random point in board-local 2D
            offset_x = random.uniform(-WORKSPACE_RADIUS, WORKSPACE_RADIUS)
            offset_y = random.uniform(-WORKSPACE_RADIUS, WORKSPACE_RADIUS)
            
            if self.board_transform_util is not None and self.board_transform_util.is_ready:
                # Transform board-local point to base_link
                board_pt = np.array([[offset_x, offset_y, 0.0, 1.0]])
                base_pt = self.board_transform_util.board_to_base(board_pt)[0]
                self.target_x = base_pt[0]
                self.target_y = base_pt[1]
                self.target_z = base_pt[2]
            else:
                # Fallback: use workspace center with offsets
                self.target_x = self.workspace_center[0] + offset_x
                self.target_y = self.workspace_center[1]
                self.target_z = self.workspace_center[2] + offset_y
            
            self.get_logger().info(
                f"   Target (on board): X={self.target_x:.3f}, "
                f"Y={self.target_y:.3f}, Z={self.target_z:.3f}")
        else:
            # STATIC WORKSPACE: Use fixed bounds (robot_arm2 compatibility)
            self.target_x = random.uniform(WORKSPACE_BOUNDS['x_min'], WORKSPACE_BOUNDS['x_max'])
            self.target_y = random.uniform(WORKSPACE_BOUNDS['y_min'], WORKSPACE_BOUNDS['y_max'])
            self.target_z = random.uniform(WORKSPACE_BOUNDS['z_min'], WORKSPACE_BOUNDS['z_max'])
            
            self.get_logger().info(f"   Target: X={self.target_x:.3f}, Y={self.target_y:.3f}, Z={self.target_z:.3f}")
        
        # Publish to camera overlay (visual_servoing mode)
        target_msg = Point(x=self.target_x, y=self.target_y, z=self.target_z)
        self.rl_target_pub.publish(target_msg)
        # Publish target position for optional external visualizers.
        try:
            target_msg = Point()
            target_msg.x = self.target_x
            target_msg.y = self.target_y
            target_msg.z = self.target_z
            self.target_position_pub.publish(target_msg)
        except Exception as e:
            self.get_logger().debug(f"   Could not publish target position: {e}")
    
    def step(self, action: np.ndarray) -> Tuple[Optional[np.ndarray], float, bool, dict]:
        """
        Execute one environment step using DIRECT JOINT CONTROL
        
        Args:
            action: 6D ABSOLUTE joint angles (radians) - target joint positions
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.current_step += 1
        
        # Get state before action
        state_before = self.get_state()
        if state_before is None:
            self.get_logger().error("State not available before action!")
            return None, -10.0, True, {'error': 'state_unavailable'}
        
        # Calculate distance before (dist_3d is at index 15 in 18D state)
        dist_before = state_before[15]
        
        # Convert agent's [0, 180°] to Gazebo's [-90°, 90°]
        target_joints = np.array(action) - self.joint_offsets
        
        # Clip to internal Gazebo joint limits
        target_joints = np.clip(target_joints, self.gazebo_limits_low, self.gazebo_limits_high)
        
        # Execute movement — duration auto-scaled to real servo speed
        success = self._move_to_joint_positions(target_joints)
        
        # Wait for movement to complete and state to update
        time.sleep(0.3)
        
        # Get state after action
        next_state = self.get_state()
        
        if next_state is None:
            self.get_logger().error("State not available after action!")
            return None, -10.0, True, {'error': 'state_unavailable'}
        
        # Calculate reward
        dist_after = next_state[15]  # dist_3d
        reward, done = self._calculate_reward(dist_after, dist_before)
        
        # Check for ground collision (Z <= -49cm) - SAFETY FEATURE
        # Base is elevated at +0.5m, so ground is at -0.5m in base_link frame
        GROUND_SAFETY_Z = -0.49  # 1cm above ground
        if self.robot_z <= GROUND_SAFETY_Z:
            reward = -50.0  # Heavy penalty for dangerous position
            done = True
            # Convert back to world coordinates for the log message (add 0.5)
            world_z = self.robot_z + 0.5
            self.get_logger().warn(f"⚠️ DANGER! Robot too low! World Z={world_z*100:.1f}cm <= 1cm (Rel Z={self.robot_z*100:.1f}cm)")
            self.get_logger().warn(f"   Heavy penalty applied (-50) - Resetting to home...")
            # AUTO-RESET: Move robot to home position to prevent damage
            home_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self._move_to_joint_positions(home_position, duration=1.0)
            time.sleep(0.5)  # Wait for recovery
        
        # NOTE: Board is collision-free (transparent) for clean RL training
        # We keep sparse rewards (0/-1) matching the old robot_arm2 project
        # The board's position is used only for target generation, not penalties
        
        # Check episode termination
        if self.current_step >= self.max_episode_steps:
            done = True
            self.get_logger().info(f"Episode ended: max steps reached ({self.max_episode_steps})")
        
        # Info dict
        info = {
            'distance': dist_after,
            'success': success,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, dist_after: float, dist_before: float) -> Tuple[float, bool]:
        """Calculate sparse reward: 0 for success, -1 for failure."""
        done = False
        
        if dist_after < self.goal_tolerance:
            reward = 0.0
            done = True
            self.get_logger().info(f"🎯 Goal reached! Distance: {dist_after*1000:.1f}mm")
        else:
            reward = -1.0
        
        return reward, done
    
    def _move_to_joint_positions(self, target_positions: np.ndarray, duration: float = None) -> bool:
        """
        Move robot to specified joint positions via the configured backend.
        
        Args:
            target_positions: Target joint angles [6] in radians
            duration: Trajectory duration in seconds (None = auto from servo speed)
        
        Returns:
            True if movement successful
        """
        if len(target_positions) != 6:
            self.get_logger().error(f"Expected 6 joint angles, got {len(target_positions)}")
            return False
        return self.motion_backend.move_to_joint_positions(target_positions, duration=duration)

    def _stream_joint_positions(self, target_positions: np.ndarray, duration: float = 0.01) -> bool:
        """Dispatch high-rate streaming commands through the configured backend."""
        return self.motion_backend.stream_joint_positions(target_positions, duration=duration)

    def _estimate_real_duration(self, target_positions: np.ndarray) -> float:
        """Estimate movement duration using the active backend's timing model."""
        return self.motion_backend.estimate_real_duration(target_positions)

    def home(self, duration: float = 2.0) -> bool:
        """Return the robot to home through the configured backend."""
        return self.motion_backend.home(duration=duration)

    def export_pi_replay_plan(self, joint_samples_rad, sample_dt: float) -> dict:
        """Downsample a joint command trace into a Pi-safe replay artifact."""
        return self.motion_backend.mapper.export_pi_replay_plan(
            joint_samples_rad=joint_samples_rad,
            sample_dt=sample_dt,
            joint_limits_low=self.gazebo_limits_low,
            joint_limits_high=self.gazebo_limits_high,
        )

    def replay_exported_plan(self, replay_plan: dict, label: str = 'replay') -> bool:
        """Replay a previously exported Pi-safe trajectory on hardware if supported."""
        return self.motion_backend.replay_exported_plan(replay_plan, label=label)


def main(args=None):
    """Test the RL environment"""
    rclpy.init(args=args)
    
    try:
        env = RLEnvironment()
        
        # Spin to process callbacks
        rclpy.spin(env)
        
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
