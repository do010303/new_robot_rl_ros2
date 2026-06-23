#!/usr/bin/env python3
"""
PID Tuning RL Environment for 6-DOF Robot Arm

This environment wraps the existing RLEnvironment via COMPOSITION (not inheritance)
to provide a PID gain tuning interface. The RL agent learns optimal Kp, Ki, Kd
gains for each joint.

Target generation:
    - Generates targets in JOINT SPACE (random valid joint configurations)
    - Uses FK to compute XYZ for visualization (sphere teleport + camera overlay)
    - No Neural IK dependency — FK is exact math from URDF

Architecture (single-step MDP per episode):
    1. Reset robot to home
    2. Generate random joint target, FK → XYZ for visualization
    3. RL agent observes state (24D) and outputs PID gains (18D)
    4. PID controller tracks trajectory from current → target
    5. Reward = -tracking_error (IAE) - effort penalty
    6. Episode ends after one complete movement

References:
    - Autotuning PID using Actor-Critic Deep RL (2022), arXiv:2212.00013
    - Actor-critic learning based PID control for robotic manipulators (2024)
"""

import os
import sys
import numpy as np
import time
from typing import Tuple, Optional, Dict, List

import rclpy

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from controllers.pid_joint_controller import PIDJointController
from controllers.trajectory_generator import TrajectoryGenerator

# Gym spaces
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces


# =============================================================================
# CONFIGURATION
# =============================================================================

# PID trajectory parameters
TRAJECTORY_STEPS = 50        # Updated to 50 to match Raspberry Pi 50Hz PWM Hardware loop
TRAJECTORY_DT = 0.02         # 50Hz Control Timestep
TRAJECTORY_DURATION = 1.0    # Keep physical duration at 1.0 second
SETTLE_TIME = 0.3            # Time to wait after trajectory completion

# Drawing-mode evaluation smoothing:
# Waypoint error is measured at segment boundaries. If we measure exactly at the boundary
# step, we can catch transient overshoot (especially at sharp corners). Holding the
# waypoint for a few ticks before measuring makes MaxWP reflect steady tracking quality.
DRAWING_WAYPOINT_HOLD_STEPS = 5   # 5 ticks @50Hz = 100ms hold before boundary measurement
DRAWING_BOUNDARY_INTEGRAL_DECAY = 0.15

# IK weights: position-first approach
# Position accuracy is paramount; orientation and posture are secondary refinements.
IK_ORIENT_WEIGHT = 0.01     # Reaching/default orientation penalty
IK_DRAWING_ORIENT_WEIGHT = 0.0005  # Drawing should prioritize Cartesian path geometry
IK_J4_REG_WEIGHT = 0.001    # Keep wrist roll near zero to prevent swinging
IK_CONTINUITY_WEIGHT = 1e-4  # Tiny penalty to keep joints close to seed (prevents branch jumps)
IK_ORIENTATION_TARGET = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

# Reward weights
REWARD_ALPHA = 1.0           # Weight for IAE (tracking error)
REWARD_BETA = 0.01           # Weight for control effort
REWARD_GAMMA = 10.0          # Weight for final position error

# Reward shaping for drawing quality
# Avg waypoint error captures overall quality; max waypoint error captures spikes/outliers.
# Keep reaching-mode behavior unchanged by passing max_wp_mm=None from the caller.
REWARD_WP_AVG_W = 1.0
REWARD_WP_MAX_W = 0.5
REWARD_NORM_IAE_W = 1.0
REWARD_EFFORT_W = 0.005
REWARD_SMOOTH_DELTA_W = 0.20
REWARD_SMOOTH_JERK_W = 0.10

# Default joint limits fallback (raw Gazebo angles). The active PID tuning
# session should inherit the live limits from base_env so it stays aligned with
# the currently loaded robot description.
DEFAULT_JOINT_LIMITS_LOW = np.array([-1.5708, -1.0472, -1.5708, -1.5708, -1.5708, -1.5708])
DEFAULT_JOINT_LIMITS_HIGH = np.array([1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708])

# Joint 4 convention:
# - software-facing mapped space: 90deg lock
# - Gazebo raw space: 0 rad lock
# - physical servo space: 90deg lock
J4_LOCK_RAW_RAD = 0.0
J4_LOCK_MAPPED_DEG = 90.0

# Target sampling: how much of the joint range to sample from
TARGET_RANGE_FRACTION = 0.7


# =============================================================================
# PID TUNING ENVIRONMENT
# =============================================================================

class PIDTuningEnv:
    """
    RL Environment for PID Gain Tuning (wraps RLEnvironment via composition).
    
    State Space (24D):
        - Joint positions q_actual (6)
        - Joint velocities q̇_actual (6) 
        - Target joint positions q_goal (6)
        - Tracking errors e = q_goal - q_actual (6)
    
    Action Space (18D):
        - Kp gains for 6 joints (6)
        - Ki gains for 6 joints (6)
        - Kd gains for 6 joints (6)
    
    Each episode is a single-step MDP:
        observe state → output PID gains → execute trajectory → receive reward
    """
    
    def __init__(self, base_env, n_joints: int = 6, mode: str = 'reaching',
                 ik_policy_mode: Optional[str] = None):
        """
        Initialize PID Tuning Environment.
        
        Args:
            base_env: The existing RLEnvironment instance (provides ROS2 interface)
            n_joints: Number of joints (default: 6)
            mode: 'reaching' or 'drawing'

        """
        self.base_env = base_env
        self.n_joints = n_joints
        self.control_backend_name = getattr(base_env, 'control_backend_name', 'sim')
        
        # PID controller and trajectory generator
        self.pid = PIDJointController(n_joints=n_joints)
        self.traj_gen = TrajectoryGenerator(
            n_joints=n_joints, 
            dt=TRAJECTORY_DT,
            default_duration=TRAJECTORY_DURATION
        )
        
        # Mode and drawing state
        self.mode = mode
        self.shape_joint_waypoints = []
        self.shape_xyz_waypoints = []

        # Inherit the live raw Gazebo joint limits and mapped 0-180 offsets from
        # the base environment so PID tuning matches option 1's convention.
        self.joint_limits_low = np.asarray(
            getattr(base_env, 'gazebo_limits_low', DEFAULT_JOINT_LIMITS_LOW),
            dtype=np.float64,
        ).copy()
        self.joint_limits_high = np.asarray(
            getattr(base_env, 'gazebo_limits_high', DEFAULT_JOINT_LIMITS_HIGH),
            dtype=np.float64,
        ).copy()
        self.joint_offsets = np.asarray(
            getattr(base_env, 'joint_offsets', np.full(n_joints, np.pi / 2.0)),
            dtype=np.float64,
        ).copy()
        self.j4_lock_raw = J4_LOCK_RAW_RAD
        self.j4_lock_mapped_deg = J4_LOCK_MAPPED_DEG
        
        # RL spaces
        self.state_dim = 24  # 6 pos + 6 vel + 6 target + 6 error
        self.action_dim = 18  # 6 Kp + 6 Ki + 6 Kd
        
        # Observation space: 24D
        obs_low = np.concatenate([
            np.full(n_joints, -np.pi),      # joint positions min
            np.full(n_joints, -10.0),        # joint velocities min
            self.joint_limits_low,           # target joints min
            np.full(n_joints, -2 * np.pi),   # tracking error min
        ])
        obs_high = np.concatenate([
            np.full(n_joints, np.pi),        # joint positions max
            np.full(n_joints, 10.0),         # joint velocities max
            self.joint_limits_high,          # target joints max
            np.full(n_joints, 2 * np.pi),    # tracking error max
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Action space: 18D in [-1, 1] (sigmoid-scaled to gain ranges)
        self.action_space = spaces.Box(
            low=np.full(self.action_dim, -1.0),
            high=np.full(self.action_dim, 1.0),
            dtype=np.float32
        )
        
        # Current episode state
        self.current_q_goal = np.zeros(n_joints)
        self.current_target_xyz = np.zeros(3)
        self.home_position = np.zeros(n_joints)
        
        # Episode counter
        self.episode_count = 0
        
        # Logging
        self.gain_history = []  # Track gain evolution
        self.last_episode_artifact = None
        
        self._log("PID Tuning Environment initialized")
        self._log(f"  State dim: {self.state_dim}, Action dim: {self.action_dim}")
        self._log(f"  Target gen: joint-space random → FK for visualization")
        self._log(
            f"  IK: position-first "
            f"(reach_orient_w={IK_ORIENT_WEIGHT}, draw_orient_w={IK_DRAWING_ORIENT_WEIGHT}, "
            f"j4_reg={IK_J4_REG_WEIGHT})"
        )
        self._log(
            f"  Joint 4 lock: mapped={self.j4_lock_mapped_deg:.0f}deg "
            f"/ gazebo_raw={self.j4_lock_raw:.2f}rad / physical≈90deg"
        )
        self._log(f"  Trajectory: {TRAJECTORY_STEPS} steps, {TRAJECTORY_DURATION}s")
        self._log(f"  PID gain ranges: Kp=[0, {self.pid.GAIN_RANGES['Kp'][1]}], "
                  f"Ki=[0, {self.pid.GAIN_RANGES['Ki'][1]}], "
                  f"Kd=[0, {self.pid.GAIN_RANGES['Kd'][1]}]")
    
    def _log(self, msg: str):
        """Log via the base environment's ROS logger."""
        self.base_env.get_logger().info(f"[PID-Tune] {msg}")
    
    def _spin(self, n: int = 5, timeout: float = 0.1):
        """Spin ROS to process callbacks."""
        for _ in range(n):
            rclpy.spin_once(self.base_env, timeout_sec=timeout)
    
    def _get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current joint positions and velocities from the base environment."""
        self._spin(3)
        q = np.array(self.base_env.joint_positions, dtype=np.float64)
        qd = np.array(self.base_env.joint_velocities, dtype=np.float64)
        return q, qd

    def _solve_ik_waypoint(self, target_xyz: np.ndarray, q_seed: np.ndarray,
                           log_context: str = '',
                           orient_weight: float = IK_ORIENT_WEIGHT) -> Tuple[np.ndarray, float]:
        """
        Position-first IK solver.

        Loss = pos_loss + 0.01 * orient_loss + 0.001 * j4_reg

        Position accuracy dominates; orientation and Joint 4 regularization
        are secondary refinements that prevent pen-direction flip and wrist
        roll drift without sacrificing positional accuracy.
        """
        from scipy.optimize import minimize
        from rl.fk_ik_utils import fk_with_orientation

        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        q_seed = np.asarray(q_seed, dtype=np.float64)

        def ik_loss(q):
            pos, v_pen = fk_with_orientation(list(q), raw=True)
            pos_loss = np.sum((np.asarray(pos, dtype=np.float64) - target_xyz) ** 2)
            orient_loss = np.sum((np.asarray(v_pen, dtype=np.float64) - IK_ORIENTATION_TARGET) ** 2)
            j4_reg = (q[3] - self.j4_lock_raw) ** 2
            continuity = np.sum((q - q_seed) ** 2)  # Prefer solutions near seed
            return (pos_loss
                    + orient_weight * orient_loss
                    + IK_J4_REG_WEIGHT * j4_reg
                    + IK_CONTINUITY_WEIGHT * continuity)

        bounds = list(zip(self.joint_limits_low, self.joint_limits_high))
        bounds[3] = (self.j4_lock_raw, self.j4_lock_raw)
        res = minimize(ik_loss, q_seed, bounds=bounds, method='L-BFGS-B')

        q_solution = np.asarray(res.x, dtype=np.float64)

        # Verify actual position error
        fk_pos, _ = fk_with_orientation(list(q_solution), raw=True)
        pos_err_mm = np.linalg.norm(np.asarray(fk_pos) - target_xyz) * 1000.0

        if pos_err_mm > 5.0:
            self._log(
                f"⚠️ IK position miss {pos_err_mm:.1f}mm for {log_context}"
            )

        return q_solution, pos_err_mm

    def _smooth_replay_trace(self, joint_samples_rad: List[np.ndarray], passes: int = 2) -> List[np.ndarray]:
        """Lightly smooth the nominal replay trajectory before sending it to hardware."""
        if not joint_samples_rad:
            return []

        smoothed = np.asarray(joint_samples_rad, dtype=np.float64).copy()
        if smoothed.ndim != 2 or len(smoothed) < 3:
            return [sample.copy() for sample in smoothed]

        for _ in range(max(1, passes)):
            prev = smoothed.copy()
            smoothed[1:-1] = (
                (0.2 * prev[:-2]) +
                (0.6 * prev[1:-1]) +
                (0.2 * prev[2:])
            )

        smoothed = np.clip(smoothed, self.joint_limits_low, self.joint_limits_high)
        return [sample.copy() for sample in smoothed]

    # =========================================================================
    # TARGET GENERATION (Joint-space + FK visualization)
    # =========================================================================
    
    def _generate_random_target(self) -> np.ndarray:
        """
        Generate target ON THE BOARD and use Numerical IK to find joint angles.
        
        This perfectly matches the old visual servoing logic:
        1. Call base_env._randomize_target() (handles board constraints & visualization)
        2. Read the generated XYZ position from base_env
        3. Use scipy.optimize (Numerical IK) to find exact joint angles
        
        Returns:
            q_goal: Target joint configuration [n_joints]
        """
        # 1. Use the EXACT same target generation as old RL (board constrained)
        self.base_env._randomize_target()
        
        # 2. Get the XYZ target on the board
        target_xyz = np.array([
            self.base_env.target_x,
            self.base_env.target_y,
            self.base_env.target_z
        ])
        self.current_target_xyz = target_xyz
        
        q_start, _ = self._get_joint_state()
        q_goal, pos_err_mm = self._solve_ik_waypoint(
            target_xyz, q_start, log_context='reaching target',
            orient_weight=IK_ORIENT_WEIGHT,
        )
        
        self._log(f"Target Board XYZ=[{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}] "
                  f"→ Joints={np.degrees(q_goal).astype(int)}° (err={pos_err_mm:.1f}mm)")
        
        return q_goal
    
    # =========================================================================
    # ENVIRONMENT INTERFACE
    # =========================================================================
    
    def get_state(self) -> np.ndarray:
        """
        Build 24D state vector.
        
        Returns:
            state: [q_actual(6), q_vel(6), q_goal(6), error(6)]
        """
        q_actual, q_vel = self._get_joint_state()
        error = self.current_q_goal - q_actual
        
        state = np.concatenate([
            q_actual,               # Joint positions (6)
            q_vel,                  # Joint velocities (6)
            self.current_q_goal,    # Target joints (6)
            error,                  # Tracking error (6)
        ]).astype(np.float32)
        
        return state
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        
        1. Move robot to home position
        2. Generate random joint target
        3. FK → XYZ → teleport visual sphere
        4. Return initial 24D state
        
        Returns:
            Initial state observation (24D)
        """
        self.episode_count += 1
        self._log(f"=== Episode {self.episode_count} Reset ===")
        
        if self.mode == 'drawing':
            # Drawing mode: reset DrawingEnvironment (generates shape + moves home)
            self._log("Resetting DrawingEnvironment (shape generation + home)...")
            _ = self.base_env.reset_environment()
            self._spin(50) # 0.5 seconds of active spinning
            
            # Extract Cartesian shape waypoints from DrawingEnvironment
            if hasattr(self.base_env, 'waypoints') and len(self.base_env.waypoints) > 0:
                self.shape_xyz_waypoints = self.base_env.waypoints.copy()
            else:
                self._log("⚠️ No waypoints from DrawingEnvironment, falling back")
                self.shape_xyz_waypoints = np.array([[0.4, 0.0, 0.5]])
                
            self._log(f"📐 Solving IK for {len(self.shape_xyz_waypoints)} shape waypoints...")
            
            self.shape_joint_waypoints = []
            q_seed = self.home_position.copy()
            max_err = 0.0

            for wp_idx, wp_xyz in enumerate(self.shape_xyz_waypoints):
                target = np.array(wp_xyz)
                q_solution, pos_err_mm = self._solve_ik_waypoint(
                    target, q_seed,
                    log_context=(
                        f"WP {wp_idx + 1}/{len(self.shape_xyz_waypoints)} "
                        f"[{target[0]:.3f},{target[1]:.3f},{target[2]:.3f}]"
                    ),
                    orient_weight=IK_DRAWING_ORIENT_WEIGHT,
                )
                self.shape_joint_waypoints.append(q_solution.copy())
                q_seed = q_solution.copy()
                max_err = max(max_err, pos_err_mm)
                
            self.current_q_goal = self.shape_joint_waypoints[-1]
            self.current_target_xyz = self.shape_xyz_waypoints[-1].copy()
            self._log(f"✅ IK solved for {len(self.shape_joint_waypoints)} waypoints (max err={max_err:.1f}mm)")
            
            # Pre-move the robot to the START of the shape before RL episode begins
            # This prevents penalizing the PID for the "reach" phase
            self._log("Moving arm to shape start position...")
            success = self.base_env._move_to_joint_positions(self.shape_joint_waypoints[0], duration=2.0)
            if not success:
                self._log("⚠️ Move to shape start position reported failure; continuing with PID episode")
            self._spin(100) # 1.0 seconds of active spinning

        else:
            # Reaching mode: move home + random target
            self._log("Moving to home position...")
            success = self.base_env._move_to_joint_positions(self.home_position, duration=2.0)
            if not success:
                self._log("⚠️ Home move reported failure; continuing with PID episode")
            self._spin(50) # 0.5s settling
            self.current_q_goal = self._generate_random_target()
        
        # Reset PID controller state
        self.pid.reset()
        
        # Get initial state
        state = self.get_state()
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one episode: set PID gains → track trajectory → return reward.
        
        This is a SINGLE-STEP MDP: one action per episode, one reward.
        The "step" encompasses an entire trajectory execution.
        
        Args:
            action: 18D PID gains in [-1, 1] (will be sigmoid-scaled)
        
        Returns:
            next_state: Final 24D state after movement
            reward: Negative tracking error (higher = better tracking)
            done: Always True (single-step episode)
            info: Dict with tracking metrics and gain values
        """
        action = np.array(action, dtype=np.float64)
        if not self.base_env.motion_backend.supports_high_rate_streaming:
            raise RuntimeError(
                f"Backend '{self.control_backend_name}' does not support PID high-rate streaming. "
                "Use sim/sim_to_real_shadow for training or real_replay for artifact playback."
            )
        
        # 1. Set PID gains from RL agent output
        self.pid.set_gains_from_normalized(action)
        gains = self.pid.get_gains_dict()
        
        self._log(f"PID Gains: Kp={np.round(gains['Kp'], 2)}, "
                  f"Ki={np.round(gains['Ki'], 3)}, Kd={np.round(gains['Kd'], 3)}")
        
        # 2. Generate trajectory
        q_start, _ = self._get_joint_state()
        
        if self.mode == 'reaching':
            # Single sweeping trajectory
            trajectory = self.traj_gen.linear(
                q_start, self.current_q_goal, 
                n_steps=TRAJECTORY_STEPS
            )
            # Append settling steps
            settling_steps = int(SETTLE_TIME / TRAJECTORY_DT)
            if settling_steps > 0:
                trajectory = np.vstack([
                    trajectory, 
                    np.tile(self.current_q_goal, (settling_steps, 1))
                ])
        else:
            # Drawing mode: multiple interconnected segments
            trajectory_list = []
            current_pos = q_start
            
            # Reduced steps per short drawing segment (usually waypoints are close)
            SEGMENT_STEPS = 20
            
            # Since reset() already moved us to shape_joint_waypoints[0], 
            # we draw the remaining points
            for wp in self.shape_joint_waypoints[1:]:
                seg = self.traj_gen.linear(current_pos, wp, n_steps=SEGMENT_STEPS)
                trajectory_list.append(seg)
                current_pos = wp
                
            trajectory = np.vstack(trajectory_list)
            
            # Append settling steps at the very end
            settling_steps = int(SETTLE_TIME / TRAJECTORY_DT)
            if settling_steps > 0:
                trajectory = np.vstack([
                    trajectory, 
                    np.tile(current_pos, (settling_steps, 1))
                ])
                
        self._log(f"Tracking: {len(trajectory)} steps ("
                  f"{len(self.shape_joint_waypoints) if self.mode=='drawing' else 1} segments), "
                  f"{np.degrees(np.linalg.norm(self.current_q_goal - q_start)):.1f}° net movement")
        
        # 3. Execute trajectory with PID control
        self.pid.reset()  # Clear integrator for clean tracking
        commanded_joint_trace = []
        actual_joint_trace = []
        replay_joint_trace = [np.asarray(sample, dtype=np.float64).copy() for sample in trajectory]
        
        # For drawing mode: track actual position at each waypoint boundary
        waypoint_cartesian_errors_mm = []
        actual_path_xyz = []  # To visualize the physically drawn line
        # We start checking at index 1 because the reset moved us to idx 0
        segment_boundary_idx = 1
        SEGMENT_STEPS = 20 if self.mode == 'drawing' else 0

        def _sleep_to_rate(start_time: float):
            """Enforce fixed control rate for a single tick."""
            elapsed = time.time() - start_time
            if elapsed < TRAJECTORY_DT:
                time.sleep(TRAJECTORY_DT - elapsed)

        for i, q_desired in enumerate(trajectory):
            step_start_time = time.time()

            # Get current state
            q_actual, _ = self._get_joint_state()
            
            # PID computes corrected position command
            q_command = self.pid.compute(q_desired, q_actual, dt=TRAJECTORY_DT)
            
            # Clip to joint limits
            q_command = np.clip(q_command, self.joint_limits_low, self.joint_limits_high)
            commanded_joint_trace.append(q_command.copy())
            
            # Send position command via ZERO-OVERHEAD Topic Publisher
            self.base_env._stream_joint_positions(q_command, duration=TRAJECTORY_DT)

            # Brief spin to process feedback
            self._spin(1, timeout=0.0)

            # Drawing mode: measure accuracy at each waypoint boundary
            if self.mode == 'drawing':
                # After the spin, base_env.joint_positions is typically more up to date
                # than the local q_actual snapshot from the top of this loop.
                q_after = np.array(self.base_env.joint_positions, dtype=np.float64)
                actual_joint_trace.append(q_after.copy())
                if i % 3 == 0:  # Record path with decent resolution
                    from rl.fk_ik_utils import fk
                    xyz_now = np.array(fk(q_after.tolist(), raw=True))
                    actual_path_xyz.append(xyz_now)
                if SEGMENT_STEPS > 0:
                    step_in_trajectory = i + 1  # 1-indexed
                    if (step_in_trajectory % SEGMENT_STEPS == 0 and
                        segment_boundary_idx < len(self.shape_xyz_waypoints)):
                        # Sharp corners can carry integral/derivative state from the previous
                        # segment into the next one. Soften PID memory before settling.
                        self.pid.soften_for_waypoint(
                            integral_decay=DRAWING_BOUNDARY_INTEGRAL_DECAY
                        )

                        # Hold the waypoint for a few ticks before evaluating boundary error.
                        # This reduces false spikes caused by measuring mid-transient.
                        wp_q = np.array(self.shape_joint_waypoints[segment_boundary_idx], dtype=np.float64)
                        for hold_k in range(DRAWING_WAYPOINT_HOLD_STEPS):
                            hold_start = time.time()
                            q_hold, _ = self._get_joint_state()
                            q_cmd_hold = self.pid.compute(wp_q, q_hold, dt=TRAJECTORY_DT)
                            q_cmd_hold = np.clip(q_cmd_hold, self.joint_limits_low, self.joint_limits_high)
                            commanded_joint_trace.append(q_cmd_hold.copy())
                            replay_joint_trace.append(wp_q.copy())
                            self.base_env._stream_joint_positions(q_cmd_hold, duration=TRAJECTORY_DT)
                            self._spin(1, timeout=0.0)

                            q_hold_after = np.array(self.base_env.joint_positions, dtype=np.float64)
                            actual_joint_trace.append(q_hold_after.copy())
                            if (hold_k % 2) == 0:
                                from rl.fk_ik_utils import fk
                                actual_path_xyz.append(np.array(fk(q_hold_after.tolist(), raw=True)))

                            _sleep_to_rate(hold_start)

                        from rl.fk_ik_utils import fk
                        q_eval = np.array(self.base_env.joint_positions, dtype=np.float64)
                        xyz_at_boundary = np.array(fk(q_eval.tolist(), raw=True))
                        target_xyz = self.shape_xyz_waypoints[segment_boundary_idx]
                        err_mm = np.linalg.norm(target_xyz - xyz_at_boundary) * 1000.0
                        waypoint_cartesian_errors_mm.append(err_mm)
                        segment_boundary_idx += 1
            else:
                q_after = np.array(self.base_env.joint_positions, dtype=np.float64)
                actual_joint_trace.append(q_after.copy())

            _sleep_to_rate(step_start_time)
        
        # 4. Wait for robot to settle is now handled by the appended trajectory steps
        # Just process any final ROS callbacks
        self._spin(5)
        
        # 5. Get final state and compute reward
        q_final, qd_final = self._get_joint_state()
        final_error = np.linalg.norm(self.current_q_goal - q_final)
        
        # Calculate strict Cartesian Reaching Error (in mm)
        from rl.fk_ik_utils import fk
        xyz_final = np.array(fk(q_final.tolist(), raw=True))
        cartesian_dist_mm = np.linalg.norm(self.current_target_xyz - xyz_final) * 1000.0
        
        # For drawing mode: use AVERAGE error across all waypoints
        if self.mode == 'drawing' and len(waypoint_cartesian_errors_mm) > 0:
            avg_wp_error_mm = np.mean(waypoint_cartesian_errors_mm)
            max_wp_error_mm = np.max(waypoint_cartesian_errors_mm)
        else:
            avg_wp_error_mm = cartesian_dist_mm
            max_wp_error_mm = cartesian_dist_mm
        
        # Get PID tracking metrics
        metrics = self.pid.get_episode_metrics()
        
        # Calculate total required movement distance to normalize IAE
        if self.mode == 'drawing' and len(self.shape_joint_waypoints) > 1:
            # Sum the cumulative joint-space distance across ALL shape segments
            total_movement_rad = 0.0
            for k in range(1, len(self.shape_joint_waypoints)):
                total_movement_rad += np.sum(np.abs(
                    self.shape_joint_waypoints[k] - self.shape_joint_waypoints[k-1]
                ))
            total_movement_rad = max(total_movement_rad, 0.1)
        else:
            q_start = self.home_position
            total_movement_rad = max(np.sum(np.abs(self.current_q_goal - q_start)), 0.1)

        # Compute reward:
        # - reaching: endpoint Cartesian miss only (keep previous behavior)
        # - drawing: penalize both average and worst waypoint errors to reduce spikes
        if self.mode == 'drawing':
            reward_avg_mm = avg_wp_error_mm
            reward_max_mm = max_wp_error_mm
        else:
            reward_avg_mm = cartesian_dist_mm
            reward_max_mm = None

        reward = self._compute_reward(
            metrics,
            final_error,
            avg_wp_mm=reward_avg_mm,
            max_wp_mm=reward_max_mm,
            total_movement_rad=total_movement_rad,
        )
        normalized_iae = metrics['iae'] / total_movement_rad
        normalized_command_delta = metrics['command_delta'] / total_movement_rad
        normalized_command_jerk = metrics['command_jerk'] / total_movement_rad
        
        # Build final state
        next_state = self.get_state()
        replay_joint_trace_smoothed = self._smooth_replay_trace(replay_joint_trace)

        try:
            replay_plan = self.base_env.export_pi_replay_plan(
                joint_samples_rad=replay_joint_trace_smoothed,
                sample_dt=TRAJECTORY_DT,
            )
            replay_error = None
        except Exception as exc:
            replay_plan = None
            replay_error = str(exc)
        
        # Log results
        if self.mode == 'drawing':
            self._log(f"Result: err={np.degrees(final_error):.2f}° "
                      f"AvgWpMiss={avg_wp_error_mm:.1f}mm MaxWpMiss={max_wp_error_mm:.1f}mm "
                      f"({len(waypoint_cartesian_errors_mm)}/{len(self.shape_xyz_waypoints)} wps) "
                      f"IAE={metrics['iae']:.4f} R={reward:.2f}")
        else:
            self._log(f"Result: err={np.degrees(final_error):.2f}° "
                      f"CartesianMiss={cartesian_dist_mm:.1f}mm "
                      f"IAE={metrics['iae']:.4f} R={reward:.2f}")
        
        # Store gain history
        self.gain_history.append({
            'episode': self.episode_count,
            'Kp': gains['Kp'].copy(),
            'Ki': gains['Ki'].copy(),
            'Kd': gains['Kd'].copy(),
            'iae': metrics['iae'],
            'final_error': final_error,
            'reward': reward,
            'target_xyz': self.current_target_xyz.copy(),
        })
        
        # Info dict
        info = {
            'iae': metrics['iae'],
            'normalized_iae': normalized_iae,
            'effort': metrics['effort'],
            'command_delta': metrics['command_delta'],
            'command_jerk': metrics['command_jerk'],
            'normalized_command_delta': normalized_command_delta,
            'normalized_command_jerk': normalized_command_jerk,
            'final_error': final_error,
            'cartesian_dist_mm': avg_wp_error_mm if self.mode == 'drawing' else cartesian_dist_mm,
            'mean_error': metrics['mean_error'],
            'max_error': metrics['max_error'],
            'gains': gains,
            'episode': self.episode_count,
            'target_xyz': self.current_target_xyz.copy(),
            'total_movement_rad': total_movement_rad,
            'control_backend': self.control_backend_name,
            'trajectory_dt_sec': TRAJECTORY_DT,
            'start_joint_rad': q_start.copy(),
            'goal_joint_rad': self.current_q_goal.copy(),
            'commanded_trajectory_rad': [cmd.tolist() for cmd in commanded_joint_trace],
            'replay_trajectory_rad': [cmd.tolist() for cmd in replay_joint_trace_smoothed],
            'actual_joint_trace_rad': [q.tolist() for q in actual_joint_trace],
            'joint_trace_time_sec': [step_idx * TRAJECTORY_DT for step_idx in range(len(actual_joint_trace))],
            'replay_plan': replay_plan,
            'replay_export_error': replay_error,
        }
        
        if self.mode == 'drawing':
            info['waypoint_errors_mm'] = waypoint_cartesian_errors_mm
            info['avg_wp_error_mm'] = avg_wp_error_mm
            info['max_wp_error_mm'] = max_wp_error_mm
            info['actual_path_xyz'] = np.array(actual_path_xyz)
            info['target_shape_xyz'] = np.array(self.shape_xyz_waypoints)
            target_meta = {
                'shape_joint_waypoints': [wp.tolist() for wp in self.shape_joint_waypoints],
                'shape_xyz_waypoints': [np.asarray(wp, dtype=np.float64).tolist() for wp in self.shape_xyz_waypoints],
            }
        else:
            target_meta = {
                'target_xyz': self.current_target_xyz.copy().tolist(),
                'target_joint_goal': self.current_q_goal.copy().tolist(),
            }

        self.last_episode_artifact = {
            'episode': self.episode_count,
            'mode': self.mode,
            'control_backend': self.control_backend_name,
            'trajectory_dt_sec': TRAJECTORY_DT,
            'gains': {
                'Kp': gains['Kp'].copy().tolist(),
                'Ki': gains['Ki'].copy().tolist(),
                'Kd': gains['Kd'].copy().tolist(),
            },
            'metrics': {
                'reward': float(reward),
                'iae': float(metrics['iae']),
                'normalized_iae': float(normalized_iae),
                'effort': float(metrics['effort']),
                'command_delta': float(metrics['command_delta']),
                'command_jerk': float(metrics['command_jerk']),
                'normalized_command_delta': float(normalized_command_delta),
                'normalized_command_jerk': float(normalized_command_jerk),
                'final_error_rad': float(final_error),
                'cartesian_dist_mm': float(info['cartesian_dist_mm']),
            },
            'start_joint_rad': q_start.copy().tolist(),
            'goal_joint_rad': self.current_q_goal.copy().tolist(),
            'commanded_trajectory_rad': [cmd.tolist() for cmd in commanded_joint_trace],
            'replay_trajectory_rad': [cmd.tolist() for cmd in replay_joint_trace_smoothed],
            'actual_joint_trace_rad': [q.tolist() for q in actual_joint_trace],
            'joint_trace_time_sec': [step_idx * TRAJECTORY_DT for step_idx in range(len(actual_joint_trace))],
            'replay_plan': replay_plan,
            'replay_export_error': replay_error,
            'target_metadata': target_meta,
        }
        if self.mode == 'drawing':
            self.last_episode_artifact['actual_path_xyz'] = np.asarray(actual_path_xyz, dtype=np.float64).tolist()
        
        auto_shadow_replay = os.environ.get('PID_SHADOW_AUTO_REPLAY', '0').strip().lower() in {'1', 'true', 'yes', 'y'}
        if self.control_backend_name == 'sim_to_real_shadow' and commanded_joint_trace and auto_shadow_replay:
            self._log("🔄 Replaying episode trajectory on physical robot...")
            ok = self.base_env.motion_backend.replay_episode_trajectory(
                commanded_trace_rad=replay_joint_trace_smoothed,
                sample_dt=TRAJECTORY_DT,
                joint_limits_low=self.base_env.gazebo_limits_low,
                joint_limits_high=self.base_env.gazebo_limits_high,
            )
            if ok:
                self._log("✅ Pi replay complete")
            else:
                self._log("⚠️ Pi replay had issues (training continues)")
        elif self.control_backend_name == 'sim_to_real_shadow' and commanded_joint_trace:
            self._log("⏭️ Shadow hardware replay skipped; artifact/export is saved for manual Pi-local replay")

        # Single-step MDP: always done after one trajectory
        done = True
        
        return next_state, reward, done, info
    
    def _compute_reward(
        self,
        metrics: Dict,
        final_error: float,
        avg_wp_mm: float,
        max_wp_mm: Optional[float],
        total_movement_rad: float,
    ) -> float:
        """
        Compute reward from tracking metrics.
        
        Focuses heavily on Cartesian accuracy (reaching the board) rather than raw tracking lag,
        since tracking lag naturally scales with trajectory distance, injecting massive variance.
        """
        iae = metrics['iae']
        effort = metrics['effort']
        command_delta = metrics.get('command_delta', 0.0)
        command_jerk = metrics.get('command_jerk', 0.0)

        # Normalize IAE by total trajectory movement length
        # This prevents a long motion (high natural IAE) from automatically getting a terrible reward
        normalized_iae = iae / total_movement_rad
        normalized_command_delta = command_delta / total_movement_rad
        normalized_command_jerk = command_jerk / total_movement_rad

        reward = (
            -REWARD_NORM_IAE_W * normalized_iae
            -REWARD_EFFORT_W * effort
            -REWARD_SMOOTH_DELTA_W * normalized_command_delta
            -REWARD_SMOOTH_JERK_W * normalized_command_jerk
            -REWARD_WP_AVG_W * float(avg_wp_mm)
        )

        # Only apply max-waypoint penalty when provided (drawing mode).
        if max_wp_mm is not None:
            reward += -REWARD_WP_MAX_W * float(max_wp_mm)
        
        return reward
    
    def get_gain_history(self) -> list:
        """Return the full gain history for plotting."""
        return self.gain_history
    
    def get_best_gains(self) -> Optional[Dict]:
        """Return the gains that achieved the best (highest) reward."""
        if not self.gain_history:
            return None
        
        best = max(self.gain_history, key=lambda x: x['reward'])
        return {
            'Kp': best['Kp'],
            'Ki': best['Ki'],
            'Kd': best['Kd'],
            'reward': best['reward'],
            'iae': best['iae'],
            'episode': best['episode'],
        }

    def get_last_episode_artifact(self) -> Optional[Dict]:
        """Return the most recent deterministic replay artifact."""
        return self.last_episode_artifact

    def replay_artifact(self, artifact: Dict, label: str = 'pid_replay') -> bool:
        """Replay a previously exported trajectory on the configured hardware backend."""
        replay_plan = artifact.get('replay_plan')
        if not replay_plan:
            self._log(f"⚠️ Artifact '{label}' has no replay plan")
            return False

        start_joint = artifact.get('start_joint_rad')
        if start_joint is not None:
            self._log(f"Moving hardware to replay start pose for {label}...")
            start_positions_deg = self.base_env.motion_backend.mapper.gazebo_positions_to_pi_deg(
                np.array(start_joint, dtype=np.float64)
            )
            start_plan = {
                'replay_rate_hz': 0.5,
                'segments': [{
                    'duration_sec': 2.0,
                    'joint_names_pi': list(self.base_env.motion_backend.mapper.pi_joint_names),
                    'positions_deg': [start_positions_deg[name] for name in self.base_env.motion_backend.mapper.pi_joint_names],
                }],
            }
            self.base_env.replay_exported_plan(start_plan, label=f'{label}_start')
            self._spin(20)

        return self.base_env.replay_exported_plan(replay_plan, label=label)
