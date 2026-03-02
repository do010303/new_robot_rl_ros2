#!/usr/bin/env python3
"""
Deploy Drawing Model on Raspberry Pi

Traces a 10-waypoint equilateral triangle using trained SAC + Neural IK.
Designed for drawing task deployment with waypoint-by-waypoint progress.

Features:
- Episode/step control matching training script
- Auto model loading from rl_deployment directory
- Comprehensive logging and analysis plots
- Trajectory visualization

Usage:
    python3 deploy_drawing_on_pi.py --episodes 5 --steps 100
    python3 deploy_drawing_on_pi.py --model actor_drawing_quant.onnx --ik neural_ik_quant.onnx
"""

import numpy as np
import time
import os
import sys
import argparse
import json
from datetime import datetime

# =============================================================================
# DRAWING CONFIGURATION (matches training)
# =============================================================================

# State dimensions for DRAWING task
STATE_DIM = 18  # joints(6) + EE(3) + target(3) + dist(3) + dist3d(1) + progress(1) + remaining(1)
ACTION_DIM = 3  # SAC outputs 3D delta direction
JOINT_DIM = 6   # Output joint angles

# Waypoint configuration
POINTS_PER_EDGE = 3   # 3 points per edge
TOTAL_WAYPOINTS = 10  # 3 edges × 3 + 1 return

# Triangle parameters (matches training)
SHAPE_SIZE = 0.15     # 15cm triangle
Y_PLANE = 0.20        # 20cm from ground
WAYPOINT_TOLERANCE = 0.005  # 0.5cm tolerance

# Control parameters
CONTROL_RATE_HZ = 5.0
STEP_SIZE = 0.15      # Max 15cm per step

# Joint limits (radians) - ±90°
JOINT_LIMITS = np.array([
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708]
])

# Default model names
DEFAULT_ACTOR_MODEL = "actor_drawing_quant.onnx"
DEFAULT_IK_MODEL = "neural_ik_quant.onnx"


def generate_triangle_waypoints():
    """Generate 10-waypoint equilateral triangle (matches training)"""
    height = SHAPE_SIZE * np.sqrt(3) / 2
    cx, cz = 0.0, 0.25  # Center
    
    # Vertices - START FROM TOP (apex)
    p1 = np.array([cx, Y_PLANE, cz + 2*height/3])          # Top (apex/START)
    p2 = np.array([cx - SHAPE_SIZE/2, Y_PLANE, cz - height/3])  # Bottom-left
    p3 = np.array([cx + SHAPE_SIZE/2, Y_PLANE, cz - height/3])  # Bottom-right
    
    corners = [p1, p2, p3, p1]  # TOP→BL→BR→TOP
    
    waypoints = []
    for edge in range(3):
        start = corners[edge]
        end = corners[edge + 1]
        for t in np.linspace(0, 1, POINTS_PER_EDGE, endpoint=False):
            point = start + t * (end - start)
            waypoints.append(point)
    
    # Return to start
    waypoints.append(p1)
    
    return np.array(waypoints)


# =============================================================================
# SIMPLE FORWARD KINEMATICS
# =============================================================================

def forward_kinematics(joints):
    """Simple FK for 6DOF arm - returns EE position [x, y, z]"""
    # Link lengths (approximate)
    L1 = 0.05   # Base to shoulder
    L2 = 0.10   # Shoulder to elbow
    L3 = 0.10   # Elbow to wrist1
    L4 = 0.08   # Wrist1 to wrist2
    L5 = 0.05   # Wrist2 to EE
    
    j1, j2, j3, j4, j5, j6 = joints
    
    # Simplified calculation
    r = L2 * np.cos(j2) + L3 * np.cos(j2 + j3) + L4 * np.cos(j2 + j3 + j4)
    
    x = r * np.sin(j1)
    y = L1 + L2 * np.sin(j2) + L3 * np.sin(j2 + j3) + L4 * np.sin(j2 + j3 + j4) + L5
    z = r * np.cos(j1) + 0.25
    
    return np.array([x, y, z])


# =============================================================================
# LOGGING AND ANALYSIS
# =============================================================================

class DeploymentLogger:
    """Comprehensive logging for deployment analysis"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Episode-level stats
        self.episode_rewards = []
        self.episode_waypoints = []
        self.episode_steps = []
        self.episode_success = []
        self.episode_min_distances = []
        
        # Step-level logs (per episode)
        self.current_episode_log = []
        self.all_trajectories = []
        
    def start_episode(self, episode_num):
        """Start logging a new episode"""
        self.current_episode_log = []
        self.current_episode_num = episode_num
        
    def log_step(self, step, ee_pos, target, distance, joints, action, waypoint_idx):
        """Log a single step"""
        self.current_episode_log.append({
            'step': step,
            'ee_pos': ee_pos.tolist(),
            'target': target.tolist(),
            'distance': float(distance),
            'joints_rad': joints.tolist(),
            'joints_deg': (np.degrees(joints) + 90.0).tolist(),
            'action': action.tolist() if action is not None else None,
            'waypoint_idx': waypoint_idx,
            'timestamp': time.time()
        })
        
    def end_episode(self, total_steps, waypoints_reached, success, total_reward, min_distance):
        """End episode and save stats"""
        self.episode_steps.append(total_steps)
        self.episode_waypoints.append(waypoints_reached)
        self.episode_success.append(success)
        self.episode_rewards.append(total_reward)
        self.episode_min_distances.append(min_distance)
        
        # Save trajectory
        if self.current_episode_log:
            ee_trajectory = [log['ee_pos'] for log in self.current_episode_log]
            self.all_trajectories.append(ee_trajectory)
            
    def save_logs(self):
        """Save all logs to files"""
        # Save episode summary CSV
        csv_path = os.path.join(self.log_dir, f'deployment_summary_{self.timestamp}.csv')
        with open(csv_path, 'w') as f:
            f.write("episode,steps,waypoints_reached,success,reward,min_distance_cm\n")
            for i in range(len(self.episode_steps)):
                f.write(f"{i+1},{self.episode_steps[i]},{self.episode_waypoints[i]},")
                f.write(f"{self.episode_success[i]},{self.episode_rewards[i]:.2f},")
                f.write(f"{self.episode_min_distances[i]*100:.2f}\n")
        
        # Save detailed JSON logs
        json_path = os.path.join(self.log_dir, f'deployment_detailed_{self.timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'config': {
                    'total_waypoints': TOTAL_WAYPOINTS,
                    'shape_size_cm': SHAPE_SIZE * 100,
                    'tolerance_cm': WAYPOINT_TOLERANCE * 100
                },
                'episode_stats': {
                    'steps': self.episode_steps,
                    'waypoints_reached': self.episode_waypoints,
                    'success': self.episode_success,
                    'rewards': self.episode_rewards,
                    'min_distances': self.episode_min_distances
                },
                'trajectories': self.all_trajectories
            }, f, indent=2)
        
        print(f"\n📝 Logs saved:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
        
        return csv_path, json_path
        
    def generate_plots(self):
        """Generate analysis plots"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for Pi
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib not installed. Skipping plots.")
            return None
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Deployment Analysis - {self.timestamp}', fontsize=14)
        
        episodes = list(range(1, len(self.episode_steps) + 1))
        
        # 1. Waypoints Reached per Episode
        ax1 = axes[0, 0]
        ax1.bar(episodes, self.episode_waypoints, color='steelblue', alpha=0.7)
        ax1.axhline(y=TOTAL_WAYPOINTS, color='g', linestyle='--', label=f'Target: {TOTAL_WAYPOINTS}')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Waypoints Reached')
        ax1.set_title('Waypoints Reached per Episode')
        ax1.legend()
        ax1.set_ylim(0, TOTAL_WAYPOINTS + 1)
        
        # 2. Steps per Episode  
        ax2 = axes[0, 1]
        ax2.bar(episodes, self.episode_steps, color='orange', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Steps per Episode')
        
        # 3. Success Rate (cumulative)
        ax3 = axes[0, 2]
        cumulative_success = np.cumsum(self.episode_success) / np.arange(1, len(self.episode_success) + 1) * 100
        ax3.plot(episodes, cumulative_success, 'g-', linewidth=2)
        ax3.fill_between(episodes, cumulative_success, alpha=0.3, color='green')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Cumulative Success Rate (%)')
        ax3.set_title('Cumulative Success Rate')
        ax3.set_ylim(0, 105)
        
        # 4. Minimum Distance per Episode
        ax4 = axes[1, 0]
        min_dist_cm = [d * 100 for d in self.episode_min_distances]
        ax4.bar(episodes, min_dist_cm, color='coral', alpha=0.7)
        ax4.axhline(y=WAYPOINT_TOLERANCE*100, color='g', linestyle='--', label=f'Tolerance: {WAYPOINT_TOLERANCE*100:.1f}cm')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Min Distance to Waypoint (cm)')
        ax4.set_title('Closest Approach per Episode')
        ax4.legend()
        
        # 5. Episode Rewards
        ax5 = axes[1, 1]
        ax5.plot(episodes, self.episode_rewards, 'b-o', markersize=4)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Reward')
        ax5.set_title('Episode Rewards')
        
        # 6. Trajectory Plot (last episode XZ plane)
        ax6 = axes[1, 2]
        if self.all_trajectories:
            last_traj = np.array(self.all_trajectories[-1])
            ax6.plot(last_traj[:, 0] * 100, last_traj[:, 2] * 100, 'b-', alpha=0.7, label='EE Path')
            ax6.scatter(last_traj[0, 0] * 100, last_traj[0, 2] * 100, c='green', s=100, marker='o', label='Start')
            ax6.scatter(last_traj[-1, 0] * 100, last_traj[-1, 2] * 100, c='red', s=100, marker='x', label='End')
            
            # Draw target triangle
            waypoints = generate_triangle_waypoints()
            triangle = np.vstack([waypoints[:, [0, 2]], waypoints[0, [0, 2]]]) * 100
            ax6.plot(triangle[:, 0], triangle[:, 1], 'g--', alpha=0.5, label='Target Triangle')
            
        ax6.set_xlabel('X (cm)')
        ax6.set_ylabel('Z (cm)')
        ax6.set_title('XZ Trajectory (Last Episode)')
        ax6.legend(loc='upper right', fontsize=8)
        ax6.axis('equal')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.log_dir, f'deployment_plots_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"   Plots: {plot_path}")
        return plot_path


# =============================================================================
# DRAWING DEPLOYMENT
# =============================================================================

class DrawingDeployment:
    """Deploy trained drawing model on real robot"""
    
    def __init__(self, actor_path, neural_ik_path=None, use_ros=True, log_dir=None):
        self.actor_path = actor_path
        self.neural_ik_path = neural_ik_path
        self.use_ros = use_ros
        
        # Model sessions
        self.ort_session = None
        self.nik_session = None
        
        # State
        self.joint_positions = np.zeros(6)
        self.waypoints = generate_triangle_waypoints()
        
        # Logger
        log_dir = log_dir or os.path.dirname(os.path.abspath(__file__))
        self.logger = DeploymentLogger(log_dir)
        
        print("="*70)
        print("🎨 DRAWING DEPLOYMENT")
        print("="*70)
        print(f"   Triangle: {TOTAL_WAYPOINTS} waypoints ({POINTS_PER_EDGE} per edge)")
        print(f"   Size: {SHAPE_SIZE*100:.0f}cm | Tolerance: {WAYPOINT_TOLERANCE*100:.1f}cm")
        print("="*70)
    
    def load_models(self):
        """Load ONNX models for inference"""
        try:
            import onnxruntime as ort
        except ImportError:
            print("❌ onnxruntime not installed. Install with: pip install onnxruntime")
            return False
        
        print("\n📦 Loading models...")
        
        # Load Actor ONNX
        if os.path.exists(self.actor_path):
            self.ort_session = ort.InferenceSession(self.actor_path)
            print(f"✅ Actor: {self.actor_path}")
        else:
            print(f"❌ Actor not found: {self.actor_path}")
            return False
        
        # Load Neural IK ONNX
        if self.neural_ik_path and os.path.exists(self.neural_ik_path):
            self.nik_session = ort.InferenceSession(self.neural_ik_path)
            print(f"✅ Neural IK: {self.neural_ik_path}")
        else:
            print("⚠️  Neural IK not loaded - will need external IK")
        
        return True
    
    def setup_ros(self):
        """Setup ROS2 interfaces"""
        if not self.use_ros:
            print("⚠️  Running in simulation mode (no ROS)")
            return True
        
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
            
            rclpy.init()
            self.node = rclpy.create_node('drawing_deployment')
            
            # Joint state subscriber
            self.joint_sub = self.node.create_subscription(
                JointState, '/pca9685_servo/joint_states', self._joint_callback, 10
            )
            
            # Joint command publisher
            self.joint_pub = self.node.create_publisher(
                JointState, '/pca9685_servo/command', 10
            )
            
            print("✅ ROS2 interfaces ready")
            print("   Subscribing to: /pca9685_servo/joint_states")
            print("   Publishing to: /pca9685_servo/command")
            
            # Warmup
            print("⏳ Warming up publisher...")
            time.sleep(1.0)
            print("✅ Publisher ready")
            return True
            
        except Exception as e:
            print(f"❌ ROS2 setup failed: {e}")
            self.use_ros = False
            return False
    
    def _joint_callback(self, msg):
        """Update joint positions from servo feedback"""
        if len(msg.position) >= 6:
            # Servo feedback is in radians from unified node
            self.joint_positions = np.array(msg.position[:6])
    
    def get_state(self, current_waypoint_idx, waypoints_reached):
        """Construct 18D state vector (matches training)"""
        ee_pos = forward_kinematics(self.joint_positions)
        target = self.waypoints[current_waypoint_idx]
        dist_xyz = target - ee_pos
        dist_3d = np.linalg.norm(dist_xyz)
        progress = waypoints_reached / TOTAL_WAYPOINTS
        remaining = TOTAL_WAYPOINTS - waypoints_reached
        
        state = np.concatenate([
            self.joint_positions,    # 6
            ee_pos,                  # 3
            target,                  # 3
            dist_xyz,                # 3
            [dist_3d],               # 1
            [progress],              # 1
            [remaining]              # 1
        ])
        
        return state.astype(np.float32)
    
    def run_actor(self, state):
        """Run SAC actor inference"""
        if self.ort_session is None:
            return np.zeros(ACTION_DIM)
        
        input_name = self.ort_session.get_inputs()[0].name
        output = self.ort_session.run(None, {input_name: state.reshape(1, -1)})
        return output[0][0]
    
    def run_neural_ik(self, target_xyz):
        """Run Neural IK to get joint angles"""
        if self.nik_session is None:
            return None
        
        input_name = self.nik_session.get_inputs()[0].name
        output = self.nik_session.run(None, {input_name: target_xyz.astype(np.float32).reshape(1, -1)})
        return output[0][0]
    
    def compute_target_position(self, action, ee_pos, waypoint):
        """Compute target XYZ from action"""
        direction = waypoint - ee_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0.001:
            direction_norm = direction / distance
            move_amount = (action[0] + 1) / 2 * STEP_SIZE
            fine_adjust = action[1:3] * 0.02
            
            delta = direction_norm * move_amount
            target_xyz = ee_pos + delta
            target_xyz[0] += fine_adjust[0]
            target_xyz[2] += fine_adjust[1] if len(fine_adjust) > 1 else 0
        else:
            target_xyz = waypoint
        
        # Clamp to safe bounds
        target_xyz = np.clip(target_xyz, [-0.15, 0.10, 0.05], [0.15, 0.30, 0.45])
        
        return target_xyz
    
    def send_joints(self, joints):
        """Send joint commands - publishes each joint as separate message"""
        joints = np.clip(joints, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        
        if self.use_ros:
            from sensor_msgs.msg import JointState
            
            joints_deg = np.degrees(joints) + 90.0
            joint_names = ['base', 'shoulder', 'elbow', 'wrist_roll', 'wrist_pitch', 'pen']
            
            for name, angle in zip(joint_names, joints_deg):
                msg = JointState()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.name = [name]
                msg.position = [float(angle)]
                self.joint_pub.publish(msg)
        else:
            self.joint_positions = joints
        
        return joints
    
    def run_episode(self, episode_num, max_steps):
        """Run a single episode"""
        # Reset state for new episode
        current_waypoint_idx = 0
        waypoints_reached = 0
        episode_reward = 0.0
        min_distance = float('inf')
        
        self.logger.start_episode(episode_num)
        
        print(f"\n{'='*60}")
        print(f"📍 Episode {episode_num}")
        print(f"{'='*60}")
        
        step = 0
        while step < max_steps and waypoints_reached < TOTAL_WAYPOINTS:
            step += 1
            
            # Get state
            state = self.get_state(current_waypoint_idx, waypoints_reached)
            ee_pos = forward_kinematics(self.joint_positions)
            waypoint = self.waypoints[current_waypoint_idx]
            
            # Run actor
            action = self.run_actor(state)
            
            # Compute target position
            target_xyz = self.compute_target_position(action, ee_pos, waypoint)
            
            # Run Neural IK
            joints = self.run_neural_ik(target_xyz)
            
            if joints is not None:
                self.send_joints(joints)
            
            # Wait for movement
            time.sleep(1.0 / CONTROL_RATE_HZ)
            
            # Spin ROS
            if self.use_ros:
                import rclpy
                rclpy.spin_once(self.node, timeout_sec=0.1)
            
            # Check waypoint
            ee_after = forward_kinematics(self.joint_positions)
            distance = np.linalg.norm(ee_after - waypoint)
            min_distance = min(min_distance, distance)
            
            # Log step
            self.logger.log_step(step, ee_after, waypoint, distance, 
                               self.joint_positions, action, current_waypoint_idx)
            
            # Reward (sparse, matches training)
            if distance <= WAYPOINT_TOLERANCE:
                waypoints_reached += 1
                episode_reward += 10.0  # Waypoint reward
                print(f"  ✅ WP{current_waypoint_idx + 1} reached! ({waypoints_reached}/{TOTAL_WAYPOINTS})")
                
                if waypoints_reached >= TOTAL_WAYPOINTS:
                    episode_reward += 100.0  # Completion bonus
                    break
                    
                current_waypoint_idx += 1
            else:
                episode_reward -= 0.1  # Step penalty
            
            # Progress output (every 10 steps)
            if step % 10 == 0:
                print(f"  Step {step:3d} | EE: [{ee_after[0]:.3f}, {ee_after[1]:.3f}, {ee_after[2]:.3f}] | "
                      f"Dist: {distance*100:.2f}cm | WP{current_waypoint_idx+1}/{TOTAL_WAYPOINTS}")
        
        # Episode complete
        success = waypoints_reached >= TOTAL_WAYPOINTS
        self.logger.end_episode(step, waypoints_reached, success, episode_reward, min_distance)
        
        # Summary
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"\n  {status} | Steps: {step} | Waypoints: {waypoints_reached}/{TOTAL_WAYPOINTS} | "
              f"Min Dist: {min_distance*100:.2f}cm | Reward: {episode_reward:.1f}")
        
        return success, waypoints_reached, episode_reward
    
    def run(self, num_episodes=1, max_steps=100):
        """Main deployment loop"""
        print(f"\n{'='*70}")
        print(f"🎨 STARTING DEPLOYMENT: {num_episodes} episodes, {max_steps} max steps each")
        print(f"{'='*70}")
        
        # Display waypoints
        print("\nWaypoints:")
        for i, wp in enumerate(self.waypoints):
            print(f"  WP{i+1}: [{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}]")
        
        total_successes = 0
        total_waypoints = 0
        
        start_time = time.time()
        
        for ep in range(1, num_episodes + 1):
            success, waypoints, reward = self.run_episode(ep, max_steps)
            total_successes += int(success)
            total_waypoints += waypoints
        
        elapsed = time.time() - start_time
        
        # Final summary
        print(f"\n{'='*70}")
        print(f"📊 DEPLOYMENT COMPLETE")
        print(f"{'='*70}")
        print(f"   Episodes: {num_episodes}")
        print(f"   Successes: {total_successes}/{num_episodes} ({100*total_successes/num_episodes:.1f}%)")
        print(f"   Total Waypoints: {total_waypoints}/{num_episodes * TOTAL_WAYPOINTS}")
        print(f"   Time: {elapsed:.1f}s ({elapsed/num_episodes:.1f}s per episode)")
        
        # Save logs and generate plots
        self.logger.save_logs()
        self.logger.generate_plots()
        
        return total_successes > 0
    
    def cleanup(self):
        """Cleanup resources"""
        if self.use_ros:
            try:
                import rclpy
                self.node.destroy_node()
                rclpy.shutdown()
            except:
                pass


def find_model(model_name, script_dir):
    """Find model file in standard locations"""
    # Direct path
    if os.path.exists(model_name):
        return model_name
    
    # Same directory as script
    path = os.path.join(script_dir, model_name)
    if os.path.exists(path):
        return path
    
    # onnx_models subdirectory
    path = os.path.join(script_dir, 'onnx_models', model_name)
    if os.path.exists(path):
        return path
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Deploy Drawing Model on Raspberry Pi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 deploy_drawing_on_pi.py --episodes 5 --steps 100
    python3 deploy_drawing_on_pi.py --model actor_drawing_quant.onnx --ik neural_ik_quant.onnx
    python3 deploy_drawing_on_pi.py --no-ros  # Simulation mode
        """
    )
    
    # Episode/step control
    parser.add_argument('--episodes', '-e', type=int, default=1,
                        help='Number of episodes to run (default: 1)')
    parser.add_argument('--steps', '-s', type=int, default=100,
                        help='Max steps per episode (default: 100)')
    
    # Model paths
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_ACTOR_MODEL,
                        help=f'SAC actor ONNX model (default: {DEFAULT_ACTOR_MODEL})')
    parser.add_argument('--ik', type=str, default=DEFAULT_IK_MODEL,
                        help=f'Neural IK ONNX model (default: {DEFAULT_IK_MODEL})')
    
    # Options
    parser.add_argument('--no-ros', action='store_true', 
                        help='Run in simulation mode (no ROS)')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for logs and plots')
    
    args = parser.parse_args()
    
    # Find script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Auto-find models
    actor_path = find_model(args.model, script_dir)
    ik_path = find_model(args.ik, script_dir)
    
    if actor_path is None:
        print(f"❌ Actor model not found: {args.model}")
        print(f"   Searched in: {script_dir}")
        return 1
    
    # Create deployment
    deployer = DrawingDeployment(
        actor_path=actor_path,
        neural_ik_path=ik_path,
        use_ros=not args.no_ros,
        log_dir=args.log_dir or script_dir
    )
    
    # Load models
    if not deployer.load_models():
        print("❌ Failed to load models")
        return 1
    
    # Setup ROS
    if not args.no_ros:
        if not deployer.setup_ros():
            print("❌ Failed to setup ROS")
            return 1
    
    try:
        # Run deployment
        success = deployer.run(num_episodes=args.episodes, max_steps=args.steps)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        success = False
    finally:
        deployer.cleanup()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
