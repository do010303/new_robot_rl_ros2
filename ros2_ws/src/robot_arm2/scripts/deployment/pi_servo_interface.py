#!/usr/bin/env python3
"""
PCA9685 Servo Interface for 6DOF Robot Arm - ROS2 Humble

This node provides ROS2 services and topics to control 6 servos via PCA9685.
Designed for Raspberry Pi deployment with I2C connection.

Hardware:
    - Raspberry Pi (3B+/4/5)
    - PCA9685 16-channel PWM driver (I2C address 0x40)
    - 6 servos connected to channels 0-5

Topics:
    - /servo/joint_states (sensor_msgs/JointState): Current joint positions
    
Services:
    - /servo/set_positions (SetJointPositions): Set all 6 joint positions
    - /servo/enable (Trigger): Enable servos
    - /servo/disable (Trigger): Disable servos
    - /servo/home (Trigger): Move to home position

Usage:
    ros2 run robot_arm2 pi_servo_interface
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
import numpy as np
import time

# Try to import PCA9685 library (only works on Pi)
try:
    from adafruit_servokit import ServoKit
    HAS_HARDWARE = True
except ImportError:
    HAS_HARDWARE = False
    print("⚠️  PCA9685 library not found - running in simulation mode")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Servo channel mapping for 6DOF arm
SERVO_CHANNELS = [0, 1, 2, 3, 4, 5]  # PCA9685 channels for joints 1-6

# Joint names (matching URDF)
JOINT_NAMES = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']

# Joint limits in radians (from URDF)
JOINT_LIMITS = {
    'Joint 1': (-3.14159, 3.14159),    # ±180°
    'Joint 2': (-1.5708, 1.5708),      # ±90°
    'Joint 3': (-1.5708, 1.5708),      # ±90°
    'Joint 4': (-1.5708, 1.5708),      # ±90°
    'Joint 5': (-3.14159, 3.14159),    # ±180°
    'Joint 6': (-3.14159, 3.14159),    # ±180°
}

# Servo calibration: maps radians to servo degrees
# Format: (rad_min, rad_max) -> (servo_deg_min, servo_deg_max)
# Adjust these based on your physical servo mounting
SERVO_CALIBRATION = {
    0: {'rad_range': (-3.14159, 3.14159), 'deg_range': (0, 180), 'invert': False},
    1: {'rad_range': (-1.5708, 1.5708), 'deg_range': (0, 180), 'invert': False},
    2: {'rad_range': (-1.5708, 1.5708), 'deg_range': (0, 180), 'invert': True},
    3: {'rad_range': (-1.5708, 1.5708), 'deg_range': (0, 180), 'invert': False},
    4: {'rad_range': (-3.14159, 3.14159), 'deg_range': (0, 180), 'invert': False},
    5: {'rad_range': (-3.14159, 3.14159), 'deg_range': (0, 180), 'invert': False},
}

# Home position in radians
HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Control rate (Hz)
PUBLISH_RATE = 50.0


# ============================================================================
# CUSTOM SERVICE (inline definition for simplicity)
# ============================================================================

# We'll use a simple approach with Float64MultiArray for positions
from std_msgs.msg import Float64MultiArray


# ============================================================================
# SERVO INTERFACE NODE
# ============================================================================

class PiServoInterface(Node):
    """ROS2 node for PCA9685 servo control"""
    
    def __init__(self):
        super().__init__('pi_servo_interface')
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🤖 Pi Servo Interface - 6DOF Robot Arm")
        self.get_logger().info("=" * 60)
        
        # Initialize hardware
        self.enabled = False
        self.kit = None
        
        if HAS_HARDWARE:
            try:
                self.kit = ServoKit(channels=16)
                self.get_logger().info("✅ PCA9685 initialized on I2C")
                
                # Set PWM frequency for servos (50Hz typical)
                # Note: ServoKit handles this internally
                
            except Exception as e:
                self.get_logger().error(f"❌ Failed to initialize PCA9685: {e}")
                self.kit = None
        else:
            self.get_logger().warn("⚠️  Running in SIMULATION mode (no hardware)")
        
        # Current joint positions (radians)
        self.joint_positions = np.array(HOME_POSITION, dtype=np.float32)
        self.joint_velocities = np.zeros(6, dtype=np.float32)
        self.last_positions = self.joint_positions.copy()
        self.last_time = time.time()
        
        # Publisher: joint states
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/servo/joint_states',
            10
        )
        
        # Subscriber: position commands
        self.position_sub = self.create_subscription(
            Float64MultiArray,
            '/servo/command',
            self.command_callback,
            10
        )
        
        # Services
        self.enable_srv = self.create_service(
            Trigger,
            '/servo/enable',
            self.enable_callback
        )
        
        self.disable_srv = self.create_service(
            Trigger,
            '/servo/disable',
            self.disable_callback
        )
        
        self.home_srv = self.create_service(
            Trigger,
            '/servo/home',
            self.home_callback
        )
        
        # Timer for publishing joint states
        timer_period = 1.0 / PUBLISH_RATE
        self.timer = self.create_timer(timer_period, self.publish_joint_states)
        
        self.get_logger().info(f"📡 Publishing joint states at {PUBLISH_RATE} Hz")
        self.get_logger().info(f"📥 Listening for commands on /servo/command")
        self.get_logger().info("✅ Pi Servo Interface ready!")
    
    def rad_to_servo_deg(self, channel: int, rad: float) -> float:
        """Convert radians to servo degrees for a specific channel"""
        cal = SERVO_CALIBRATION[channel]
        rad_min, rad_max = cal['rad_range']
        deg_min, deg_max = cal['deg_range']
        
        # Clamp to range
        rad = np.clip(rad, rad_min, rad_max)
        
        # Linear interpolation
        ratio = (rad - rad_min) / (rad_max - rad_min)
        
        if cal['invert']:
            ratio = 1.0 - ratio
        
        deg = deg_min + ratio * (deg_max - deg_min)
        return float(deg)
    
    def servo_deg_to_rad(self, channel: int, deg: float) -> float:
        """Convert servo degrees to radians for a specific channel"""
        cal = SERVO_CALIBRATION[channel]
        rad_min, rad_max = cal['rad_range']
        deg_min, deg_max = cal['deg_range']
        
        # Clamp to range
        deg = np.clip(deg, deg_min, deg_max)
        
        # Linear interpolation
        ratio = (deg - deg_min) / (deg_max - deg_min)
        
        if cal['invert']:
            ratio = 1.0 - ratio
        
        rad = rad_min + ratio * (rad_max - rad_min)
        return float(rad)
    
    def set_servo_position(self, channel: int, angle_rad: float) -> bool:
        """Set servo position in radians"""
        if not self.enabled:
            self.get_logger().warn("Servos not enabled!", throttle_duration_sec=1.0)
            return False
        
        # Convert to servo degrees
        angle_deg = self.rad_to_servo_deg(channel, angle_rad)
        
        # Clamp to safe range
        angle_deg = np.clip(angle_deg, 0, 180)
        
        if self.kit is not None:
            try:
                self.kit.servo[channel].angle = angle_deg
                return True
            except Exception as e:
                self.get_logger().error(f"Failed to set servo {channel}: {e}")
                return False
        else:
            # Simulation mode - just update internal state
            return True
    
    def command_callback(self, msg: Float64MultiArray):
        """Handle position command"""
        if len(msg.data) != 6:
            self.get_logger().warn(f"Expected 6 positions, got {len(msg.data)}")
            return
        
        positions = np.array(msg.data, dtype=np.float32)
        
        # Apply joint limits
        for i, name in enumerate(JOINT_NAMES):
            min_rad, max_rad = JOINT_LIMITS[name]
            positions[i] = np.clip(positions[i], min_rad, max_rad)
        
        # Set servo positions
        success = True
        for i, pos in enumerate(positions):
            if not self.set_servo_position(i, pos):
                success = False
        
        if success:
            self.joint_positions = positions
            self.get_logger().debug(f"Set positions: {np.degrees(positions)}")
    
    def enable_callback(self, request, response):
        """Enable servos"""
        self.enabled = True
        response.success = True
        response.message = "Servos enabled"
        self.get_logger().info("✅ Servos ENABLED")
        return response
    
    def disable_callback(self, request, response):
        """Disable servos (release torque)"""
        self.enabled = False
        
        # Release all servos
        if self.kit is not None:
            for channel in SERVO_CHANNELS:
                try:
                    self.kit.servo[channel].angle = None  # Release
                except:
                    pass
        
        response.success = True
        response.message = "Servos disabled"
        self.get_logger().info("⏹️  Servos DISABLED")
        return response
    
    def home_callback(self, request, response):
        """Move to home position"""
        if not self.enabled:
            self.enabled = True  # Auto-enable
            self.get_logger().info("Auto-enabling servos for home")
        
        # Move to home
        for i, pos in enumerate(HOME_POSITION):
            self.set_servo_position(i, pos)
        
        self.joint_positions = np.array(HOME_POSITION, dtype=np.float32)
        
        response.success = True
        response.message = f"Moved to home: {HOME_POSITION}"
        self.get_logger().info("🏠 Moved to HOME position")
        return response
    
    def publish_joint_states(self):
        """Publish current joint states"""
        # Calculate velocities
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt > 0:
            self.joint_velocities = (self.joint_positions - self.last_positions) / dt
        
        self.last_positions = self.joint_positions.copy()
        self.last_time = current_time
        
        # Create message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = self.joint_positions.tolist()
        msg.velocity = self.joint_velocities.tolist()
        msg.effort = [0.0] * 6  # No torque sensing
        
        self.joint_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = PiServoInterface()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
