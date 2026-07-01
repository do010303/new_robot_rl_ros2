#!/usr/bin/env python3
"""
Digital Twin: Gazebo to Pi Realtime Mirror
==========================================
Subscribes to Gazebo's /joint_states (RADIANS)
and publishes JointState commands to Pi's /pca9685_servo/command (DEGREES).

Uses mapping for 6-DOF robot arm:
  Revolute 20 -> base
  Revolute 22 -> shoulder
  Revolute 23 -> elbow
  Revolute 26 -> wrist_roll
  Revolute 28 -> wrist_pitch
  Revolute 30 -> pen
"""

import argparse
import math
import sys
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Gazebo joint -> (Pi name, pi_home_deg, inverted)
REVERSE_MAPPING = {
    "Revolute 20": ("base",         90.0,  False),
    "Revolute 22": ("shoulder",     90.0,  False),
    "Revolute 23": ("elbow",        90.0,  False),
    "Revolute 26": ("wrist_roll",   90.0,  False),
    "Revolute 28": ("wrist_pitch",  90.0,  False),
    "Revolute 30": ("pen",          90.0,  False),
}

def rad_to_deg(rad):
    return rad * 180.0 / math.pi

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

class SimToPiMirror(Node):
    def __init__(self, rate_hz=10.0, deadband_deg=0.5):
        super().__init__('sim_to_pi_mirror')

        self.rate_hz = rate_hz
        self.deadband_deg = deadband_deg

        self.command_pub = self.create_publisher(
            JointState,
            "/pca9685_servo/command",
            10
        )

        self.js_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_callback,
            10
        )

        self.last_send_time = 0.0
        self.min_interval = 1.0 / self.rate_hz
        self.last_sent_positions = {}  # Track last sent positions (in degrees)
        self.msg_count = 0

        self.get_logger().info("🔄 Sim-to-Real Mirror (Gazebo -> Pi) started")
        self.get_logger().info(f"   Rate limited: {self.rate_hz} Hz")
        self.get_logger().info(f"   Dead-band: {self.deadband_deg}°")

    def gazebo_rad_to_pi_deg(self, gazebo_rad, home_deg, inverted):
        """Convert Gazebo radians to Pi servo degrees."""
        offset_deg = rad_to_deg(gazebo_rad)
        if inverted:
            offset_deg = -offset_deg
        pi_deg = home_deg + offset_deg
        return clamp(pi_deg, 0.0, 180.0)

    def joint_states_callback(self, msg: JointState):
        now = time.monotonic()
        if (now - self.last_send_time) < self.min_interval:
            return

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        has_significant_change = False

        # Build degree mappings for logging
        log_pairs = []

        for gz_name, position in zip(msg.name, msg.position):
            if gz_name in REVERSE_MAPPING:
                pi_name, home, inv = REVERSE_MAPPING[gz_name]
                pi_deg = self.gazebo_rad_to_pi_deg(position, home, inv)

                last = self.last_sent_positions.get(pi_name, None)
                delta = abs(pi_deg - last) if last is not None else 999.0
                log_pairs.append((pi_name, rad_to_deg(position), pi_deg, delta))

                if last is None or delta > self.deadband_deg:
                    cmd.name.append(pi_name)
                    cmd.position.append(pi_deg)
                    has_significant_change = True

        if not cmd.name or not has_significant_change:
            return

        self.command_pub.publish(cmd)
        self.last_send_time = now

        # Update last sent positions
        for name, pos in zip(cmd.name, cmd.position):
            self.last_sent_positions[name] = pos

        self.msg_count += 1
        if self.msg_count <= 5 or self.msg_count % 50 == 0:
            log_str = " | ".join([f"{name}: sim={sim_deg:.1f}°, cmd={cmd_deg:.1f}°, delta={delta:.1f}°" 
                                  for name, sim_deg, cmd_deg, delta in log_pairs])
            self.get_logger().info(f"🔄 Mirror Frame #{self.msg_count}:\n   {log_str}")

def main(args=None):
    # Parse CLI arguments if run directly (ROS arguments stripped by rclpy.init later)
    parser = argparse.ArgumentParser(description="Sim-to-Real Mirror Node")
    parser.add_argument("--rate-hz", type=float, default=10.0, help="Publish rate in Hz")
    parser.add_argument("--deadband-deg", type=float, default=0.5, help="Minimum joint angle change to trigger publish")
    
    # Strip ROS-specific arguments from sys.argv before parsing
    parsed_args, unknown = parser.parse_known_args()
    
    rclpy.init(args=args)
    node = SimToPiMirror(rate_hz=parsed_args.rate_hz, deadband_deg=parsed_args.deadband_deg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
