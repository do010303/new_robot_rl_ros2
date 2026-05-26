#!/usr/bin/env python3
"""
Connection verification utility for visual_servoing digital twin.
Subscribes to '/pca9685_servo/joint_states' to check if the Pi is reachable.
"""

import sys
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class ConnectionVerifier(Node):
    def __init__(self):
        super().__init__('connection_verifier')
        self.received_msg = None
        self.sub = self.create_subscription(
            JointState,
            '/pca9685_servo/joint_states',
            self.callback,
            10
        )
        self.get_logger().info("📡 Waiting for /pca9685_servo/joint_states...")

    def callback(self, msg):
        self.received_msg = msg

def main(args=None):
    rclpy.init(args=args)
    node = ConnectionVerifier()

    timeout_sec = 5.0
    start_time = time.time()
    
    print("\n==================================================")
    print("📡 PRE-FLIGHT VERIFICATION: Checking Pi connection")
    print("==================================================")
    print(f"Waiting up to {timeout_sec} seconds for joint states...")

    while rclpy.ok() and (time.time() - start_time) < timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.received_msg is not None:
            break

    print("\n--------------------------------------------------")
    if node.received_msg is not None:
        print("✅ CONNECTION SUCCESSFUL!")
        print(f"Received message after {time.time() - start_time:.2f} seconds.")
        print(f"Joint names: {node.received_msg.name}")
        print(f"Positions: {[round(p, 4) for p in node.received_msg.position]}")
        print("--------------------------------------------------")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    else:
        print("❌ CONNECTION TIMEOUT!")
        print(f"Could not receive any messages on '/pca9685_servo/joint_states' within {timeout_sec}s.")
        print("Please verify:")
        print(" 1. The Raspberry Pi robot node is running.")
        print(" 2. Both machines are on the same network.")
        print(" 3. FastDDS is configured correctly with unicast peers in fastdds_twin.xml.")
        print(" 4. Environment variables FASTRTPS_DEFAULT_PROFILES_FILE and RMW_IMPLEMENTATION are set.")
        print("--------------------------------------------------")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

if __name__ == '__main__':
    main()
