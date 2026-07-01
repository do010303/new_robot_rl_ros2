#!/usr/bin/env python3
"""
Connection verification utility for visual_servoing digital twin.
Subscribes to '/pca9685_servo/joint_states' to check if the Pi is reachable.
"""

import os
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
    ros_localhost_only = os.environ.get("ROS_LOCALHOST_ONLY", "not set (default 0)")
    ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "not set (default 0)")
    rmw_implementation = os.environ.get("RMW_IMPLEMENTATION", "not set")
    fastrtps_profile = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "not set")

    # Validate environment setup
    has_env_error = False
    print("\n==================================================")
    print("📡 PRE-FLIGHT VERIFICATION: Checking Pi connection")
    print("==================================================")
    print("🔍 Checking ROS2 Network Setup:")
    print(f"  ROS_LOCALHOST_ONLY            : {ros_localhost_only}")
    print(f"  ROS_DOMAIN_ID                 : {ros_domain_id}")
    print(f"  RMW_IMPLEMENTATION            : {rmw_implementation}")
    print(f"  FASTRTPS_DEFAULT_PROFILES_FILE : {fastrtps_profile}")
    print("--------------------------------------------------")

    # 1. Check ROS_LOCALHOST_ONLY
    if ros_localhost_only == "1":
        print("  ❌ ROS_LOCALHOST_ONLY is 1! (Should be 0)")
        print("     👉 Run: export ROS_LOCALHOST_ONLY=0")
        has_env_error = True
    else:
        print("  ✅ ROS_LOCALHOST_ONLY looks good.")

    # 2. Check RMW_IMPLEMENTATION
    if rmw_implementation != "rmw_fastrtps_cpp":
        print("  ❌ RMW_IMPLEMENTATION is not 'rmw_fastrtps_cpp'!")
        print("     👉 Run: export RMW_IMPLEMENTATION=rmw_fastrtps_cpp")
        has_env_error = True
    else:
        print("  ✅ RMW_IMPLEMENTATION looks good.")

    # 3. Check FASTRTPS_DEFAULT_PROFILES_FILE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    expected_xml = os.path.abspath(os.path.join(script_dir, "..", "..", "config", "fastdds_twin.xml"))
    
    if fastrtps_profile == "not set":
        print("  ❌ FASTRTPS_DEFAULT_PROFILES_FILE is not set!")
        print(f"     👉 Run: export FASTRTPS_DEFAULT_PROFILES_FILE={expected_xml}")
        has_env_error = True
    elif not os.path.exists(fastrtps_profile):
        print(f"  ❌ FASTRTPS_DEFAULT_PROFILES_FILE path does not exist: {fastrtps_profile}")
        print(f"     👉 Run: export FASTRTPS_DEFAULT_PROFILES_FILE={expected_xml}")
        has_env_error = True
    else:
        print("  ✅ FASTRTPS_DEFAULT_PROFILES_FILE looks good.")

    if has_env_error:
        print("--------------------------------------------------")
        print("⚠️  WARNING: Network environment errors detected.")
        print("   If you continue, ROS2 might not find the Pi.")
        print("--------------------------------------------------")

    rclpy.init(args=args)
    node = ConnectionVerifier()

    timeout_sec = 5.0
    start_time = time.time()
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
        print(" 1. The Raspberry Pi robot node is running (ros2 launch wicom_roboarm wicom_roboarm.launch.py).")
        print(" 2. Both machines are on the same network.")
        print(" 3. FastDDS is configured correctly with unicast peers in fastdds_twin.xml.")
        print(" 4. Environment variables FASTRTPS_DEFAULT_PROFILES_FILE and RMW_IMPLEMENTATION are set.")
        print(" 5. ROS_LOCALHOST_ONLY must be set to 0 on both machines!")
        print("--------------------------------------------------")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

if __name__ == '__main__':
    main()
