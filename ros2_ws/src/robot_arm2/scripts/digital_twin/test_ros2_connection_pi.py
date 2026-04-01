#!/usr/bin/env python3
"""
Digital Twin Communication Test — RASPBERRY PI side
=====================================================
Run this on your RASPBERRY PI.

What it does:
  1. Subscribes to /twin/ping (String) from the laptop.
  2. Echoes back on /twin/pong (String) so the laptop can measure RTT.

Usage:
  # On the Pi, make sure:
  export ROS_DOMAIN_ID=0
  export ROS_LOCALHOST_ONLY=0
  
  # Then run:
  python3 test_ros2_connection_pi.py
  
  # In a SEPARATE terminal, also launch the arm node:
  ros2 launch wicom_roboarm wicom_roboarm.launch.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PiConnectionTestNode(Node):
    def __init__(self):
        super().__init__("twin_connection_test_pi")

        # ── Subscribe to laptop pings ──
        self.ping_sub = self.create_subscription(
            String, "/twin/ping", self._on_ping, 10
        )

        # ── Publish pongs back ──
        self.pong_pub = self.create_publisher(String, "/twin/pong", 10)

        self.ping_count = 0

        self.get_logger().info("=" * 60)
        self.get_logger().info("  PI Communication Test Node Started")
        self.get_logger().info("  Listening for pings on /twin/ping ...")
        self.get_logger().info("  Will echo pongs on /twin/pong ...")
        self.get_logger().info("=" * 60)

    def _on_ping(self, msg: String):
        self.ping_count += 1

        # Echo back as pong, preserving the send timestamp
        try:
            parts = msg.data.split("|")
            if len(parts) >= 3 and parts[0] == "ping":
                pong = String()
                pong.data = f"pong|{parts[1]}|{parts[2]}"
                self.pong_pub.publish(pong)
                self.get_logger().info(
                    f"📡 Ping #{parts[1]} received, pong sent back!"
                )
        except Exception as e:
            self.get_logger().warn(f"Error processing ping: {e}")


def main():
    rclpy.init()
    node = PiConnectionTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
