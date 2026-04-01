#!/usr/bin/env python3
"""
Digital Twin Communication Test — LAPTOP side
==============================================
Run this on your LAPTOP after connecting to the Pi's Wi-Fi hotspot.

What it does:
  1. Publishes a heartbeat on /twin/ping (String) at 2 Hz.
  2. Subscribes to /twin/pong (String) — the Pi should echo back.
  3. Subscribes to /pca9685_servo/joint_states (JointState) — real arm telemetry.
  4. Prints latency and joint positions as they arrive.

Usage:
  # First, make sure both machines share the same ROS_DOMAIN_ID:
  export ROS_DOMAIN_ID=0
  export ROS_LOCALHOST_ONLY=0
  
  # Then run:
  python3 test_ros2_connection.py
"""

import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState


class ConnectionTestNode(Node):
    def __init__(self):
        super().__init__("twin_connection_test_laptop")

        # ── Heartbeat publisher ──
        self.ping_pub = self.create_publisher(String, "/twin/ping", 10)
        self.ping_timer = self.create_timer(0.5, self._send_ping)  # 2 Hz
        self.ping_seq = 0

        # ── Heartbeat subscriber (Pi echoes back) ──
        self.pong_sub = self.create_subscription(
            String, "/twin/pong", self._on_pong, 10
        )
        self.pong_count = 0
        self.last_rtt_ms = None

        # ── Real arm telemetry subscriber ──
        self.js_sub = self.create_subscription(
            JointState, "/pca9685_servo/joint_states", self._on_joint_states, 10
        )
        self.js_count = 0
        self.last_js_time = None

        # ── Status printer ──
        self.status_timer = self.create_timer(3.0, self._print_status)

        self.get_logger().info("=" * 60)
        self.get_logger().info("  LAPTOP Communication Test Node Started")
        self.get_logger().info("  Sending pings on /twin/ping ...")
        self.get_logger().info("  Listening for pongs on /twin/pong ...")
        self.get_logger().info("  Listening for joint_states on /pca9685_servo/joint_states ...")
        self.get_logger().info("=" * 60)

    def _send_ping(self):
        msg = String()
        self.ping_seq += 1
        msg.data = f"ping|{self.ping_seq}|{time.time():.6f}"
        self.ping_pub.publish(msg)

    def _on_pong(self, msg: String):
        self.pong_count += 1
        try:
            parts = msg.data.split("|")
            if len(parts) >= 3 and parts[0] == "pong":
                send_time = float(parts[2])
                rtt_ms = (time.time() - send_time) * 1000.0
                self.last_rtt_ms = rtt_ms
                self.get_logger().info(
                    f"✅ PONG received! seq={parts[1]}  RTT={rtt_ms:.1f} ms"
                )
        except Exception:
            self.get_logger().info(f"✅ PONG received (raw): {msg.data}")

    def _on_joint_states(self, msg: JointState):
        self.js_count += 1
        self.last_js_time = time.time()
        if self.js_count % 10 == 1:  # Print every 10th message
            names = list(msg.name)
            import math
            positions_deg = [f"{math.degrees(p):.1f}°" for p in msg.position]
            self.get_logger().info(
                f"🦾 Joint States #{self.js_count}: {dict(zip(names, positions_deg))}"
            )

    def _print_status(self):
        self.get_logger().info("─" * 50)
        self.get_logger().info(f"  Pings sent:      {self.ping_seq}")
        self.get_logger().info(f"  Pongs received:  {self.pong_count}")
        rtt_str = f"{self.last_rtt_ms:.1f} ms" if self.last_rtt_ms else "N/A"
        self.get_logger().info(f"  Last RTT:        {rtt_str}")
        self.get_logger().info(f"  JointStates rcv: {self.js_count}")

        if self.pong_count == 0 and self.ping_seq > 6:
            self.get_logger().warn(
                "⚠️  No pongs received! Check:\n"
                "    1. Pi is running test_ros2_connection_pi.py\n"
                "    2. Both have same ROS_DOMAIN_ID\n"
                "    3. ROS_LOCALHOST_ONLY=0 on both machines\n"
                "    4. Laptop is connected to Pi's Wi-Fi"
            )
        if self.js_count == 0 and self.ping_seq > 6:
            self.get_logger().warn(
                "⚠️  No joint_states received! Check:\n"
                "    1. Pi is running: ros2 launch wicom_roboarm wicom_roboarm.launch.py\n"
                "    2. Topic name is /pca9685_servo/joint_states"
            )
        self.get_logger().info("─" * 50)


def main():
    rclpy.init()
    node = ConnectionTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
