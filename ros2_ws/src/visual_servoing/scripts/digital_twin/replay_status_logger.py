#!/usr/bin/env python3
"""Mirror Pi replay status messages into a laptop-side JSONL deploy log."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ReplayStatusLogger(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("replay_status_logger")
        self.args = args
        os.makedirs(os.path.abspath(args.log_dir), exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = args.output or os.path.join(
            os.path.abspath(args.log_dir),
            f"pi_replay_status_log_{timestamp}.jsonl",
        )
        self.count = 0
        self.done = False
        self.sub = self.create_subscription(String, args.topic, self._status_cb, 10)
        self.get_logger().info(f"Listening on {args.topic}")
        self.get_logger().info(f"Writing laptop mirror log to {self.log_path}")

    def _status_cb(self, msg: String) -> None:
        self.count += 1
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {"event": "RAW", "data": msg.data}

        payload.setdefault("mirror_received_index", self.count)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")

        event = payload.get("event", "")
        status = payload.get("status", "")
        if event == "SEGMENT_DONE":
            self.get_logger().info(
                f"Ep {payload.get('episode')} Seg {payload.get('segment_idx')}: "
                f"{status} max_err={payload.get('max_err_deg')}deg"
            )
        elif event in {"START", "DONE"}:
            self.get_logger().info(f"{event}: {payload}")
        else:
            self.get_logger().info(str(payload))

        if event == "DONE" and self.args.exit_on_done:
            self.done = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Log Pi replay_status JSON messages on the laptop.")
    parser.add_argument("--topic", default="/pca9685_servo/replay_status")
    parser.add_argument("--log-dir", default=os.path.expanduser("~/new_rl_ros2/ros2_ws/src/visual_servoing/scripts/training_results/logs"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--exit-on-done", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = ReplayStatusLogger(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.get_logger().info(f"Saved {node.count} replay status messages to {node.log_path}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
