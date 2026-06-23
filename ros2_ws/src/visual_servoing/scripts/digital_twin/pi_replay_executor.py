#!/usr/bin/env python3
"""Execute an exported Pi replay plan locally and log hardware status."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from digital_twin.replay_plan_json import load_replay_plan_json


def _duration_msg(duration_sec: float):
    from builtin_interfaces.msg import Duration

    sec = int(duration_sec)
    nanosec = int((duration_sec - sec) * 1e9)
    return Duration(sec=sec, nanosec=nanosec)


class PiReplayExecutor(Node):
    def __init__(self, plan: Dict, args: argparse.Namespace):
        super().__init__("pi_replay_executor")
        self.plan = plan
        self.args = args
        self.joint_names = list(plan["joint_names"])
        self.tolerance_deg = float(args.tolerance_deg if args.tolerance_deg is not None else plan.get("joint_error_tolerance_deg", 2.0))

        self.trajectory_pub = self.create_publisher(JointTrajectory, args.trajectory_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.home_client = self.create_client(Trigger, args.home_service)
        self.joint_state_sub = self.create_subscription(JointState, args.joint_state_topic, self._joint_state_cb, 10)

        self.latest_joint_state_deg: Optional[Dict[str, float]] = None
        self.latest_joint_state_time = 0.0

        log_dir = os.path.abspath(args.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"pi_replay_executor_log_{timestamp}.jsonl")

    def _joint_state_cb(self, msg: JointState) -> None:
        values = {}
        for name, value in zip(msg.name, msg.position):
            angle = float(value)
            if abs(angle) < 6.3:
                angle = angle * 180.0 / 3.141592653589793
            values[name] = angle
        if values:
            self.latest_joint_state_deg = values
            self.latest_joint_state_time = time.monotonic()

    def _publish_status(self, payload: Dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.status_pub.publish(msg)

    def _write_log(self, payload: Dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _build_trajectory(self, names: List[str], positions_deg: List[float], duration_sec: float) -> JointTrajectory:
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in positions_deg]
        point.time_from_start = _duration_msg(duration_sec)
        traj.points = [point]
        return traj

    def _call_home(self) -> None:
        if self.args.no_home:
            return
        if not self.home_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"Home service not available: {self.args.home_service}")
            return
        future = self.home_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result and result.success:
            self.get_logger().info("Home service completed")
        elif result:
            self.get_logger().warn(f"Home service returned failure: {result.message}")
        else:
            self.get_logger().warn("Home service timed out")

    def _measure_error(self, names: List[str], positions_deg: List[float]) -> tuple[str, Dict[str, float], float, float]:
        now = time.monotonic()
        if self.latest_joint_state_deg is None:
            return "LOST", {}, 0.0, 999.0

        feedback_age = now - self.latest_joint_state_time
        errors = {}
        for name, cmd in zip(names, positions_deg):
            if name not in self.latest_joint_state_deg:
                continue
            errors[name] = abs(float(cmd) - float(self.latest_joint_state_deg[name]))

        if len(errors) != len(names) or feedback_age > self.args.feedback_timeout:
            return "LOST", errors, max(errors.values()) if errors else 0.0, feedback_age

        max_error = max(errors.values()) if errors else 0.0
        return ("OK" if max_error <= self.tolerance_deg else "LAG"), errors, max_error, feedback_age

    def run(self) -> bool:
        self.get_logger().info(
            f"Executing plan: {self.plan.get('segment_count')} segments, "
            f"{self.plan.get('replay_rate_hz')}Hz, tolerance={self.tolerance_deg:.1f}deg"
        )
        self.get_logger().info(f"Pi local log: {self.log_path}")

        session = {
            "event": "START",
            "schema": self.plan.get("schema"),
            "source_artifact": self.plan.get("source_artifact"),
            "segment_count": self.plan.get("segment_count"),
            "replay_rate_hz": self.plan.get("replay_rate_hz"),
            "tolerance_deg": self.tolerance_deg,
            "log_path": self.log_path,
        }
        self._publish_status(session)
        self._write_log(session)

        self._call_home()
        time.sleep(self.args.start_settle)

        ok_count = 0
        lag_count = 0
        lost_count = 0
        started = time.monotonic()

        for episode in range(self.args.episodes):
            for seg in self.plan["segments"]:
                names = list(seg.get("joint_names") or self.joint_names)
                positions_deg = [float(v) for v in seg["positions_deg"]]
                duration_sec = float(seg["duration_sec"])
                idx = int(seg["idx"])

                if self.args.dry_run:
                    self.get_logger().info(f"DRY segment {idx}: {dict(zip(names, positions_deg))} dur={duration_sec:.3f}s")
                else:
                    self.trajectory_pub.publish(self._build_trajectory(names, positions_deg, duration_sec))

                time.sleep(duration_sec + self.args.feedback_settle)
                rclpy.spin_once(self, timeout_sec=0.0)
                status, errors, max_error, feedback_age = self._measure_error(names, positions_deg)
                if status == "OK":
                    ok_count += 1
                elif status == "LAG":
                    lag_count += 1
                else:
                    lost_count += 1

                payload = {
                    "event": "SEGMENT_DONE",
                    "episode": episode + 1,
                    "segment_idx": idx,
                    "duration_sec": duration_sec,
                    "cmd_deg": {name: pos for name, pos in zip(names, positions_deg)},
                    "err_deg": errors,
                    "max_err_deg": round(max_error, 4),
                    "feedback_age_sec": round(feedback_age, 4),
                    "status": status,
                    "elapsed_sec": round(time.monotonic() - started, 4),
                }
                self._publish_status(payload)
                self._write_log(payload)
                if self.args.print_segments:
                    self.get_logger().info(
                        f"Ep {episode + 1}/{self.args.episodes} Seg {idx + 1}/{len(self.plan['segments'])}: "
                        f"{status} max_err={max_error:.2f}deg feedback_age={feedback_age:.3f}s"
                    )

        summary = {
            "event": "DONE",
            "ok_segments": ok_count,
            "lag_segments": lag_count,
            "lost_segments": lost_count,
            "total_segments": ok_count + lag_count + lost_count,
            "elapsed_sec": round(time.monotonic() - started, 4),
            "log_path": self.log_path,
        }
        self._publish_status(summary)
        self._write_log(summary)
        self.get_logger().info(
            f"Done: OK={ok_count}, LAG={lag_count}, LOST={lost_count}, log={self.log_path}"
        )
        return lag_count == 0 and lost_count == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pi_replay_plan_v1 JSON plan using local Pi timing.")
    parser.add_argument("--plan", required=True, help="Path to exported pi_replay_plan_*.json.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--tolerance-deg", type=float, default=None)
    parser.add_argument("--trajectory-topic", default="/pca9685_servo/trajectory")
    parser.add_argument("--joint-state-topic", default="/pca9685_servo/joint_states")
    parser.add_argument("--status-topic", default="/pca9685_servo/replay_status")
    parser.add_argument("--home-service", default="/pca9685_servo/home")
    parser.add_argument("--log-dir", default=os.path.expanduser("~/ros2_ws/replay_logs"))
    parser.add_argument("--feedback-timeout", type=float, default=0.5)
    parser.add_argument("--feedback-settle", type=float, default=0.03)
    parser.add_argument("--start-settle", type=float, default=1.0)
    parser.add_argument("--no-home", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-segments", action="store_true")
    args = parser.parse_args()

    plan = load_replay_plan_json(args.plan)
    rclpy.init()
    node = PiReplayExecutor(plan, args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
