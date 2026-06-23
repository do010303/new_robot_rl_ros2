#!/usr/bin/env python3
"""
Pi-local replay executor for quantized robot-arm JSON plans.

This node intentionally has no dependency on the laptop training project. Copy a
pi_replay_plan_v1 JSON file to the Raspberry Pi, run this node next to
wicom_roboarm_unified_node.py, and it will publish timed JointTrajectory
segments using the Pi's local clock.
"""

import argparse
import json
import math
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def _duration_msg(duration_sec: float) -> Duration:
    sec = int(duration_sec)
    nanosec = int((float(duration_sec) - sec) * 1e9)
    return Duration(sec=sec, nanosec=nanosec)


def _load_plan(path: str) -> dict:
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        plan = json.load(f)

    if plan.get("schema") != "pi_replay_plan_v1":
        raise ValueError(f"Unsupported replay schema: {plan.get('schema')}")
    if not plan.get("joint_names"):
        raise ValueError("Replay plan missing joint_names")
    if not plan.get("segments"):
        raise ValueError("Replay plan has no segments")

    joint_names = list(plan["joint_names"])
    for idx, seg in enumerate(plan["segments"]):
        names = list(seg.get("joint_names") or joint_names)
        positions = list(seg.get("positions_deg", []))
        duration = float(seg.get("duration_sec", 0.0))
        if len(names) != len(positions):
            raise ValueError(
                f"Segment {idx} has {len(names)} joint names but {len(positions)} positions"
            )
        if duration <= 0.0:
            raise ValueError(f"Segment {idx} has invalid duration_sec={duration}")
        for name, pos in zip(names, positions):
            value = float(pos)
            if value < 0.0 or value > 180.0:
                raise ValueError(f"Segment {idx} joint {name} outside [0, 180] deg: {value}")
    return plan


class PiReplayExecutor(Node):
    def __init__(self, args, plan):
        super().__init__("pi_replay_executor")
        self.args = args
        self.plan = plan
        self.joint_names = list(plan["joint_names"])
        self.tolerance_deg = float(args.tolerance_deg if args.tolerance_deg is not None else plan.get("joint_error_tolerance_deg", 2.0))

        self.trajectory_pub = self.create_publisher(JointTrajectory, args.trajectory_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.joint_state_sub = self.create_subscription(
            JointState,
            args.joint_state_topic,
            self._joint_state_cb,
            10,
        )
        self.home_client = self.create_client(Trigger, args.home_service)

        self.latest_joint_state_deg = None
        self.latest_joint_state_time = 0.0

        log_dir = os.path.abspath(os.path.expanduser(args.log_dir))
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"pi_replay_executor_log_{timestamp}.jsonl")

    def _joint_state_cb(self, msg: JointState):
        values = {}
        for name, value in zip(msg.name, msg.position):
            angle = float(value)
            # wicom_roboarm publishes radians; keep degree compatibility for
            # older nodes or hand-made test publishers.
            if abs(angle) < 6.3:
                angle = math.degrees(angle)
            values[name] = angle
        if values:
            self.latest_joint_state_deg = values
            self.latest_joint_state_time = time.monotonic()

    def _publish_status(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.status_pub.publish(msg)

    def _write_log(self, payload: dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _status(self, payload: dict):
        self._publish_status(payload)
        self._write_log(payload)

    def _home(self):
        if self.args.no_home:
            return
        if not self.home_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn(f"Home service not available: {self.args.home_service}")
            return
        future = self.home_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            self.get_logger().warn("Home service timed out")
        elif result.success:
            self.get_logger().info("Home service completed")
        else:
            self.get_logger().warn(f"Home service failed: {result.message}")

    def _build_trajectory(self, names, positions_deg, duration_sec):
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in positions_deg]
        point.time_from_start = _duration_msg(duration_sec)
        traj.points = [point]
        return traj

    def _measure_error(self, names, positions_deg):
        if self.latest_joint_state_deg is None:
            return "LOST", {}, 0.0, 999.0

        feedback_age = time.monotonic() - self.latest_joint_state_time
        errors = {}
        for name, cmd in zip(names, positions_deg):
            if name in self.latest_joint_state_deg:
                errors[name] = abs(float(cmd) - float(self.latest_joint_state_deg[name]))

        if len(errors) != len(names) or feedback_age > self.args.feedback_timeout:
            return "LOST", errors, max(errors.values()) if errors else 0.0, feedback_age

        max_error = max(errors.values()) if errors else 0.0
        return ("OK" if max_error <= self.tolerance_deg else "LAG"), errors, max_error, feedback_age

    def run(self):
        replay_rate_override = None
        if self.args.replay_rate_hz is not None:
            replay_rate_override = max(0.001, float(self.args.replay_rate_hz))
        effective_replay_rate = (
            replay_rate_override
            if replay_rate_override is not None
            else self.plan.get("replay_rate_hz")
        )
        self.get_logger().info(
            f"Pi-local replay: plan={self.args.plan} segments={len(self.plan['segments'])} "
            f"episodes={self.args.episodes} replay_rate={effective_replay_rate}Hz "
            f"tolerance={self.tolerance_deg:.1f}deg"
        )
        self.get_logger().info(f"Writing log: {self.log_path}")

        start_payload = {
            "event": "START",
            "source_artifact": self.plan.get("source_artifact"),
            "segment_count": len(self.plan["segments"]),
            "plan_replay_rate_hz": self.plan.get("replay_rate_hz"),
            "effective_replay_rate_hz": effective_replay_rate,
            "duration_source": "override_replay_rate_hz" if replay_rate_override is not None else "plan_duration_sec",
            "tolerance_deg": self.tolerance_deg,
            "log_path": self.log_path,
        }
        self._status(start_payload)

        ok_count = 0
        lag_count = 0
        lost_count = 0
        started = time.monotonic()

        for episode in range(self.args.episodes):
            self._home()
            time.sleep(max(0.0, float(self.args.start_settle)))

            for seg in self.plan["segments"]:
                names = list(seg.get("joint_names") or self.joint_names)
                positions_deg = [float(v) for v in seg["positions_deg"]]
                plan_duration_sec = float(seg["duration_sec"])
                duration_sec = (1.0 / replay_rate_override) if replay_rate_override is not None else plan_duration_sec
                seg_idx = int(seg.get("idx", 0))

                if self.args.dry_run:
                    self.get_logger().info(
                        f"DRY Ep {episode + 1} Seg {seg_idx}: "
                        f"{dict(zip(names, positions_deg))} dur={duration_sec:.3f}s"
                    )
                else:
                    self.trajectory_pub.publish(
                        self._build_trajectory(names, positions_deg, duration_sec)
                    )

                deadline = time.monotonic() + duration_sec + max(0.0, self.args.feedback_settle)
                while time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.01)

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
                    "segment_idx": seg_idx,
                    "duration_sec": duration_sec,
                    "plan_duration_sec": plan_duration_sec,
                    "effective_replay_rate_hz": effective_replay_rate,
                    "cmd_deg": {name: pos for name, pos in zip(names, positions_deg)},
                    "err_deg": errors,
                    "max_err_deg": round(max_error, 4),
                    "feedback_age_sec": round(feedback_age, 4),
                    "status": status,
                    "elapsed_sec": round(time.monotonic() - started, 4),
                }
                self._status(payload)

                if self.args.print_segments:
                    self.get_logger().info(
                        f"Ep {episode + 1}/{self.args.episodes} "
                        f"Seg {seg_idx + 1}/{len(self.plan['segments'])}: "
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
        self._status(summary)
        self.get_logger().info(
            f"Done: OK={ok_count} LAG={lag_count} LOST={lost_count} log={self.log_path}"
        )


def _parse_args():
    parser = argparse.ArgumentParser(description="Execute pi_replay_plan_v1 JSON on the Raspberry Pi.")
    parser.add_argument("--plan", required=True, help="Path to pi_replay_plan_v1 JSON on the Pi.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--replay-rate-hz", type=float, default=None, help="Override replay speed on the Pi. If omitted, use duration_sec stored in JSON.")
    parser.add_argument("--tolerance-deg", type=float, default=None)
    parser.add_argument("--trajectory-topic", default="/pca9685_servo/trajectory")
    parser.add_argument("--joint-state-topic", default="/pca9685_servo/joint_states")
    parser.add_argument("--status-topic", default="/pca9685_servo/replay_status")
    parser.add_argument("--home-service", default="/pca9685_servo/home")
    parser.add_argument("--log-dir", default="~/ros2_ws/replay_logs")
    parser.add_argument("--feedback-timeout", type=float, default=0.5)
    parser.add_argument("--feedback-settle", type=float, default=0.03)
    parser.add_argument("--start-settle", type=float, default=1.0)
    parser.add_argument("--no-home", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-segments", action="store_true")
    args, ros_args = parser.parse_known_args()
    return args, ros_args


def main():
    args, ros_args = _parse_args()
    plan = _load_plan(args.plan)
    rclpy.init(args=ros_args)
    node = PiReplayExecutor(args, plan)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warn("Replay interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
