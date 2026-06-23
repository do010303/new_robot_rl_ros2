#!/usr/bin/env python3
"""Helpers for exporting and validating Pi-local replay plan JSON files."""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from rl.control_backends import GazeboPiMapper


DEFAULT_GAZEBO_LIMITS_LOW = np.array([
    -3.1415, -3.1415, -3.1415, -1.5708, -3.1415, -3.1415
], dtype=np.float64)
DEFAULT_GAZEBO_LIMITS_HIGH = np.array([
    3.1415, 3.1415, 3.1415, 1.5708, 3.1415, 3.1415
], dtype=np.float64)


def _trajectory_from_replay_plan(mapper: GazeboPiMapper, replay_plan: Dict[str, Any]) -> List[np.ndarray]:
    trajectory = []
    for seg in replay_plan.get("segments", []):
        names = seg.get("joint_names_pi", [])
        positions = seg.get("positions_deg", [])
        pos_deg_dict = {name: float(pos) for name, pos in zip(names, positions)}
        pos_rad = np.zeros(len(mapper.gazebo_joint_names), dtype=np.float64)
        for gz_idx, gz_name in enumerate(mapper.gazebo_joint_names):
            _, pi_name, home_deg, inverted = mapper.gazebo_lookup[gz_name]
            if pi_name in pos_deg_dict:
                pos_rad[gz_idx] = mapper.pi_deg_to_gazebo_rad(
                    pos_deg_dict[pi_name],
                    home_deg,
                    inverted,
                )
        trajectory.append(pos_rad)
    return trajectory


def load_artifact_trajectory(artifact_path: str, mapper: Optional[GazeboPiMapper] = None) -> tuple[Dict[str, Any], List[np.ndarray], float]:
    """Load the best available replay trajectory from a PID artifact."""
    mapper = mapper or GazeboPiMapper()
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)

    trajectory_list = artifact.get("replay_trajectory_rad", [])
    if not trajectory_list:
        trajectory_list = artifact.get("commanded_trajectory_rad", [])

    if trajectory_list:
        trajectory = [np.asarray(cmd, dtype=np.float64) for cmd in trajectory_list]
    else:
        trajectory = _trajectory_from_replay_plan(mapper, artifact.get("replay_plan", {}))

    if not trajectory:
        raise ValueError(f"Artifact has no replay_trajectory_rad, commanded_trajectory_rad, or replay_plan segments: {artifact_path}")

    sample_dt = float(artifact.get("trajectory_dt_sec", 0.02))
    return artifact, trajectory, sample_dt


def build_replay_plan_json(
    artifact_path: str,
    replay_rate_hz: float,
    output_path: Optional[str] = None,
    joint_error_tolerance_deg: float = 2.0,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a saved trajectory artifact into an inspectable Pi replay plan JSON payload."""
    mapper = GazeboPiMapper()
    artifact, trajectory, sample_dt = load_artifact_trajectory(artifact_path, mapper=mapper)

    plan = mapper.export_pi_replay_plan(
        joint_samples_rad=trajectory,
        sample_dt=sample_dt,
        joint_limits_low=DEFAULT_GAZEBO_LIMITS_LOW,
        joint_limits_high=DEFAULT_GAZEBO_LIMITS_HIGH,
        replay_rate_hz=float(replay_rate_hz),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_mode = mode or artifact.get("mode", "unknown")
    segments = []
    t_from_start = 0.0
    for idx, seg in enumerate(plan["segments"]):
        duration = float(seg["duration_sec"])
        segments.append({
            "idx": idx,
            "sample_index": int(seg.get("sample_index", idx)),
            "t_from_start_sec": round(t_from_start, 6),
            "duration_sec": duration,
            "joint_names": list(seg["joint_names_pi"]),
            "positions_deg": [float(v) for v in seg["positions_deg"]],
        })
        t_from_start += duration

    payload = {
        "schema": "pi_replay_plan_v1",
        "created_at": timestamp,
        "source_artifact": os.path.abspath(artifact_path),
        "source_mode": resolved_mode,
        "source_dt_sec": float(sample_dt),
        "replay_rate_hz": float(replay_rate_hz),
        "joint_error_tolerance_deg": float(joint_error_tolerance_deg),
        "joint_names": list(mapper.pi_joint_names),
        "segment_count": len(segments),
        "duration_sec": round(t_from_start, 6),
        "downsample_stride": int(plan.get("downsample_stride", 1)),
        "lead_steps_applied": int(plan.get("lead_steps_applied", 0)),
        "original_start_joint_deg": [float(v) for v in plan.get("original_start_joint_deg", [])],
        "segments": segments,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return payload


def load_replay_plan_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    if plan.get("schema") != "pi_replay_plan_v1":
        raise ValueError(f"Unsupported replay plan schema: {plan.get('schema')}")
    validate_replay_plan(plan)
    return plan


def validate_replay_plan(plan: Dict[str, Any]) -> None:
    joint_names = plan.get("joint_names", [])
    if not joint_names:
        raise ValueError("Replay plan is missing joint_names")

    segments = plan.get("segments", [])
    if not segments:
        raise ValueError("Replay plan has no segments")

    for idx, seg in enumerate(segments):
        names = seg.get("joint_names") or joint_names
        positions = seg.get("positions_deg", [])
        duration = float(seg.get("duration_sec", 0.0))
        if duration <= 0.0:
            raise ValueError(f"Segment {idx} has non-positive duration_sec={duration}")
        if len(names) != len(positions):
            raise ValueError(f"Segment {idx} has {len(names)} joint names but {len(positions)} positions")
        for name, pos in zip(names, positions):
            value = float(pos)
            if value < 0.0 or value > 180.0:
                raise ValueError(f"Segment {idx} joint {name} is outside [0, 180] deg: {value}")

