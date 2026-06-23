#!/usr/bin/env python3
"""Export a PID replay artifact into an inspectable Pi-local JSON plan."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from digital_twin.replay_plan_json import build_replay_plan_json


def _default_output_path(mode: str) -> str:
    scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(scripts_dir, "training_results", "replay_plans")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"pi_replay_plan_{mode}_{timestamp}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Pi-local replay plan JSON from a PID artifact.")
    parser.add_argument("--artifact", required=True, help="Path to pid_best_artifact_*.pkl or compatible replay artifact.")
    parser.add_argument("--mode", default=None, help="Optional mode label stored in the JSON metadata.")
    parser.add_argument("--rate", type=float, default=3.0, help="Replay rate in Hz. Start with 2 or 3; try 5 after stable.")
    parser.add_argument("--tolerance-deg", type=float, default=2.0, help="Joint error tolerance for Pi executor OK/LAG status.")
    parser.add_argument("--output", default=None, help="Output JSON path. Defaults to training_results/replay_plans/.")
    args = parser.parse_args()

    output = args.output or _default_output_path(args.mode or "artifact")
    payload = build_replay_plan_json(
        artifact_path=args.artifact,
        replay_rate_hz=args.rate,
        output_path=output,
        joint_error_tolerance_deg=args.tolerance_deg,
        mode=args.mode,
    )

    print("Exported Pi replay plan")
    print(f"  output       : {output}")
    print(f"  segments     : {payload['segment_count']}")
    print(f"  duration     : {payload['duration_sec']:.2f}s")
    print(f"  replay_rate  : {payload['replay_rate_hz']:.2f}Hz")
    print(f"  tolerance    : {payload['joint_error_tolerance_deg']:.2f}deg")
    print(f"  lead_steps   : {payload['lead_steps_applied']}")


if __name__ == "__main__":
    main()
