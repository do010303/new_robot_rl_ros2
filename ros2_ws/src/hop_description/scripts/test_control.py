#!/usr/bin/env python3
"""
Simple script to test controlling the HOP drone in Gazebo Harmonic via CLI.
Make sure Gazebo is running with the robot spawned before executing this!
"""
import subprocess
import time
import sys

def publish_joint_cmd(joint_name, position):
    topic = f"/model/hop/joint/{joint_name}/cmd_pos"
    msg_type = "gz.msgs.Double"
    msg_content = f"data: {position}"
    
    cmd = [
        "gz", "topic", "-t", topic, "-m", msg_type, "-p", msg_content
    ]
    
    print(f"Publishing {position} to {joint_name}...")
    subprocess.run(cmd)

if __name__ == "__main__":
    print("Testing Gazebo Harmonic Joint Controllers for HOP drone...")
    print("---")
    
    # Test arm poses
    commands = [
        ("arm_joint_1", 1.0),
        ("arm_joint_2", -0.5),
        ("arm_joint_3", 1.5),
        ("arm_joint_4", 0.5),
        ("arm_joint_5", -1.0),
        ("arm_joint_6", 1.0),
    ]
    
    for joint, pos in commands:
        publish_joint_cmd(joint, pos)
        time.sleep(0.5)
        
    print("\nTest completed! Check Gazebo GUI to see if the arm moved.")
