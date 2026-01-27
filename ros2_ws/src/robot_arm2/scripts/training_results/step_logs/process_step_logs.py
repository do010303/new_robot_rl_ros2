import json
import math
import sys
import os
import torch
import numpy as np

# Add path to scripts folder to import NeuralIK
# process_step_logs.py is in .../training_results/step_logs
# neural_ik.py is in .../rl
# So we need to go up 3 levels to 'scripts'
script_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(script_dir, "../../rl"))
checkpoints_dir = os.path.abspath(os.path.join(script_dir, "../../checkpoints"))

sys.path.insert(0, os.path.dirname(rl_dir)) # Add 'scripts' to path so we can do 'from rl.neural_ik import NeuralIK'

try:
    from rl.neural_ik import NeuralIK
except ImportError:
    # Fallback if running from a different location
    sys.path.insert(0, "/home/ducanh/new_rl_ros2/ros2_ws/src/robot_arm2/scripts")
    from rl.neural_ik import NeuralIK

input_file = "step_log_20260126_155218.jsonl"
output_file = "filtered_step_log.txt"
nik_path = os.path.join(checkpoints_dir, "neural_ik.pth")

def to_degrees(rads):
    return [math.degrees(r) for r in rads]

def format_vector_space(vec, precision=2):
    if hasattr(vec, 'tolist'):
        vec = vec.tolist()
    return " ".join([f"{x:.{precision}f}" for x in vec])

def main():
    print(f"Reading {input_file}...")
    
    # Load NeuralIK
    print(f"Loading NeuralIK from {nik_path}...")
    if not os.path.exists(nik_path):
        print("Error: Neural IK model not found. Cannot compute final joints.")
        return
        
    neural_ik = NeuralIK(device=torch.device('cpu'))
    neural_ik.load(nik_path)
    
    episodes_data = {}
    
    # Read and group by episode
    try:
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    ep = entry['episode']
                    if ep not in episodes_data:
                        episodes_data[ep] = []
                    episodes_data[ep].append(entry)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return
    
    # Find episodes that completed the shape
    complete_episodes = set()
    for ep, steps in episodes_data.items():
        for step in steps:
            if step.get('shape_complete', False):
                complete_episodes.add(ep)
                break
    
    print(f"Found {len(complete_episodes)} episodes that completed the shape.")
                
    # Process only complete episodes
    with open(output_file, 'w') as out:
        out.write("Log format: Waypoint Reached -> Joint Angles (Deg) -> EE Position (m)\n")
        out.write(f"Only episodes that COMPLETED the shape ({len(complete_episodes)} total)\n")
        out.write("Joints: From NEXT step (if avail) or computed via NeuralIK (final step)\n")
        out.write("Copy the 'CMD' line to paste into 'train_robot.py' Manual Mode (Option 1)\n\n")
        
        sorted_episodes = sorted([ep for ep in episodes_data.keys() if ep in complete_episodes])
        
        count = 0
        for ep in sorted_episodes:
            steps = episodes_data[ep]
            # Sort steps by step number
            steps.sort(key=lambda x: x['step'])
            
            prev_reached = 0
            
            for i, step in enumerate(steps):
                reward = step.get('reward', -1)
                reached = step.get('waypoints_reached', 0)
                shape_complete = step.get('shape_complete', False)
                
                # Only process when reward = 0 (waypoint reached) AND waypoints_reached increased
                # OR if shape is complete (last step)
                if (reward == 0 and reached > prev_reached) or shape_complete:
                    
                    # 1. Get Target Position (Waypoint)
                    target = step.get('target', [0,0,0]) # This is the waypoint position
                    ee = step.get('ee_after', [0,0,0])   # Actual position reached
                    
                    # 2. Get Joint Angles
                    rads = None
                    
                    if i + 1 < len(steps):
                        # Intermediate step: Use joints from next step (accurate)
                        next_step = steps[i + 1]
                        rads = next_step['joints']
                    else:
                        # LAST STEP (Shape Complete): No next step.
                        # Compute joints using NeuralIK for the reached position (ee_after)
                        # The robot effectively reached 'ee_after', so we want joints for that.
                        # Note: 'action' in log is normalized [-1,1], but we want joints for 'ee_after'.
                        # NeuralIK.predict takes cartesian pos and gives joints.
                        # Since pure SAC+NeuralIK was used, likely joints_action = NeuralIK(target_pos_from_action)
                        # And resulting ee ~ target_pos_from_action ~ ee_after.
                        # So NeuralIK(ee_after) is a good approximation of the final pose.
                        
                        ee_np = np.array(ee)
                        rads = neural_ik.predict(ee_np)
                        
                        # Add note to log maybe?
                    
                    if rads is None:
                        continue

                    degs = to_degrees(rads)
                    
                    deg_str = format_vector_space(degs, 2)
                    ee_str = format_vector_space(ee, 4)
                    target_str = format_vector_space(target, 4)
                    
                    # Output clear block
                    wp_label = f"Waypoint {reached} Reached"
                    if shape_complete:
                        # Ensure we don't duplicate if 'reached' didn't increment but shape is done
                        # Actually 'reward=0' logic usually covers the last step too?
                        # If shape_complete is set, waypoints_reached likely matches total.
                        # Let's trust 'reached' value.
                        wp_label += " (Shape Complete)"
                    
                    out.write(f"Episode {ep} | {wp_label}\n")
                    out.write(f"  Target:    {target_str}\n")
                    out.write(f"  EE Pos:    {ee_str}\n")
                    out.write(f"  CMD (deg): {deg_str}\n")
                    out.write("-" * 50 + "\n")
                    
                    count += 1
                    
                    # Special handling for last step to avoid duplication if loop continues
                    # But usually 'reached > prev_reached' handles it.
                    # For last step, reached reaches max (10).
                    prev_reached = reached
                    
                    if shape_complete:
                        # Ensure we don't process same step twice if logic is weird
                        break

    print(f"Processed {count} waypoint events from {len(complete_episodes)} complete episodes.")
    print(f"Written to {output_file}")

if __name__ == "__main__":
    main()
