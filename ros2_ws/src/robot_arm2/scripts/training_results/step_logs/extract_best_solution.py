import json
import math
import sys
import os
import torch
import numpy as np

# Add path to scripts folder to import NeuralIK
script_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(script_dir, "../../rl"))
checkpoints_dir = os.path.abspath(os.path.join(script_dir, "../../checkpoints"))

sys.path.insert(0, os.path.dirname(rl_dir)) 

try:
    from rl.neural_ik import NeuralIK
except ImportError:
    sys.path.insert(0, "/home/ducanh/new_rl_ros2/ros2_ws/src/robot_arm2/scripts")
    from rl.neural_ik import NeuralIK

input_file = "step_log_20260126_155218.jsonl"
output_file = "best_waypoints.txt"
nik_path = os.path.join(checkpoints_dir, "neural_ik.pth")

def to_degrees(rads):
    return [math.degrees(r) for r in rads]

def format_vector_space(vec, precision=4):
    if hasattr(vec, 'tolist'):
        vec = vec.tolist()
    return " ".join([f"{x:.{precision}f}" for x in vec])

def main():
    print(f"Reading {input_file}...")
    
    # Load NeuralIK
    if not os.path.exists(nik_path):
        print("Error: Neural IK model not found.")
        return
        
    neural_ik = NeuralIK(device=torch.device('cpu'))
    neural_ik.load(nik_path)
    
    episodes_data = {}
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
    
    # Identify complete episodes
    complete_episodes = set()
    for ep, steps in episodes_data.items():
        for step in steps:
            if step.get('shape_complete', False):
                complete_episodes.add(ep)
                break
    
    print(f"Analyzing {len(complete_episodes)} complete episodes...")
    
    # Store candidates for each waypoint
    # waypoint_candidates[wp_idx] = list of {error, details_str}
    waypoint_candidates = {i: [] for i in range(1, 11)}
    
    sorted_episodes = sorted([ep for ep in episodes_data.keys() if ep in complete_episodes])
    
    for ep in sorted_episodes:
        steps = episodes_data[ep]
        steps.sort(key=lambda x: x['step'])
        
        prev_reached = 0
        
        for i, step in enumerate(steps):
            reward = step.get('reward', -1)
            reached = step.get('waypoints_reached', 0)
            shape_complete = step.get('shape_complete', False)
            
            if (reward == 0 and reached > prev_reached) or shape_complete:
                
                target = np.array(step.get('target', [0,0,0]))
                ee = np.array(step.get('ee_after', [0,0,0]))
                
                # Calculate Error
                error = np.linalg.norm(target - ee)
                
                # Get Joints
                rads = None
                if i + 1 < len(steps):
                    rads = steps[i + 1]['joints']
                else:
                    # Last step: Predict
                    rads = neural_ik.predict(ee)
                
                if rads is None:
                    continue

                degs = to_degrees(rads)
                
                # Format Data
                deg_str = format_vector_space(degs, 2)
                ee_str = format_vector_space(ee, 4)
                target_str = format_vector_space(target, 4)
                
                wp_idx = reached
                # Handle edge case where last step logs differently
                if shape_complete and wp_idx == 0: 
                     wp_idx = 10 # Should be 10 if complete
                
                candidate = {
                    'episode': ep,
                    'error': error,
                    'target': target_str,
                    'ee': ee_str,
                    'cmd': deg_str
                }
                
                if wp_idx in waypoint_candidates:
                    waypoint_candidates[wp_idx].append(candidate)
                
                prev_reached = reached
                if shape_complete: break

    # Select Best
    print(f"Finding best solutions...")
    with open(output_file, 'w') as out:
        out.write("BEST SOLUTIONS PER WAYPOINT (Min Distance Error)\n\n")
        
        for wp in range(1, 11):
            candidates = waypoint_candidates[wp]
            if not candidates:
                out.write(f"Waypoint {wp}: No Data\n")
                continue
            
            # Sort by error (ascending)
            best = sorted(candidates, key=lambda x: x['error'])[0]
            
            out.write(f"Waypoint {wp} (Best from Ep {best['episode']})\n")
            out.write(f"  Error:     {best['error']*100:.4f} cm\n")
            out.write(f"  Target:    {best['target']}\n")
            out.write(f"  EE Pos:    {best['ee']}\n")
            out.write(f"  CMD (deg): {best['cmd']}\n")
            out.write("-" * 50 + "\n")

    print(f"Done! Written to {output_file}")

if __name__ == "__main__":
    main()
