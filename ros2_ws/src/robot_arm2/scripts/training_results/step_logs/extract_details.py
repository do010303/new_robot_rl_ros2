import re

file_path = 'successful_episodes_detailed.txt'
output_path = 'detailed_waypoint_log.txt'

def parse_vector(line):
    # Extracts numbers from a string like "[ 1.2, 3.4, -5.6]"
    # Returns a string formatted for output
    nums = re.findall(r'[-+]?\d*\.\d+|\d+', line)
    return " ".join(nums)

def analyze_episodes(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(output_path, 'w') as out:
        current_episode = None
        current_joints = None
        current_ee = None
        current_target = None
        
        for line in lines:
            line = line.strip()
            
            # Detect Episode
            match_ep = re.search(r'EPISODE (\d+)', line)
            if match_ep:
                current_episode = match_ep.group(1)
                
            # Parse Step Data
            if line.startswith("Joints (deg):"):
                current_joints = parse_vector(line)
            elif line.startswith("EE After:"):
                current_ee = parse_vector(line)
            elif line.startswith("Target:"):
                current_target = parse_vector(line)
            
            # Detect Waypoint Reached
            match_wp = re.search(r'>>> WAYPOINT (\d+) REACHED! <<<', line)
            if match_wp:
                wp_num = match_wp.group(1)
                
                # Format output
                # "ep1 waypoints 1 reached - angles ... ee after ... target ..."
                if wp_num == "1":
                    prefix = f"ep{current_episode} "
                else:
                    prefix = f"{' ' * (len(current_episode) + 3)}" # Indent to align with "epX "
                
                # Aligning strictly with "epX waypoints Y ..." format
                # But user example:
                # ep1 waypoints 1 reached - angles ...
                #          waypoints 2 ...
                
                if int(wp_num) == 1:
                     out_line = f"ep{current_episode} waypoints {wp_num} reached - angles {current_joints}  ee after {current_ee} target {current_target}\n"
                else:
                     # 4 spaces for "epX " roughly? User used indentation. 
                     # Let's just use strict "epX" for start, and spaces for others.
                     # Actually, to be safe and clear, let's just make every line start with epX?
                     # User asked "no i meant it should be like..." with indentation.
                     # "ep1 ... \n         waypoints 2 ..."
                     indent = " " * len(f"ep{current_episode} ")
                     out_line = f"{indent}waypoints {wp_num} reached - angles {current_joints}  ee after {current_ee} target {current_target}\n"
                
                out.write(out_line)

    print(f"Detailed log generated in {output_path}")

if __name__ == "__main__":
    analyze_episodes(file_path)
