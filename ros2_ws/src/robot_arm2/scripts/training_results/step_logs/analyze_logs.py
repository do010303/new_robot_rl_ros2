import re

file_path = 'successful_episodes_detailed.txt'

def analyze_episodes(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split by episode separator
    # Assuming "===================================================================================================="
    # and "EPISODE <number>"
    
    episodes = re.split(r'={20,}\nEPISODE (\d+)\n={20,}', content)
    
    # episodes[0] is the header before first episode
    # episodes[1] is episode number, episodes[2] is content, episodes[3] is num, episodes[4] is content...
    
    summary = []
    
    count_shape_complete = 0
    total_episodes = 0
    
    for i in range(1, len(episodes), 2):
        ep_num = episodes[i]
        ep_content = episodes[i+1]
        total_episodes += 1
        
        # Check for waypoints
        waypoints = re.findall(r'>>> WAYPOINT (\d+) REACHED! <<<', ep_content)
        is_complete = "SHAPE COMPLETE" in ep_content
        
        if is_complete:
            count_shape_complete += 1
            
        summary.append({
            'episode': ep_num,
            'waypoints_reached': waypoints,
            'complete': is_complete
        })
        
    with open('filtered_report.txt', 'w') as out:
        out.write(f"Total Episodes Found in File: {total_episodes}\n")
        out.write(f"Episodes completing shape: {count_shape_complete}\n")
        out.write("-" * 50 + "\n")
        out.write(f"{'Episode':<10} | {'Waypoints Reached':<20} | {'Complete?':<10}\n")
        out.write("-" * 50 + "\n")
        for s in summary:
            wps = "All (1-10)" if len(s['waypoints_reached']) == 10 else ", ".join(s['waypoints_reached'])
            out.write(f"{s['episode']:<10} | {wps:<20} | {s['complete']}\n")
            
    print("Report generated in filtered_report.txt")

if __name__ == "__main__":
    analyze_episodes(file_path)
