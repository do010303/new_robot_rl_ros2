#!/usr/bin/env python3
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config (now that it is restored)
from drawing.drawing_config import TOTAL_WAYPOINTS, SHAPE_TYPE

def regenerate_plot_from_csv():
    # File to reupdate
    basename = "drawing_training_sac_drawing_neuralIK_20260126_222826"
    csv_file = os.path.join(os.path.dirname(__file__), 'training_results', 'csv', f'{basename}.csv')
    png_file = os.path.join(os.path.dirname(__file__), 'training_results', 'png', f'{basename}.png')
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return

    print(f"Loading data from {csv_file}...")
    
    episodes = []
    rewards = []
    waypoints_reached = []
    shape_completions = []
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header
        # Header: Episode, Reward, Waypoints_Reached, Shape_Complete, ...
        
        for row in reader:
            if not row: continue
            try:
                episodes.append(int(float(row[0])))
                rewards.append(float(row[1]))
                waypoints_reached.append(float(row[2]))
                shape_completions.append(int(float(row[3])))
            except ValueError:
                continue
                
    episodes = np.array(episodes)
    rewards = np.array(rewards)
    waypoints_reached = np.array(waypoints_reached)
    shape_completions = np.array(shape_completions)
    
    # Cumulative averages
    def cumulative_avg(data):
        return [np.mean(data[:i+1]) for i in range(len(data))]
    
    reward_avg = cumulative_avg(rewards)
    waypoints_avg = cumulative_avg(waypoints_reached)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f'Drawing Training Statistics - {SHAPE_TYPE.capitalize()} (Reupdated)\nTarget: {TOTAL_WAYPOINTS} Waypoints'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards (top-left)
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color='blue', linewidth=1.5, label='Episode Reward')
    ax.plot(episodes, reward_avg, color='darkblue', linewidth=3.0, label='Cumulative Average')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Waypoints Reached (top-right)
    ax = axes[0, 1]
    total_wp = TOTAL_WAYPOINTS # Should be 5 now
    ax.scatter(episodes, waypoints_reached, marker='o', color='green', s=40, alpha=0.6, label='Waypoints')
    ax.plot(episodes, waypoints_avg, color='darkgreen', linewidth=3.0, label='Cumulative Average')
    ax.axhline(y=total_wp, color='gold', linestyle='--', linewidth=2, label=f'Target ({total_wp})')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Waypoints Reached', fontsize=12)
    ax.set_title('Waypoints Reached per Episode', fontsize=14, fontweight='bold')
    ax.set_ylim([-1, total_wp + 2])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Shape Completion Rate (bottom-left)
    ax = axes[1, 0]
    completion_avg = cumulative_avg(shape_completions)
    completion_avg_pct = [c * 100 for c in completion_avg]
    
    complete_eps = [episodes[i] for i, c in enumerate(shape_completions) if c == 1]
    incomplete_eps = [episodes[i] for i, c in enumerate(shape_completions) if c == 0]
    
    if len(complete_eps) > 0:
        ax.scatter(complete_eps, [100]*len(complete_eps), marker='o', color='green', s=40, alpha=0.6, label='Complete')
    if len(incomplete_eps) > 0:
        ax.scatter(incomplete_eps, [0]*len(incomplete_eps), marker='x', color='red', s=40, alpha=0.6, label='Incomplete')
        
    ax.plot(episodes, completion_avg_pct, color='darkgreen', linewidth=3.0, label='Completion Rate %')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Completion (%)', fontsize=12)
    ax.set_title('Shape Completion Rate', fontsize=14, fontweight='bold')
    ax.set_ylim([-5, 105])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary Stats (bottom-right)
    ax = axes[1, 1]
    ax.axis('off')
    
    num_complete = sum(shape_completions)
    completion_rate = 100.0 * num_complete / len(shape_completions) if len(shape_completions) > 0 else 0
    best_waypoints = max(waypoints_reached) if len(waypoints_reached) > 0 else 0
    avg_waypoints_val = np.mean(waypoints_reached) if len(waypoints_reached) > 0 else 0
    
    summary_text = f"""
📊 Summary (Reupdated)
━━━━━━━━━━━━━━━━━━━━━━

Episodes: {len(episodes)}

Rewards:
  • Final Avg: {reward_avg[-1]:.2f}
  • Best: {max(rewards):.2f}

Waypoints (Target {total_wp}):
  • Best: {best_waypoints}/{total_wp}
  • Avg: {avg_waypoints_val:.1f}/{total_wp}

Shape Completion:
  • Completed: {num_complete}/{len(shape_completions)}
  • Rate: {completion_rate:.1f}%
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot (OVERWRITE)
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Reupdated plot saved to: {png_file}")

if __name__ == "__main__":
    regenerate_plot_from_csv()
