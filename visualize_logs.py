"""
Visualize agent stress levels and positions from log files.
"""
import os
import glob
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize agent stress logs")
    
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory containing log files")
    parser.add_argument("--output", type=str, default="stress_visualization.mp4", help="Output video file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second in output video")
    
    return parser.parse_args()

def load_stress_logs(log_dir):
    """Load stress logs from the specified directory."""
    # Find the most recent stress log file
    stress_logs = sorted(glob.glob(os.path.join(log_dir, "stress_log_*.csv")))
    
    if not stress_logs:
        print(f"No stress logs found in {log_dir}")
        return None
    
    latest_log = stress_logs[-1]
    print(f"Loading stress log: {latest_log}")
    
    # Parse the log file
    data = defaultdict(list)
    
    with open(latest_log, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            timestamp = float(row[0])
            agent_id = int(row[1])
            stress = float(row[2])
            pos_x = float(row[3])
            pos_y = float(row[4])
            
            data[timestamp].append({
                'agent_id': agent_id,
                'stress': stress,
                'position': (pos_x, pos_y)
            })
    
    # Convert to sorted list of timestamps and data
    timestamps = sorted(data.keys())
    return timestamps, data

def visualize_stress_over_time(timestamps, data, output_file="stress_over_time.png"):
    """Visualize average stress level over time."""
    # Calculate average stress at each timestamp
    avg_stress = []
    for t in timestamps:
        stresses = [agent['stress'] for agent in data[t]]
        avg_stress.append(np.mean(stresses))
    
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, avg_stress)
    plt.xlabel('Time (s)')
    plt.ylabel('Average Stress Level')
    plt.title('Average Agent Stress Over Time')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    
    print(f"Saved stress over time plot to {output_file}")

def animate_agent_positions(timestamps, data, output_file="agent_positions.mp4", fps=30):
    """Create an animation of agent positions colored by stress level."""
    # Find bounds for the plot
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for t in timestamps:
        for agent in data[t]:
            x, y = agent['position']
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    
    # Add some padding
    pad = 50
    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the first frame
    scatter = ax.scatter([], [], c=[], cmap='coolwarm', vmin=0, vmax=100, s=100)
    
    # Set axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Agent Positions and Stress Levels')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Stress Level')
    
    # Animation function
    def update(frame_idx):
        if frame_idx >= len(timestamps):
            return scatter,
            
        t = timestamps[frame_idx]
        positions = [agent['position'] for agent in data[t]]
        stresses = [agent['stress'] for agent in data[t]]
        
        if not positions:
            return scatter,
            
        x, y = zip(*positions)
        
        scatter.set_offsets(np.column_stack((x, y)))
        scatter.set_array(np.array(stresses))
        
        ax.set_title(f'Agent Positions and Stress Levels (Time: {t:.2f}s)')
        
        return scatter,
    
    # Create animation
    animation = FuncAnimation(
        fig, update, frames=len(timestamps), interval=1000/fps, blit=True
    )
    
    # Save animation
    animation.save(output_file, fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.close()
    
    print(f"Saved agent position animation to {output_file}")

def main():
    """Main function to visualize stress logs."""
    args = parse_arguments()
    
    # Load stress logs
    result = load_stress_logs(args.log_dir)
    if not result:
        return
        
    timestamps, data = result
    
    # Visualize stress over time
    visualize_stress_over_time(timestamps, data)
    
    # Animate agent positions
    animate_agent_positions(timestamps, data, args.output, args.fps)

if __name__ == "__main__":
    main() 