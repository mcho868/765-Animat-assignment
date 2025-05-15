# Usage Guide

## Setup

Before running the simulation, make sure you have all the required dependencies installed:

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Run the setup script to create necessary directories
python setup.py
```

## Running the Simulation

The simulation can be run with default parameters using:

```bash
python main.py
```

### Command Line Options

You can customize the simulation using various command-line options:

```
--headless        Run without visualization (useful for batch experiments)
--agents N        Number of agents (default: 20)
--obstacles N     Number of obstacles (default: 10)
--lights N        Number of light sources (default: 5)
--food N          Number of food sources (default: 5)
--time T          Maximum simulation time in seconds
--simple-ratio R  Ratio of simple agents to stress agents (0-1)
```

### Examples

Run with 30 agents, mostly stress-based:
```bash
python main.py --agents 30 --simple-ratio 0.3
```

Run a short 30-second simulation with more food:
```bash
python main.py --time 30 --food 10
```

Run a headless simulation for data collection:
```bash
python main.py --headless --time 120
```

## Visualizing Results

After running the simulation, you can visualize the collected data:

```bash
python visualize_logs.py
```

This will generate:
1. A plot of average stress levels over time (`stress_over_time.png`)
2. An animation of agent positions colored by stress level (`stress_visualization.mp4`)

### Visualization Options

```
--log-dir DIR     Directory containing log files (default: logs)
--output FILE     Output video file (default: stress_visualization.mp4)
--fps N           Frames per second in output video (default: 30)
```

## Experiment Ideas

Here are some experiments you can try:

1. **Stress Threshold Effects**:
   - Modify `stress_agent.py` to change the stress thresholds for behavior switching
   - Compare how different thresholds affect agent survival and behavior

2. **Light vs. Food**:
   - Adjust the number of light and food sources
   - Observe how this affects the balance between different behaviors

3. **Population Dynamics**:
   - Run with different ratios of simple vs. stress agents
   - Observe which ones perform better in different environments

4. **Sensor Configuration**:
   - Modify sensor positions and angles in `base_agent.py`
   - Compare how different sensory setups affect behavior

5. **Environmental Challenge**:
   - Increase obstacle density
   - See how agents adapt to more challenging environments

## Log Analysis

Simulation logs are stored in the `logs` directory with timestamp-based filenames:
- `stress_log_*.csv`: Records stress levels and positions of all agents over time
- `behavior_log_*.csv`: Records agent behaviors, headings, and speeds
- `simulation_log_*.json`: Contains simulation settings and metrics

You can analyze these logs with custom scripts or use the provided visualization tool. 