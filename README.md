# Animat Evolution Simulation

A simulation environment for evolving animats with continuous, direct sensorimotor connections influenced by internal battery levels.

## Overview

This project implements a simulation environment for evolving autonomous agents (animats) using a genetic algorithm. The animats operate based on the principle that complex behaviors can emerge from simple, continuous sensorimotor links without explicit action selection mechanisms. Key features include:

- Animats with wheels for movement and multiple sensors (food, water, trap)
- Continuous sensorimotor links evolved via genetic algorithm
- Internal batteries that influence behavior
- 2D environment with food, water, and trap objects
- Visualization of evolved behaviors

The simulation demonstrates the emergence of complex behaviors solely from the coherent action of parallel sensorimotor processes, without using explicit internal representations or action selection mechanisms.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/animat-evolution.git
cd animat-evolution
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Simulation

To run the simulation with default parameters:
```bash
python main.py --run-best
```

Command line options:
```
--headless         Run in headless mode (no visualization)
--generations N    Number of generations to evolve
--population N     Population size for genetic algorithm
--seed N           Random seed for reproducibility
--run-best         Run simulation with the best evolved animat
--visualize-only   Only visualize, don't evolve
```

Example: Run evolution for 50 generations with a population of 100:
```bash
python main.py --generations 50 --population 100 --run-best
```

## Project Structure

- `config/` - Configuration settings
- `core/` - Core simulation logic
  - `environment.py` - 2D world, objects, physics
  - `simulator.py` - Simulation loop, rendering, evolution
- `agents/` - Agent implementations
  - `base_agent.py` - Animat implementation with wheels, sensors, batteries
  - `agent_logic.py` - Genetic algorithm for evolving animats
- `utils/` - Utilities
  - `logger.py` - Logging for battery levels, behaviors, evolution
  - `math_utils.py` - Math helpers for vectors, collisions
- `main.py` - Entry point

## Animat Architecture

The Animat has:
- Two wheels for movement
- Six sensors (food, water, and trap detection on both left and right sides)
- 18 direct sensorimotor links (3 per sensor, each connecting to the wheel on the same side)
- Two batteries (internal energy levels) that decay over time and influence behavior

## Genetic Encoding

Each sensorimotor link is defined by 9 parameters:
1. Initial output offset
2. Gradient 1
3. Threshold 1
4. Gradient 2
5. Threshold 2
6. Gradient 3
7. Slope modulation (battery-influenced)
8. Offset modulation (battery-influenced)
9. Battery number (0=battery1, 1=battery2)

The genome also includes 2 sigmoid thresholds for left and right wheels, resulting in a total genome size of 9 × 18 + 2 = 164 integers.

## Environment

The environment contains:
- 3 food sources (restore battery 1)
- 3 water sources (restore battery 2)
- 9 traps (collision = death)

## Visualization

When the simulation runs in visualization mode, you can see:
- Green circles: Food sources
- Blue circles: Water sources
- Red circles: Traps
- Yellow circles: Animats

Battery levels are displayed as colored bars above each animat.

## License

This project is for educational purposes.

# Manual Mode

A new interactive manual mode has been added that allows you to control an animat directly with keyboard controls.

## Running Manual Mode

```bash
python main.py --manual
```

## Controls

- **WASD** or **Arrow Keys**: Move and turn the animat
- **ESC**: Quit the simulation
- **R**: Restart the game (when animat dies)

## Gameplay

- **Green circles**: Food sources (restore left battery)
- **Blue circles**: Water sources (restore right battery)  
- **Red circles**: Traps (avoid these - they kill the animat instantly!)

## Objective

Survive as long as possible by:
1. Avoiding red traps
2. Collecting green food to restore your left battery
3. Collecting blue water to restore your right battery
4. Managing your energy - both batteries drain over time

The game ends when you hit a trap or run out of energy (either battery reaches 0). Your survival time is tracked and displayed in real-time.

## UI Information

The manual mode displays:
- Control instructions
- Real-time survival time
- Battery levels with visual bars
- Current position
- Color-coded battery status (green=good, orange=low, red=critical)
