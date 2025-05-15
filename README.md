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
