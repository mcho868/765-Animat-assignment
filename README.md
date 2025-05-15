# 765-Animat-Assignment

A Braitenberg-inspired multi-agent simulation environment for studying action selection and stress behaviors in simple agents.

## Overview

This project implements a simulation environment based on Braitenberg vehicles, with agents that respond to environmental stimuli like light and possess stress mechanisms that influence their behavior. The environment features:

- Light sources that agents can sense
- Food resources that agents can consume
- Obstacles that agents must navigate around
- Simple agents with different tropisms (light-seeking, light-avoiding)
- Complex stress-based agents that change behaviors based on internal state

The simulation demonstrates how complex behaviors can emerge from simple rules, without explicit action selection mechanisms, following the principles described in the research paper "Evolving Action Selection and Selective Attention Without Actions Attention or Selection."

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/765-Animat-assignment.git
cd 765-Animat-assignment
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Simulation

To run the simulation with default parameters:
```bash
python main.py
```

Command line options:
```
--headless        Run simulation without visualization
--agents N        Number of agents to simulate (default: 20)
--obstacles N     Number of obstacles in environment (default: 10)
--lights N        Number of light sources (default: 5)
--food N          Number of food sources (default: 5)
--time T          Maximum simulation time in seconds
--simple-ratio R  Ratio of simple agents to stress agents (0-1)
```

Example: Run for 60 seconds with 15 agents and 8 food sources:
```bash
python main.py --agents 15 --food 8 --time 60
```

## Project Structure

- `config/` - Configuration settings
- `core/` - Core simulation logic
  - `environment.py` - World setup, resources, physics
  - `simulator.py` - Main simulation loop, rendering
- `agents/` - Agent implementations
  - `base_agent.py` - Abstract Agent class with sensors/motors
  - `agent_logic.py` - Decision-making code (rules, behavior trees)
  - `simple_agent.py` - Basic Braitenberg vehicle
  - `stress_agent.py` - Complex agent with stress-based behaviors
- `utils/` - Utilities
  - `logger.py` - Logging for agent behaviors and stress
  - `math_utils.py` - Math helpers for vectors, collisions, etc.
- `main.py` - Entry point

## Controls

While the simulation is running:
- Press `ESC` to exit

## Agent Types

1. **Simple Agents**: Basic Braitenberg vehicles with three behavior types:
   - Type 1: Random movement
   - Type 2: Light-seeking (positive phototropism)
   - Type 3: Light-avoiding (negative phototropism)

2. **Stress Agents**: More complex agents that change behavior based on stress levels:
   - Low stress: Exploration mode
   - Medium stress: Seek light
   - High stress: Seek food
   - Very high stress: Freeze/minimal movement

## Research Background

This simulation is based on the principles described in the paper "Evolving Action Selection and Selective Attention Without Actions Attention or Selection," which explores how complex behaviors can emerge without explicit action selection mechanisms. The approach builds on Valentino Braitenberg's work on synthetic psychology and "vehicles."

## License

This project is for educational purposes and part of the 765 course assignment.
