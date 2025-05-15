# Simulation Architecture

This document outlines the architecture of the Braitenberg-inspired simulation environment.

## Core Components

### Environment
The environment (`core/environment.py`) is responsible for:
- Managing all entities (agents, obstacles, light sources, food sources)
- Handling physics and collision detection
- Providing sensing capabilities for agents

### Simulator
The simulator (`core/simulator.py`) is responsible for:
- Running the main simulation loop
- Rendering the environment using PyGame
- Ticking all entities for updates
- Handling user input and events

## Agent Architecture

Agents follow a sensor-motor architecture inspired by Braitenberg vehicles:

```
           ┌───────────────┐
           │   Sensors     │
           │ (light, food) │
           └───────┬───────┘
                   │
                   ▼
┌──────────────────────────────┐
│       Decision Making         │
│  - Simple agent: Direct/Cross │
│    connections                │
│  - Stress agent: State-based  │
│    behavior                   │
└──────────────┬───────────────┘
               │
               ▼
         ┌───────────┐
         │   Motors  │
         └───────────┘
```

### Stress Mechanism

The stress mechanism works as follows:
1. Agents accumulate stress based on environmental conditions
2. Stress level determines behavior state
3. Different behaviors are activated based on the current state
4. Stress is reduced in favorable conditions (e.g., light for light-seeking agents, food consumption)

## Data Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│          │     │          │     │          │
│ Simulator│────▶│Environment│────▶│ Entities │
│          │     │          │     │          │
└──────────┘     └──────────┘     └──────────┘
      │                                 │
      │                                 │
      ▼                                 ▼
┌──────────┐                      ┌──────────┐
│          │                      │          │
│  Logger  │                      │  Agents  │
│          │                      │          │
└──────────┘                      └──────────┘
                                        │
                                        │
                                        ▼
                                 ┌──────────────┐
                                 │              │
                                 │ Agent Logic  │
                                 │              │
                                 └──────────────┘
```

## Simulation Loop

1. Process user input and events
2. Update environment state
3. Each agent:
   a. Senses environment
   b. Makes decisions based on sensory input
   c. Actuates motors based on decisions
   d. Updates position and internal state
4. Render updated state
5. Repeat 