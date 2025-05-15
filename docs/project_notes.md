# Project Notes

## Braitenberg Vehicle Concept

This simulation is inspired by Valentino Braitenberg's thought experiments described in his book "Vehicles: Experiments in Synthetic Psychology" (1984). Braitenberg vehicles are simple agents that exhibit complex behaviors through direct connections between sensors and motors.

The key insight is that complex behaviors can emerge from simple rules without explicit action selection mechanisms, which aligns with the research paper "Evolving Action Selection and Selective Attention Without Actions, Attention or Selection."

## Implementation Notes

### Sensor-Motor Connectivity

- **Direct Connections**: Connect sensors directly to motors on the same side (e.g., left sensor to left motor). This often results in avoidance behaviors.
- **Cross Connections**: Connect sensors to motors on the opposite side (e.g., left sensor to right motor). This often results in seeking behaviors.

### Stress Mechanism

The stress mechanism serves as an implicit action selection system without explicit competition between actions:

1. Different stress levels trigger different behavioral patterns
2. Behaviors automatically shift based on internal state
3. No explicit arbitration or competition between behaviors
4. Emergent prioritization (high stress behaviors take precedence naturally)

### Design Decisions

1. **Why use a state machine for stress agents?**
   - The state machine provides a clear way to model transitions between behavioral states
   - States naturally represent different "modes" of behavior
   - Easier to debug and understand than continuous blending of behaviors

2. **Why have different agent types?**
   - Simple agents demonstrate basic Braitenberg principles
   - Stress agents demonstrate more complex emergent behaviors
   - Allows comparison between different approaches

3. **Memory in agents**
   - The memory mechanism allows agents to maintain a representation of the environment
   - Memory decay simulates forgetting
   - This enables goal-directed behavior even when stimuli are not directly perceivable

## Future Improvements

Potential areas for extending the simulation:

1. **Learning mechanisms**: Implement simple learning rules to allow agents to adapt to the environment
2. **Evolutionary algorithms**: Evolve agent parameters to optimize survival
3. **More complex environments**: Add dynamic elements, hazards, or changing conditions
4. **Social interactions**: Allow agents to perceive and respond to each other
5. **Neural network controllers**: Replace direct mappings with simple neural networks

## Research Questions

The simulation environment can be used to explore questions such as:

1. How does stress-based behavior switching compare to explicit action selection?
2. What level of behavioral complexity can emerge from simple sensorimotor mappings?
3. How do different stress decay rates affect overall behavior patterns?
4. Can effective collective behaviors emerge without explicit coordination?
5. What is the minimum necessary cognitive architecture for different tasks? 