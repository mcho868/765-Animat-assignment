"""
Simple Braitenberg vehicle agent implementation.
Demonstrates basic light-seeking or light-avoiding behavior.
"""
import random
import math
from agents.base_agent import Agent, Sensor

class SimpleAgent(Agent):
    """
    A simple Braitenberg vehicle-style agent that can exhibit various tropisms:
    - Type 1: Random movement
    - Type 2: Light-seeking (Positive phototropism)
    - Type 3: Light-avoiding (Negative phototropism)
    """
    def __init__(self, entity_id, position, behavior_type=2, heading=None, radius=10):
        """
        Initialize a simple agent with the specified behavior type.
        
        Args:
            entity_id: Unique identifier for this agent
            position: Initial (x, y) position
            behavior_type: 1=random, 2=light-seeking, 3=light-avoiding
            heading: Initial heading in degrees (random if None)
            radius: Agent's physical radius
        """
        # Set random heading if none provided
        if heading is None:
            heading = random.uniform(0, 360)
            
        super().__init__(entity_id, position, heading, radius)
        
        self.behavior_type = behavior_type
        self.random_direction_timer = 0
        
    def decide(self, sensor_readings):
        """
        Determine motor activations based on sensor readings and behavior type.
        """
        left_light = sensor_readings.get("left_light", 0)
        right_light = sensor_readings.get("right_light", 0)
        
        # Normalize readings to [0, 1] range
        max_reading = max(1, max(left_light, right_light))
        left_light_norm = left_light / max_reading
        right_light_norm = right_light / max_reading
        
        # Default motor activations
        left_motor = 0.5
        right_motor = 0.5
        
        # Apply different behaviors based on type
        if self.behavior_type == 1:
            # Type 1: Random movement
            if self.random_direction_timer <= 0:
                # Choose new random direction
                left_motor = random.uniform(0.2, 1.0)
                right_motor = random.uniform(0.2, 1.0)
                self.random_direction_timer = random.uniform(1.0, 3.0)
            else:
                self.random_direction_timer -= 0.1
                
        elif self.behavior_type == 2:
            # Type 2: Light-seeking (cross connections)
            # High light on right sensor activates left motor, and vice versa
            left_motor = right_light_norm
            right_motor = left_light_norm
            
        elif self.behavior_type == 3:
            # Type 3: Light-avoiding (direct connections)
            # High light on right sensor activates right motor, and vice versa
            left_motor = left_light_norm
            right_motor = right_light_norm
            
        # Add some random noise to prevent getting stuck
        left_motor = max(0.1, min(1.0, left_motor + random.uniform(-0.1, 0.1)))
        right_motor = max(0.1, min(1.0, right_motor + random.uniform(-0.1, 0.1)))
            
        return {
            "left_motor": left_motor,
            "right_motor": right_motor
        }
        
    def update_stress(self, dt, environment):
        """
        Update stress based on environment and behavior type.
        """
        # Get base stress update from parent class
        super().update_stress(dt, environment)
        
        # For light-seeking agents, increase stress when in darkness
        if self.behavior_type == 2:
            light_intensity = environment.get_light_intensity_at(self.position)
            if light_intensity < 20:
                self.stress += 10 * dt
                
        # For light-avoiding agents, increase stress when in bright light
        elif self.behavior_type == 3:
            light_intensity = environment.get_light_intensity_at(self.position)
            if light_intensity > 80:
                self.stress += 10 * dt 