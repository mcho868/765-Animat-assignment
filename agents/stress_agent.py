"""
More complex agent that implements stress-based behavior.
This agent responds to stress levels by changing behavior patterns.
"""
import random
import math
from enum import Enum
from agents.base_agent import Agent, Sensor

class BehaviorState(Enum):
    """Enumeration of possible behavior states for the stress agent."""
    EXPLORE = 0       # Low stress, explore environment
    SEEK_LIGHT = 1    # Medium stress, seek light
    SEEK_FOOD = 2     # High stress, seek food
    FREEZE = 3        # Very high stress, minimal movement

class StressAgent(Agent):
    """
    A more complex agent that changes behavior based on stress levels.
    Implements a simple state machine for decision making.
    """
    def __init__(self, entity_id, position, heading=None, radius=12):
        """
        Initialize a stress-based agent.
        
        Args:
            entity_id: Unique identifier for this agent
            position: Initial (x, y) position
            heading: Initial heading in degrees (random if None)
            radius: Agent's physical radius
        """
        # Set random heading if none provided
        if heading is None:
            heading = random.uniform(0, 360)
            
        super().__init__(entity_id, position, heading, radius)
        
        # Additional sensors for food detection
        self.add_sensor(Sensor("front_food", 
                         position_offset=(self.radius * 1.0, 0), 
                         angle_offset=0,
                         range=60))
        
        self.current_state = BehaviorState.EXPLORE
        self.state_timer = 0
        self.min_state_time = 2.0  # Minimum time to stay in a state
        
        # Memory of light/food sources
        self.light_memory = None
        self.food_memory = None
        self.memory_decay = 0.1  # How quickly memories fade
                
    def determine_state(self):
        """Determine behavior state based on current stress level."""
        if self.stress < 30:
            return BehaviorState.EXPLORE
        elif self.stress < 60:
            return BehaviorState.SEEK_LIGHT
        elif self.stress < 90:
            return BehaviorState.SEEK_FOOD
        else:
            return BehaviorState.FREEZE
            
    def update_memory(self, environment):
        """Update agent's memory of light and food sources."""
        # Find entities in sensor range
        nearby_entities = environment.get_entities_in_range(self.position, 100)
        
        # Look for light sources
        lights = [(entity, dist) for entity, dist in nearby_entities 
                  if entity.type == environment.EntityType.LIGHT_SOURCE and entity.active]
        
        # Remember closest light
        if lights:
            closest_light = min(lights, key=lambda x: x[1])
            self.light_memory = closest_light[0].position
        elif self.light_memory:
            # Decay memory over time
            if random.random() < self.memory_decay:
                self.light_memory = None
        
        # Look for food sources
        foods = [(entity, dist) for entity, dist in nearby_entities 
                 if entity.type == environment.EntityType.FOOD_SOURCE and entity.active]
        
        # Remember closest food
        if foods:
            closest_food = min(foods, key=lambda x: x[1])
            self.food_memory = closest_food[0].position
        elif self.food_memory:
            # Decay memory over time
            if random.random() < self.memory_decay:
                self.food_memory = None
                
    def decide(self, sensor_readings):
        """
        Determine motor activations based on sensor readings and current state.
        """
        left_light = sensor_readings.get("left_light", 0)
        right_light = sensor_readings.get("right_light", 0)
        front_food = sensor_readings.get("front_food", 0)
        
        # Check if state should change
        new_state = self.determine_state()
        if self.state_timer <= 0 and new_state != self.current_state:
            self.current_state = new_state
            self.state_timer = self.min_state_time
        else:
            self.state_timer -= 0.1
            
        # Default motor activations
        left_motor = 0.5
        right_motor = 0.5
        
        # Apply different behaviors based on state
        if self.current_state == BehaviorState.EXPLORE:
            # Exploration behavior: random movement with occasional turns
            if random.random() < 0.05:
                # Random turn
                turn_direction = random.choice([-1, 1])
                left_motor = 0.7 + 0.3 * turn_direction
                right_motor = 0.7 - 0.3 * turn_direction
            else:
                # Forward movement with slight randomness
                left_motor = 0.7 + random.uniform(-0.1, 0.1)
                right_motor = 0.7 + random.uniform(-0.1, 0.1)
                
        elif self.current_state == BehaviorState.SEEK_LIGHT:
            # Light seeking behavior (cross connections)
            if max(left_light, right_light) > 20:
                # We can see light, use it for direct navigation
                left_motor = right_light / 100.0
                right_motor = left_light / 100.0
            elif self.light_memory:
                # Use memory to move toward last known light
                dx = self.light_memory[0] - self.position[0]
                dy = self.light_memory[1] - self.position[1]
                angle_to_light = math.degrees(math.atan2(dy, dx))
                
                # Calculate difference between current heading and angle to light
                angle_diff = (angle_to_light - self.heading) % 360
                if angle_diff > 180:
                    angle_diff -= 360
                    
                # Turn toward light
                if abs(angle_diff) < 10:
                    # Almost aligned, go straight
                    left_motor = 0.8
                    right_motor = 0.8
                elif angle_diff > 0:
                    # Turn left
                    left_motor = 0.2
                    right_motor = 0.8
                else:
                    # Turn right
                    left_motor = 0.8
                    right_motor = 0.2
            else:
                # No light detected and no memory, do exploration
                left_motor = 0.6 + random.uniform(-0.2, 0.2)
                right_motor = 0.6 + random.uniform(-0.2, 0.2)
                
        elif self.current_state == BehaviorState.SEEK_FOOD:
            # Food seeking behavior
            if front_food > 0:
                # Food detected ahead, move forward
                left_motor = 1.0
                right_motor = 1.0
            elif self.food_memory:
                # Use memory to move toward last known food
                dx = self.food_memory[0] - self.position[0]
                dy = self.food_memory[1] - self.position[1]
                angle_to_food = math.degrees(math.atan2(dy, dx))
                
                # Calculate difference between current heading and angle to food
                angle_diff = (angle_to_food - self.heading) % 360
                if angle_diff > 180:
                    angle_diff -= 360
                    
                # Turn toward food
                if abs(angle_diff) < 10:
                    # Almost aligned, go straight
                    left_motor = 1.0
                    right_motor = 1.0
                elif angle_diff > 0:
                    # Turn left
                    left_motor = 0.3
                    right_motor = 1.0
                else:
                    # Turn right
                    left_motor = 1.0
                    right_motor = 0.3
            else:
                # No food detected and no memory, do random search
                left_motor = 0.8 + random.uniform(-0.2, 0.2)
                right_motor = 0.8 + random.uniform(-0.2, 0.2)
                
        elif self.current_state == BehaviorState.FREEZE:
            # Freeze behavior: minimal movement
            left_motor = random.uniform(0, 0.1)
            right_motor = random.uniform(0, 0.1)
            
        # Ensure motors stay within bounds
        left_motor = max(0.0, min(1.0, left_motor))
        right_motor = max(0.0, min(1.0, right_motor))
            
        return {
            "left_motor": left_motor,
            "right_motor": right_motor
        }
        
    def update(self, dt, environment=None):
        """Update agent state."""
        if environment:
            # Update memory of environment
            self.update_memory(environment)
            
        # Call parent update method
        super().update(dt, environment)
        
    def update_stress(self, dt, environment):
        """
        Update stress based on environment conditions.
        """
        # Base stress update from parent class
        super().update_stress(dt, environment)
        
        # Additional stress factors
        
        # Stress from low energy
        if self.energy < 30:
            energy_stress = (30 - self.energy) / 10
            self.stress += energy_stress * dt
            
        # Stress from obstacles nearby
        nearby_entities = environment.get_entities_in_range(self.position, 50)
        obstacles = [(entity, dist) for entity, dist in nearby_entities 
                    if entity.type == environment.EntityType.OBSTACLE]
        
        if obstacles:
            # Stress increases with closer obstacles
            closest_obstacle = min(obstacles, key=lambda x: x[1])
            obstacle_stress = 10 * (1 - closest_obstacle[1] / 50)
            self.stress += obstacle_stress * dt
            
        # Stress decreases when food is consumed
        # (already handled in base Agent.move method)
        
        # Stress decays naturally over time
        self.stress = max(0, self.stress - 1 * dt) 