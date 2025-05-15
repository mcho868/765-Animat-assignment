"""
Base agent class for Braitenberg-style agents.
Implements sensors and motors as an abstract class.
"""
import random
import math
from abc import ABC, abstractmethod
from core.environment import Entity, EntityType
from utils.math_utils import normalize_vector, angle_to_vector, vector_to_angle, rotate_vector

class Sensor:
    """
    Sensor class representing an agent's perception of the environment.
    Each sensor has a position relative to the agent and a detection angle.
    """
    def __init__(self, name, position_offset=(0, 0), angle_offset=0, range=100, sensitivity=1.0):
        self.name = name
        self.position_offset = position_offset  # Relative to agent center
        self.angle_offset = angle_offset  # In degrees
        self.range = range
        self.sensitivity = sensitivity
        self.last_reading = 0

    def get_position(self, agent):
        """Get the absolute position of this sensor given agent position and heading."""
        offset_rotated = rotate_vector(self.position_offset, agent.heading)
        return (agent.position[0] + offset_rotated[0], 
                agent.position[1] + offset_rotated[1])
                
    def get_direction(self, agent):
        """Get the absolute direction this sensor is pointing."""
        sensor_angle = agent.heading + self.angle_offset
        return angle_to_vector(sensor_angle)
        
    def sense(self, environment, agent):
        """
        Sense the environment and return a reading.
        This base implementation senses light intensity.
        """
        sensor_pos = self.get_position(agent)
        reading = environment.get_light_intensity_at(sensor_pos) * self.sensitivity
        self.last_reading = reading
        return reading

class Motor:
    """
    Motor class representing an actuator that can affect agent movement.
    """
    def __init__(self, name, power=1.0, position=(0, 0)):
        self.name = name
        self.power = power  # Strength of the motor
        self.position = position  # Position relative to agent center (for torque calculation)
        self.activation = 0  # Current activation level

    def set_activation(self, value):
        """Set the activation level of this motor."""
        self.activation = max(0, min(1, value))  # Clamp between 0 and 1
        
    def get_force(self):
        """Get the force vector produced by this motor."""
        return self.power * self.activation
        
class Agent(Entity, ABC):
    """
    Abstract base class for all agents in the simulation.
    Implements the basic sensor-motor architecture for Braitenberg vehicles.
    """
    def __init__(self, entity_id, position, heading=0, radius=10):
        super().__init__(entity_id, EntityType.AGENT, position, radius)
        self.heading = heading  # Degrees, 0 is right, 90 is up
        self.direction = angle_to_vector(heading)
        self.velocity = (0, 0)
        self.speed = 0
        self.sensors = {}
        self.motors = {}
        self.stress = 0
        self.energy = 100
        self.default_sensors()
        self.default_motors()
        
    def default_sensors(self):
        """Create the default set of sensors."""
        # Left and right light sensors
        self.add_sensor(Sensor("left_light", 
                         position_offset=(self.radius * 0.7, self.radius * 0.7), 
                         angle_offset=-45))
        self.add_sensor(Sensor("right_light", 
                         position_offset=(self.radius * 0.7, -self.radius * 0.7), 
                         angle_offset=45))
        
    def default_motors(self):
        """Create the default set of motors."""
        # Left and right wheel motors
        self.add_motor(Motor("left_motor", position=(0, self.radius)))
        self.add_motor(Motor("right_motor", position=(0, -self.radius)))
        
    def add_sensor(self, sensor):
        """Add a sensor to this agent."""
        self.sensors[sensor.name] = sensor
        
    def add_motor(self, motor):
        """Add a motor to this agent."""
        self.motors[motor.name] = motor
        
    def sense(self, environment):
        """Get readings from all sensors."""
        readings = {}
        for name, sensor in self.sensors.items():
            readings[name] = sensor.sense(environment, self)
        return readings
    
    @abstractmethod
    def decide(self, sensor_readings):
        """
        Given sensor readings, decide on motor activations.
        Must be implemented by subclasses.
        """
        pass
        
    def update_stress(self, dt, environment):
        """Update the agent's stress level based on the environment."""
        # Default implementation: increase stress in darkness, decrease in light
        light_intensity = environment.get_light_intensity_at(self.position)
        
        # Stress increases in darkness, decreases in light
        if light_intensity < 10:
            self.stress += 5 * dt
        else:
            self.stress -= 2 * dt
            
        # Cap stress
        self.stress = max(0, min(100, self.stress))
    
    def move(self, dt, environment):
        """Update position based on current velocity."""
        # Calculate new position
        new_x = self.position[0] + self.velocity[0] * dt
        new_y = self.position[1] + self.velocity[1] * dt
        new_position = (new_x, new_y)
        
        # Check for collisions
        collision, entity = environment.check_collision(new_position, self.radius, self)
        
        if collision:
            # Handle collision - simple bounce
            if entity is None:  # Wall collision
                # Determine which wall was hit and reverse appropriate velocity component
                if new_x <= self.radius or new_x >= environment.width - self.radius:
                    self.velocity = (-self.velocity[0] * 0.8, self.velocity[1])
                if new_y <= self.radius or new_y >= environment.height - self.radius:
                    self.velocity = (self.velocity[0], -self.velocity[1] * 0.8)
                
                # Update heading after bounce
                if self.velocity[0] != 0 or self.velocity[1] != 0:
                    self.heading = vector_to_angle(self.velocity)
                    self.direction = normalize_vector(self.velocity)
                
                # Add some stress from hitting a wall
                self.stress += 10
            else:
                # Entity collision - more complex response
                # Calculate bounce direction
                collision_vector = (self.position[0] - entity.position[0],
                                   self.position[1] - entity.position[1])
                bounce_dir = normalize_vector(collision_vector)
                
                # Set velocity to bounce direction with reduced magnitude
                speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2) * 0.5
                self.velocity = (bounce_dir[0] * speed, bounce_dir[1] * speed)
                
                # Update heading after bounce
                self.heading = vector_to_angle(self.velocity)
                self.direction = normalize_vector(self.velocity)
                
                # Add some stress from collision
                self.stress += 5
                
            # Use current position since we couldn't move
            new_position = self.position
        else:
            # No collision, update position
            self.position = new_position
            
            # Try to consume food at this position
            food_consumed = environment.consume_food_at(self.position, self.radius, 10)
            if food_consumed > 0:
                self.energy += food_consumed
                self.stress = max(0, self.stress - food_consumed * 0.1)
                
        return new_position
                
    def update(self, dt, environment=None):
        """Update agent state."""
        if environment is None:
            return
            
        # Update stress based on environment
        self.update_stress(dt, environment)
        
        # Get sensor readings
        readings = self.sense(environment)
        
        # Make decision based on readings
        motor_commands = self.decide(readings)
        
        # Apply motor commands
        for motor_name, activation in motor_commands.items():
            if motor_name in self.motors:
                self.motors[motor_name].set_activation(activation)
        
        # Calculate movement based on motor activations
        left_force = self.motors["left_motor"].get_force()
        right_force = self.motors["right_motor"].get_force()
        
        # Calculate forward force and turning force
        forward_force = (left_force + right_force) / 2
        turn_force = (right_force - left_force) / 2
        
        # Update heading based on turn force
        self.heading += turn_force * 5  # Adjust this multiplier to control turn rate
        self.heading %= 360  # Keep heading in [0, 360)
        
        # Calculate direction vector from heading
        self.direction = angle_to_vector(self.heading)
        
        # Calculate velocity based on forward force and direction
        self.speed = forward_force * 50  # Adjust this multiplier to control speed
        self.velocity = (self.direction[0] * self.speed, self.direction[1] * self.speed)
        
        # Move the agent
        self.move(dt, environment)
        
        # Consume energy proportional to motor activation
        energy_cost = (left_force + right_force) * dt * 2
        self.energy -= energy_cost
        
        # Die if energy depleted
        if self.energy <= 0:
            self.active = False 