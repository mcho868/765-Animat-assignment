"""
Base Animat class with sensors, wheels, and sensorimotor links.
"""
import numpy as np
from config import settings
from core.environment import EntityType

class Animat:
    """
    Animat class implementing a simple agent with:
    - Two wheels (left and right)
    - Six sensors (food, water, trap on both left and right)
    - 18 sensorimotor links (3 per sensor, connecting to same-side wheel)
    - Two batteries (energy levels)
    """
    
    def _scale_genome_value(self, val, min_val, max_val):
        """Scales a genome integer (0-99) to the range [min_val, max_val]."""
        return (val / 99.0) * (max_val - min_val) + min_val

    def _scale_sigmoid_genome_value(self, val):
        """Scales a genome integer (0-99) to the range [-3.0, 3.0]."""
        return (val / 99.0) * 6.0 - 3.0

    def __init__(self, position, genome=None, direction=None):
        """Initialize an Animat with position and optional genome.
        
        Args:
            position: (x, y) tuple for starting position
            genome: Optional genome for sensorimotor links (will be randomized if None)
            direction: Optional initial direction (will be randomized if None)
        """
        from core.environment import Entity, EntityType
        
        # Basic entity properties
        self.position = np.array(position, dtype=float)
        self.radius = settings.ANIMAT_SIZE
        self.active = True
        self.type = EntityType.ANIMAT
        
        # Initialize direction (heading)
        if direction is None:
            angle = np.random.uniform(0, 2 * np.pi)
            self.direction = np.array([np.cos(angle), np.sin(angle)])
        else:
            self.direction = np.array(direction, dtype=float)
            self.direction /= np.linalg.norm(self.direction)  # Normalize
        
        # Energy levels (batteries)
        self.batteries = [settings.BATTERY_MAX, settings.BATTERY_MAX]
        
        # Wheels (store current speed values)
        self.wheel_speeds = [0.0, 0.0]  # [left, right]
        
        # Initialize genome (sensorimotor link parameters)
        if genome is None:
            self.initialize_random_genome()
        else:
            self.genome = genome.copy()
            
        # Extract and scale sigmoid thresholds from genome
        self.sigmoid_thresholds = [
            self._scale_sigmoid_genome_value(self.genome[-2]),  # Left wheel
            self._scale_sigmoid_genome_value(self.genome[-1])   # Right wheel
        ]
            
        # Parse genome into sensorimotor links
        self.parse_genome()
        
        self.speed_history = []  # Track speed at each timestep
        
    def initialize_random_genome(self):
        """Initialize a random genome for the animat.
        Only 9 links are genetically specified due to left-right symmetry.
        The genome encodes links for left-side sensors, and right-side links are mirrored.
        """
        # Genome format (per link): [weight, bias], final 2 genes are sigmoid thresholds for wheels
        genome_size = settings.GENOTYPE_SIZE
        self.genome = np.random.randint(0, 100, genome_size)

    def parse_genome(self):
        """Parse the genome into sensorimotor links with left-right symmetry enforcement.
        The genome encodes links for left-side sensors, and right-side links are mirrored.
        """
        self.links = []
        encoded_links = []
        for i in range(0, settings.NUM_LINKS * settings.LINK_PARAM_COUNT, settings.LINK_PARAM_COUNT):
            # Scale weight (0-99 to -1.0 to 1.0)
            weight = self._scale_genome_value(self.genome[i], -1.0, 1.0)
            # Scale bias (0-99 to -0.5 to 0.5)
            bias = self._scale_genome_value(self.genome[i + 1], -0.5, 0.5)
            link_params = {
                'weight': weight,
                'bias': bias
            }
            encoded_links.append(link_params)
        # Now create the full links with symmetry
        num_sensors = settings.NUM_SENSORS
        for sensor_idx in range(num_sensors):
            for link_offset in range(3):
                if sensor_idx % 2 == 0:  # Left sensor
                    encoded_idx = (sensor_idx // 2) * 3 + link_offset
                    self.links.append(encoded_links[encoded_idx].copy())
                else:  # Right sensor
                    left_sensor_idx = sensor_idx - 1
                    encoded_idx = (left_sensor_idx // 2) * 3 + link_offset
                    self.links.append(encoded_links[encoded_idx].copy())

    def get_sensor_to_wheel_mapping(self, sensor_index):
        """Determine which wheel a sensor connects to based on side.
        
        Args:
            sensor_index: Index of the sensor (0-7 for multi-animat)
            
        Returns:
            Index of the wheel (0=left, 1=right)
        """
        return sensor_index % 2
        
    def compute_sensor_to_wheel_output(self, sensor_value, link_index):
        """Compute the output of a sensorimotor link.
        Args:
            sensor_value: Current sensor reading (0-100)
            link_index: Index of the link to use
        Returns:
            Output value contribution to wheel
        """
        link = self.links[link_index]
        weight = link['weight']
        bias = link['bias']
        output = (sensor_value * weight) + bias
        return output
        
    def compute_wheel_speed(self, sensor_readings):
        """Compute wheel speeds based on sensor readings and sensorimotor links.
        
        Args:
            sensor_readings: Dict of sensor readings from environment
            
        Returns:
            List of wheel speeds [left_wheel, right_wheel]
        """
        # Build sensor_keys list based on NUM_SENSORS
        base_keys = ['food_left', 'food_right', 'water_left', 'water_right', 'trap_left', 'trap_right', 'other_left', 'other_right']
        sensor_keys = base_keys[:settings.NUM_SENSORS]
        sensor_values = [sensor_readings[k] for k in sensor_keys]
        
        # Reset wheel outputs
        wheel_outputs = [0.0, 0.0]  # [left, right]
        
        # For each sensor and its three corresponding links
        for sensor_idx, sensor_value in enumerate(sensor_values):
            wheel_idx = self.get_sensor_to_wheel_mapping(sensor_idx)
            base_link_idx = sensor_idx * 3
            for offset in range(3):
                link_idx = base_link_idx + offset
                wheel_outputs[wheel_idx] += self.compute_sensor_to_wheel_output(sensor_value, link_idx)
        
        # Apply sigmoid function with evolved threshold to each wheel
        left_wheel = self.sigmoid(wheel_outputs[0], self.sigmoid_thresholds[0])
        right_wheel = self.sigmoid(wheel_outputs[1], self.sigmoid_thresholds[1])
        
        # Scale to actual wheel speeds
        self.wheel_speeds = [
            left_wheel * settings.ANIMAT_MAX_SPEED,
            right_wheel * settings.ANIMAT_MAX_SPEED
        ]
        
        return self.wheel_speeds
        
    def sigmoid(self, x, threshold):
        """Compute a sigmoid function with the given threshold.
        
        Args:
            x: Input value
            threshold: Sigmoid threshold parameter
            
        Returns:
            Sigmoid activation (-1 to 1)
        """
        # The 'threshold' parameter is already scaled to a range like [-3.0, 3.0] by _scale_sigmoid_genome_value
        # Using it directly provides a reasonable slope for the sigmoid.
        return 2.0 / (1.0 + np.exp(-x * threshold)) - 1.0
        
    def update(self, dt, environment):
        """Update the animat's state for one timestep.
        
        Args:
            dt: Time delta in seconds
            environment: The environment the animat is in
        """
        if not self.active:
            return
            
        # Get sensor readings from environment
        sensor_readings = environment.get_sensor_readings(self)
        
        # Debug: Print sensor values and wheel speeds for first 3 animats and first 10 steps
        if hasattr(self, 'debug_id') and self.debug_id < 3:
            if not hasattr(self, 'debug_step'):
                self.debug_step = 0
            if self.debug_step < 10:
                print(f"Animat {self.debug_id} Step {self.debug_step}: Sensors: {sensor_readings}")
        
        # Compute wheel speeds
        self.compute_wheel_speed(sensor_readings)
        
        if hasattr(self, 'debug_id') and self.debug_id < 3 and hasattr(self, 'debug_step') and self.debug_step < 10:
            print(f"Animat {self.debug_id} Step {self.debug_step}: Wheel speeds: {self.wheel_speeds}")
            self.debug_step += 1
        
        # Move based on wheel speeds
        self.move(dt)
        # Track speed after moving
        self.speed_history.append(self.get_forward_speed())
        
        # Deplete batteries
        self.batteries[0] -= settings.BATTERY_DECAY_RATE * dt
        self.batteries[1] -= settings.BATTERY_DECAY_RATE * dt
        
        # Ensure batteries don't go below 0
        self.batteries[0] = max(0, self.batteries[0])
        self.batteries[1] = max(0, self.batteries[1])
        
        # Die if both batteries are empty
        if self.batteries[0] <= 0 and self.batteries[1] <= 0:
            self.active = False
            
    def move(self, dt):
        """Move the animat based on wheel speeds.
        
        Args:
            dt: Time delta in seconds
        """
        # Calculate forward velocity and rotation
        forward_speed = (self.wheel_speeds[0] + self.wheel_speeds[1]) / 2.0
        rotation_speed = (self.wheel_speeds[1] - self.wheel_speeds[0]) / (2.0 * self.radius)
        
        # Rotate direction vector
        angle = rotation_speed * dt
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        new_direction_x = cos_angle * self.direction[0] - sin_angle * self.direction[1]
        new_direction_y = sin_angle * self.direction[0] + cos_angle * self.direction[1]
        
        self.direction = np.array([new_direction_x, new_direction_y])
        self.direction /= np.linalg.norm(self.direction)  # Renormalize
        
        # Move forward in the direction of travel
        self.position += forward_speed * dt * self.direction
        
    def get_fitness(self):
        """Calculate the fitness of this animat based on battery levels.
        
        Returns:
            Fitness score (normalized according to paper, 0-1 range)
        """
        # F = (B1 + B2) / (2 * BATTERY_MAX) to align with paper's F = (B1 + B2)/400.0
        # where BATTERY_MAX from paper is 200.
        return (self.batteries[0] + self.batteries[1]) / (2.0 * settings.BATTERY_MAX)

    def get_forward_speed(self):
        """
        Calculates the current forward speed of the animat.
        This is the average of its two wheel speeds.
        The wheel speeds themselves are scaled by ANIMAT_MAX_SPEED.
        """
        return (self.wheel_speeds[0] + self.wheel_speeds[1]) / 2.0

    def get_average_speed(self):
        """Return the average forward speed over the animat's lifetime."""
        if not self.speed_history:
            return 0.0
        return sum(self.speed_history) / len(self.speed_history)