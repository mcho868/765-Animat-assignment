"""
Base Animat class with sensors, wheels, and sensorimotor links.
"""
import numpy as np
from config import settings
from core.environment import EntityType
from utils.logger import Logger


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
        
    def initialize_random_genome(self):
        """Initialize a random genome for the animat."""
        # Genome format (per link):
        # [offset, grad1, thresh1, grad2, thresh2, grad3, slope_mod, offset_mod, battery]
        # Final 2 genes are sigmoid thresholds for wheels
        
        genome_size = settings.GENOTYPE_SIZE
        self.genome = np.random.randint(0, 100, genome_size)
        
        # Set battery indicators (even=battery1, odd=battery2)
        for i in range(0, settings.NUM_LINKS * settings.LINK_PARAM_COUNT, settings.LINK_PARAM_COUNT):
            # Every 9th gene indicates battery
            self.genome[i + 8] = np.random.choice([0, 1])  # 0 for battery 1, 1 for battery 2
            
    def parse_genome(self):
        """Parse the genome into sensorimotor links according to paper's specification."""
        self.links = []
        self.logger = Logger()
        # Process each link's parameters
        for i in range(0, settings.NUM_LINKS * settings.LINK_PARAM_COUNT, settings.LINK_PARAM_COUNT):
            # Scale offset (0-99 to -100 to +100)
            offset_val = self._scale_genome_value(self.genome[i], -100.0, 100.0)

            # Scale gradients (0-99 to -pi/2 to +pi/2, then tan)
            angle1 = (self.genome[i + 1] / 99.0) * np.pi - (np.pi / 2.0)
            grad1_val = np.tan(angle1)

            # Scale threshold 1 (0-99 to -100 to +100)
            thresh1_val = self._scale_genome_value(self.genome[i + 2], -100.0, 100.0)

            angle2 = (self.genome[i + 3] / 99.0) * np.pi - (np.pi / 2.0)
            grad2_val = np.tan(angle2)

            # Scale raw thresh2 value first (0-99 to -100 to +100)
            thresh2_raw_scaled = self._scale_genome_value(self.genome[i + 4], -100.0, 100.0)
            # Enforce that the second threshold must follow the first
            thresh2_val = max(thresh1_val, thresh2_raw_scaled)

            angle3 = (self.genome[i + 5] / 99.0) * np.pi - (np.pi / 2.0)
            grad3_val = np.tan(angle3)

            # Slope and offset modulation degrees (paper does not explicitly scale these beyond being 0-99 derived)
            # Current code uses / 10.0. Retaining this specific scaling.
            slope_mod_val = self.genome[i + 6] / 10.0
            offset_mod_val = self.genome[i + 7] / 10.0
            
            battery_val = self.genome[i + 8] # Already 0 or 1, used directly

            link_params = {
                'offset': offset_val,
                'grad1': grad1_val,
                'thresh1': thresh1_val,
                'grad2': grad2_val,
                'thresh2': thresh2_val,
                'grad3': grad3_val,
                'slope_mod': slope_mod_val,
                'offset_mod': offset_mod_val,
                'battery': battery_val,
            }

            #Persent in agent logs
            self.logger.log_agent(link_param = link_params)
            # print(f"Link Parameters:")
            # print(f"  Offset: {link_params['offset']:.2f}")
            # print(f"  Gradients: {link_params['grad1']:.2f}, {link_params['grad2']:.2f}, {link_params['grad3']:.2f}")
            # print(f"  Thresholds: {link_params['thresh1']:.2f}, {link_params['thresh2']:.2f}")
            # print(f"  Modulations: slope={link_params['slope_mod']:.2f}, offset={link_params['offset_mod']:.2f}")
            # print(f"  Battery: {link_params['battery']}")
            self.links.append(link_params)
            
    def get_sensor_to_wheel_mapping(self, sensor_index):
        """Determine which wheel a sensor connects to based on side.
        
        Args:
            sensor_index: Index of the sensor (0-5)
            
        Returns:
            Index of the wheel (0=left, 1=right)
        """
        # Sensors 0, 2, 4 are on the left, connect to left wheel (0)
        # Sensors 1, 3, 5 are on the right, connect to right wheel (1)
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
        
        # Determine which battery influences this link
        battery_level = self.batteries[link['battery']]
        battery_factor = battery_level / settings.BATTERY_MAX  # 0-1 normalized
        
        # Apply battery modulation to parameters
        slope = link['grad1']
        if sensor_value > link['thresh1']:
            slope = link['grad2']
        if sensor_value > link['thresh2']:
            slope = link['grad3']
        
        # Apply battery modulation to slope and offset
        modulated_slope = slope + (link['slope_mod'] * battery_factor)
        modulated_offset = link['offset'] + (link['offset_mod'] * battery_factor)
        
        # Compute output
        output = modulated_offset + (modulated_slope * sensor_value)
        
        # Limit output to range [-1, 1]
        return max(-1.0, min(1.0, output))
        
    def compute_wheel_speed(self, sensor_readings):
        """Compute wheel speeds based on sensor readings and sensorimotor links.
        
        Args:
            sensor_readings: Dict of sensor readings from environment
            
        Returns:
            List of wheel speeds [left_wheel, right_wheel]
        """
        # Convert sensor readings dict to a list in the expected order
        sensor_values = [
            sensor_readings['food_left'],
            sensor_readings['food_right'],
            sensor_readings['water_left'],
            sensor_readings['water_right'],
            sensor_readings['trap_left'],
            sensor_readings['trap_right']
        ]
        
        # Reset wheel outputs
        wheel_outputs = [0.0, 0.0]  # [left, right]
        
        # For each sensor and its three corresponding links
        for sensor_idx, sensor_value in enumerate(sensor_values):
            # Determine which wheel this sensor connects to
            wheel_idx = self.get_sensor_to_wheel_mapping(sensor_idx)
            
            # Each sensor has 3 parallel links
            base_link_idx = sensor_idx * 3
            
            # Compute and accumulate the output from each link
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
        # Adjust threshold to be in a reasonable range
        scaled_threshold = threshold / 20.0
        # Sigmoid function that outputs in range [-1, 1]
        return 2.0 / (1.0 + np.exp(-x * scaled_threshold)) - 1.0
        
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
        
        # Compute wheel speeds
        self.compute_wheel_speed(sensor_readings)
        
        # Move based on wheel speeds
        self.move(dt)
        
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