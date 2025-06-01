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
    - 18 sensorimotor links (3 per sensor)
    - Two batteries (energy levels)
    
    Key paper specifications:
    - Links 0-8 connect to left wheel, links 9-17 connect to right wheel
    - Each link outputs [-1, 1], summ   ed outputs go through sigmoid
    - Final wheel speeds are [-10, 10], where +10,+10 = 2.8 units/timestep
    - Batteries start at 200, decay by 1 per timestep
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
        
        # Initialize survival time tracking
        self.survival_time = 0.0
        
        # Initialize direction (heading)
        if direction is None:
            angle = np.random.uniform(0, 2 * np.pi)
            self.direction = np.array([np.cos(angle), np.sin(angle)])
        else:
            self.direction = np.array(direction, dtype=float)
            self.direction /= np.linalg.norm(self.direction)  # Normalize
        
        # Energy levels (batteries) - paper specifies initial level 200
        self.batteries = [settings.BATTERY_MAX, settings.BATTERY_MAX]
        
        # Wheels (store current speed values in [-10, 10] range)
        self.wheel_speeds = [0.0, 0.0]  # [left, right]
        
        # Initialize genome (sensorimotor link parameters)
        if genome is None:
            self.initialize_random_genome()
        elif isinstance(genome, str) and genome == "seth":
            self.initialize_seth_genome()
        else:
            self.genome = genome.copy()
            
        # Extract and scale sigmoid thresholds from genome (positions 81-82)
        self.sigmoid_thresholds = [
            self._scale_sigmoid_genome_value(self.genome[-2]),  # Left wheel
            self._scale_sigmoid_genome_value(self.genome[-1])   # Right wheel
        ]
            
        # Parse genome into sensorimotor links
        self.parse_genome()
        
    def initialize_random_genome(self):
        """Initialize a random genome for the animat.
        
        Only 9 links are genetically specified due to left-right symmetry.
        The genome encodes links for left-side sensors, and right-side links are mirrored.
        """
        # Genome format (per link):
        # [offset, grad1, thresh1, grad2, thresh2, grad3, slope_mod, offset_mod, battery]
        # Final 2 genes are sigmoid thresholds for wheels
        
        genome_size = settings.GENOTYPE_SIZE  # Should be 83
        self.genome = np.random.randint(0, 100, genome_size)
        
        # Ensure thresh2 >= thresh1 for each of the 9 encoded links
        for i in range(0, settings.NUM_LINKS * settings.LINK_PARAM_COUNT, settings.LINK_PARAM_COUNT):
            thresh1_pos = i + 2  # Position of thresh1
            thresh2_pos = i + 4  # Position of thresh2
            
            # If thresh2 < thresh1, adjust thresh2 to be at least thresh1
            if self.genome[thresh2_pos] < self.genome[thresh1_pos]:
                # Set thresh2 to be thresh1 + a random offset (0 to remaining range)
                remaining_range = 99 - self.genome[thresh1_pos]
                if remaining_range > 0:
                    self.genome[thresh2_pos] = self.genome[thresh1_pos] + np.random.randint(1, remaining_range + 1)
                else:
                    self.genome[thresh1_pos] = max(0, self.genome[thresh1_pos] - np.random.randint(1, 10))
            
            # # Set battery indicators for the 9 encoded links (0=battery1, 1=battery2)
            # self.genome[i + 8] = np.random.choice([0, 1])  # 0 for battery 1, 1 for battery 2 # Just check if even or odd
            
    def initialize_seth_genome(self):
        """Initialize a genome based on Seth's specific link configurations from the paper.
        
        Creates the 9 links shown in the paper figures with their specific parameters:
        - Link 1 (bat 1): Food left sensor, specific piecewise linear function
        - Link 2 (bat 2): Food left sensor, declining linear function  
        - Link 3 (bat 2): Food left sensor, declining linear function
        - Link 4 (bat 2): Water left sensor, piecewise function
        - Link 5 (bat 2): Water left sensor, declining function
        - Link 6 (bat 1): Water left sensor, rising then flat function
        - Link 7 (bat 1): Trap left sensor, V-shaped function
        - Link 8 (bat 2): Trap left sensor, step function
        - Link 9 (bat 1): Trap left sensor, rising sigmoid-like function
        """
        genome_size = settings.GENOTYPE_SIZE  # Should be 83
        self.genome = np.zeros(genome_size, dtype=int)
        
        # Link 1 (bat 1) - Food left: Piecewise linear rising then declining
        self.genome[0] = 30   # offset -> scales to ~-40
        self.genome[1] = 75   # grad1 -> positive slope
        self.genome[2] = 80   # thresh1 -> ~60
        self.genome[3] = 25   # grad2 -> negative slope  
        self.genome[4] = 90   # thresh2 -> ~80
        self.genome[5] = 50   # grad3 -> ~0 slope
        self.genome[6] = 20   # slope_mod -> 0.2
        self.genome[7] = 70   # offset_mod -> positive
        self.genome[8] = 0    # battery 1
        
        # Link 2 (bat 2) - Food left: Declining linear function
        self.genome[9] = 95   # offset -> high positive (~90)
        self.genome[10] = 15  # grad1 -> negative slope
        self.genome[11] = 50  # thresh1 -> middle
        self.genome[12] = 15  # grad2 -> same negative slope
        self.genome[13] = 80  # thresh2 -> high
        self.genome[14] = 15  # grad3 -> same negative slope
        self.genome[15] = 10  # slope_mod -> low
        self.genome[16] = 30  # offset_mod -> slightly negative
        self.genome[17] = 1   # battery 2
        
        # Link 3 (bat 2) - Food left: Declining linear function (similar to link 2)
        self.genome[18] = 85  # offset -> positive
        self.genome[19] = 20  # grad1 -> negative slope
        self.genome[20] = 40  # thresh1 -> lower
        self.genome[21] = 20  # grad2 -> negative slope
        self.genome[22] = 70  # thresh2 -> high
        self.genome[23] = 20  # grad3 -> negative slope
        self.genome[24] = 15  # slope_mod -> low
        self.genome[25] = 25  # offset_mod -> negative
        self.genome[26] = 1   # battery 2
        
        # Link 4 (bat 2) - Water left: Piecewise function starting high, dropping, then rising
        self.genome[27] = 75  # offset -> positive
        self.genome[28] = 15  # grad1 -> negative slope initially
        self.genome[29] = 30  # thresh1 -> early threshold
        self.genome[30] = 5   # grad2 -> very negative slope
        self.genome[31] = 75  # thresh2 -> later threshold
        self.genome[32] = 85  # grad3 -> positive slope
        self.genome[33] = 25  # slope_mod -> moderate
        self.genome[34] = 20  # offset_mod -> negative
        self.genome[35] = 1   # battery 2
        
        # Link 5 (bat 2) - Water left: Simple declining function
        self.genome[36] = 50  # offset -> middle
        self.genome[37] = 25  # grad1 -> negative slope
        self.genome[38] = 40  # thresh1 -> middle
        self.genome[39] = 15  # grad2 -> more negative
        self.genome[40] = 80  # thresh2 -> high
        self.genome[41] = 5   # grad3 -> very negative
        self.genome[42] = 20  # slope_mod -> low
        self.genome[43] = 30  # offset_mod -> slightly negative
        self.genome[44] = 1   # battery 2
        
        # Link 6 (bat 1) - Water left: Rising then flat function
        self.genome[45] = 5   # offset -> low (~-90)
        self.genome[46] = 85  # grad1 -> positive slope
        self.genome[47] = 60  # thresh1 -> middle-high
        self.genome[48] = 50  # grad2 -> flat/small slope
        self.genome[49] = 80  # thresh2 -> high
        self.genome[50] = 50  # grad3 -> flat
        self.genome[51] = 80  # slope_mod -> high modulation
        self.genome[52] = 70  # offset_mod -> positive
        self.genome[53] = 0   # battery 1
        
        # Link 7 (bat 1) - Trap left: V-shaped function
        self.genome[54] = 85  # offset -> high positive
        self.genome[55] = 15  # grad1 -> negative slope
        self.genome[56] = 45  # thresh1 -> middle
        self.genome[57] = 75  # grad2 -> positive slope
        self.genome[58] = 70  # thresh2 -> high
        self.genome[59] = 85  # grad3 -> positive slope
        self.genome[60] = 60  # slope_mod -> high
        self.genome[61] = 50  # offset_mod -> neutral
        self.genome[62] = 0   # battery 1
        
        # Link 8 (bat 2) - Trap left: Step function dropping at ~60
        self.genome[63] = 50  # offset -> middle
        self.genome[64] = 50  # grad1 -> flat initially
        self.genome[65] = 80  # thresh1 -> high (around 60 scaled)
        self.genome[66] = 5   # grad2 -> very negative (step down)
        self.genome[67] = 85  # thresh2 -> very high
        self.genome[68] = 15  # grad3 -> negative
        self.genome[69] = 10  # slope_mod -> low
        self.genome[70] = 40  # offset_mod -> slightly negative
        self.genome[71] = 1   # battery 2
        
        # Link 9 (bat 1) - Trap left: Rising sigmoid-like function
        self.genome[72] = 5   # offset -> low start (~-90)
        self.genome[73] = 50  # grad1 -> flat initially
        self.genome[74] = 70  # thresh1 -> later
        self.genome[75] = 85  # grad2 -> steep rise
        self.genome[76] = 85  # thresh2 -> high
        self.genome[77] = 95  # grad3 -> very steep
        self.genome[78] = 90  # slope_mod -> high modulation
        self.genome[79] = 80  # offset_mod -> positive
        self.genome[80] = 0   # battery 1
        
        # Sigmoid thresholds for wheels (final 2 genes)
        self.genome[81] = 50  # Left wheel sigmoid threshold -> scales to 0.0
        self.genome[82] = 50  # Right wheel sigmoid threshold -> scales to 0.0
        
    def parse_genome(self):
        """Parse the genome into sensorimotor links with left-right symmetry enforcement.
        
        The genome encodes 9 links which are then mirrored to create 18 total links.
        Left sensors use encoded links directly, right sensors mirror them.
        """
        self.links = []
        
        # First, parse the 9 encoded links from the genome (positions 0-80)
        encoded_links = []
        for i in range(0, settings.NUM_LINKS * settings.LINK_PARAM_COUNT, settings.LINK_PARAM_COUNT):
            # Scale offset (0-99 to -100 to +100)
            offset_val = self._scale_genome_value(self.genome[i], -100.0, 100.0)

            # Scale gradients (0-99 to -pi/2 to +pi/2, then tan)
            angle1 = (self.genome[i + 1] / 99.0) * np.pi - (np.pi / 2.0)
            angle1 = np.clip(angle1, -np.pi/2.0 * 0.99, np.pi/2.0 * 0.99)
            grad1_val = np.tan(angle1)

            # Scale threshold 1 (0-99 to -100 to +100)
            thresh1_val = self._scale_genome_value(self.genome[i + 2], -100.0, 100.0)

            angle2 = (self.genome[i + 3] / 99.0) * np.pi - (np.pi / 2.0)
            angle2 = np.clip(angle2, -np.pi/2.0 * 0.99, np.pi/2.0 * 0.99)
            grad2_val = np.tan(angle2)

            # Scale raw thresh2 value first (0-99 to -100 to +100)
            thresh2_raw_scaled = self._scale_genome_value(self.genome[i + 4], -100.0, 100.0)
            # Enforce that the second threshold must follow the first
            thresh2_val = max(thresh1_val, thresh2_raw_scaled)

            angle3 = (self.genome[i + 5] / 99.0) * np.pi - (np.pi / 2.0)
            angle3 = np.clip(angle3, -np.pi/2.0 * 0.99, np.pi/2.0 * 0.99)
            grad3_val = np.tan(angle3)

            # Slope and offset modulation according to paper specifications
            # Slope modulation S ∈ (0 : 1)
            slope_mod_val = self.genome[i + 6] / 99.0  # 0-99 → 0.0 to 1.0
            # Offset modulation O ∈ (-1 : 1)  
            offset_mod_val = self._scale_genome_value(self.genome[i + 7], -1.0, 1.0)
            
            battery_val = self.genome[i + 8]%2  # Already 0 or 1, used directly

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
            encoded_links.append(link_params)
        
        # Now create the full 18 links with symmetry
        # Sensor mapping: 0=food_left, 1=food_right, 2=water_left, 3=water_right, 4=trap_left, 5=trap_right
        # Each sensor has 3 parallel links, so we have 18 total links
        
        # Create links for all 6 sensors (3 links per sensor)
        for sensor_idx in range(6):
            for link_offset in range(3):
                if sensor_idx % 2 == 0:  # Left sensor (0, 2, 4)
                    # Use encoded link directly
                    encoded_idx = (sensor_idx // 2) * 3 + link_offset
                    self.links.append(encoded_links[encoded_idx].copy())
                else:  # Right sensor (1, 3, 5)
                    # Mirror the corresponding left sensor link
                    left_sensor_idx = sensor_idx - 1
                    encoded_idx = (left_sensor_idx // 2) * 3 + link_offset
                    self.links.append(encoded_links[encoded_idx].copy())
                    
    def compute_sensor_to_wheel_output(self, sensor_value, link_index):
        """Compute the output of a sensorimotor link.
        
        Args:
            sensor_value: Current sensor reading (0-100)
            link_index: Index of the link to use
            
        Returns:
            Output value limited to [-1, 1] range as per paper
        """
        link = self.links[link_index]
        
        # Get battery modulation
        battery_level = self.batteries[link['battery']]
        
        # Apply battery modulation (as in plotting script)
        offset_modulation = (battery_level / 2.0) * link['offset_mod']
        slope_modulation_factor = ((battery_level - 100.0) / 100.0) * link['slope_mod']
        
        # Compute base piecewise linear function (maintaining continuity)
        if sensor_value <= link['thresh1']:
            output = link['offset'] + link['grad1'] * sensor_value
        elif sensor_value <= link['thresh2']:
            y1 = link['offset'] + link['grad1'] * link['thresh1']
            output = y1 + link['grad2'] * (sensor_value - link['thresh1'])
        else:
            y1 = link['offset'] + link['grad1'] * link['thresh1']
            y2 = y1 + link['grad2'] * (link['thresh2'] - link['thresh1'])
            output = y2 + link['grad3'] * (sensor_value - link['thresh2'])
        
        # Apply battery modulation
        output += offset_modulation
        output += output * slope_modulation_factor
        
        # Paper specifies link outputs are "ranged from -1 to 1"
        return max(-1.0, min(1.0, output/100))
            
    def compute_wheel_speed(self, sensor_readings):
        """Compute wheel speeds based on sensor readings and sensorimotor links.
        
        CORRECTED: Links 0-8 connect to left wheel, links 9-17 connect to right wheel
        as specified in the paper.
        
        Args:
            sensor_readings: Dict of sensor readings from environment
            
        Returns:
            List of wheel speeds [left_wheel, right_wheel] in [-10, 10] range
        """
        # Convert sensor readings dict to a list in the expected order
        sensor_values = [
            sensor_readings['food_left'],     # Sensor 0
            sensor_readings['food_right'],    # Sensor 1
            sensor_readings['water_left'],    # Sensor 2
            sensor_readings['water_right'],   # Sensor 3
            sensor_readings['trap_left'],     # Sensor 4
            sensor_readings['trap_right']     # Sensor 5
        ]
        
        # Accumulate outputs for each wheel
        left_wheel_sum = 0.0   # Sum of links 0-8
        right_wheel_sum = 0.0  # Sum of links 9-17
        
        # Process all 18 links (3 per sensor × 6 sensors)
        for link_idx in range(18):
            # Determine which sensor this link belongs to
            sensor_idx = link_idx // 3  # Links 0-2→sensor 0, 3-5→sensor 1, etc.
            sensor_value = sensor_values[sensor_idx]
            
            # Compute link output
            output = self.compute_sensor_to_wheel_output(sensor_value, link_idx)
            
            # CORRECTED: Route to wheels based on link index, not sensor side
            if link_idx < 9:  # Links 0-8 go to left wheel
                left_wheel_sum += output
            else:  # Links 9-17 go to right wheel  
                right_wheel_sum += output
        
        # Apply sigmoid function with evolved thresholds and scale to [-10, 10]
        left_wheel = self.sigmoid_to_wheel_speed(left_wheel_sum, self.sigmoid_thresholds[0])
        right_wheel = self.sigmoid_to_wheel_speed(right_wheel_sum, self.sigmoid_thresholds[1])
        
        # Store final wheel speeds (already in correct range)
        self.wheel_speeds = [left_wheel, right_wheel]
        
        return self.wheel_speeds
        
    def sigmoid_to_wheel_speed(self, x, threshold):
        """Convert summed link outputs to wheel speed via sigmoid.
        
        Paper specifies: "passed through a sigmoid function, and then scaled from -10 to 10"
        
        Args:
            x: Summed link outputs (typically in range [-9, 9])
            threshold: Evolved sigmoid threshold ([-3.0, 3.0])
            
        Returns:
            Wheel speed in [-10, 10] range
        """
        try:
            # Standard sigmoid [0, 1]
            sigmoid_01 = 1.0 / (1.0 + np.exp(-(x - threshold)))
            
            # Scale from [0, 1] to [-10, 10] as paper specifies
            wheel_speed = (sigmoid_01 * 2.0 - 1.0) * 10.0
            
            return wheel_speed
        except OverflowError:
            # Handle extreme values
            if x - threshold > 500:
                return 10.0  # Max forward
            else:
                return -10.0  # Max reverse
        
    def update(self, dt, environment):
        """Update the animat's state for one timestep.
        
        Args:
            dt: Time delta in seconds
            environment: The environment the animat is in
        """
        if not self.active:
            return
        
        # Track survival time
        self.survival_time += dt
            
        # Get sensor readings from environment
        sensor_readings = environment.get_sensor_readings(self)
        
        # Compute wheel speeds
        self.compute_wheel_speed(sensor_readings)
        
        # Move based on wheel speeds
        self.move(dt)
        
        # CORRECTED: Battery decay rate from paper
        # "decreases by 1 each time step" - assuming dt represents one timestep
        decay_amount = settings.BATTERY_DECAY_RATE * dt
        self.batteries[0] -= decay_amount
        self.batteries[1] -= decay_amount
        
        # Ensure batteries don't go below 0
        self.batteries[0] = max(0, self.batteries[0])
        self.batteries[1] = max(0, self.batteries[1])
        
        # Die if both batteries are empty
        if self.batteries[0] <= 0 and self.batteries[1] <= 0:
            self.active = False
            
    def move(self, dt):
        """Move the animat based on wheel speeds using differential drive.
        
        CORRECTED: Paper specifies that wheel speeds +10,+10 result in 
        "maximum speed of 2.8 units per time step"
        
        Args:
            dt: Time delta in seconds
        """
        # Wheel speeds are already in [-10, 10] range
        left_speed = self.wheel_speeds[0]
        right_speed = self.wheel_speeds[1]
        
        # Forward velocity is average of wheel speeds
        forward_velocity = (left_speed + right_speed) / 2.0
        
        # CORRECTED: Apply speed scaling factor from paper
        # When both wheels = +10, forward_velocity = 10, should move 2.8 units/timestep
        speed_scale_factor = settings.ANIMAT_MAX_SPEED / 10.0  # = 0.28
        actual_forward_velocity = forward_velocity * speed_scale_factor
        # Angular velocity is proportional to speed difference
        wheelbase = 2.0 * self.radius
        angular_velocity = (right_speed - left_speed) / wheelbase * speed_scale_factor
        
        # Update orientation first
        current_angle = np.arctan2(self.direction[1], self.direction[0])
        new_angle = current_angle + angular_velocity * dt
        
        # Update direction vector
        self.direction = np.array([np.cos(new_angle), np.sin(new_angle)])
        
        # Move forward in the new direction
        self.position += actual_forward_velocity * dt * self.direction
        
    def get_fitness(self):
        """Calculate the fitness of this animat based on survival time or battery level.
        
        Returns:
            Fitness score (survival time in seconds or normalized battery level)
        """
        # F = (B1 + B2) / (2 * BATTERY_MAX) to align with paper's F = (B1 + B2)/400.0
        # where BATTERY_MAX from paper is 200.

        if settings.BATTERY_FITNESS_MODE:
            return (self.batteries[0] + self.batteries[1]) / (2.0 * settings.BATTERY_MAX)
        else:
            return self.survival_time

    def get_forward_speed(self):
        """
        Calculates the current forward speed of the animat.
        This is the average of its two wheel speeds.
        """
        speed_scale_factor = settings.ANIMAT_MAX_SPEED / 10.0  # = 0.28
        actual_forward_velocity = (self.wheel_speeds[0] + self.wheel_speeds[1]) / 2.0 * speed_scale_factor
        return actual_forward_velocity