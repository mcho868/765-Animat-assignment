#!/usr/bin/env python3
"""
Test script to validate Animat piecewise functions and wheel speed computation.
This script creates test animats, plots their activation functions, and verifies
the complete sensor-to-wheel processing pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Mock settings for testing (adjust paths as needed)
class MockSettings:
    ANIMAT_SIZE = 5.0
    BATTERY_MAX = 200.0
    BATTERY_DECAY_RATE = 1.0
    GENOTYPE_SIZE = 83
    NUM_LINKS = 9
    LINK_PARAM_COUNT = 9

# Mock EntityType for testing
class MockEntityType:
    ANIMAT = "animat"

# Import or mock the Animat class (adjust import as needed)
# from agents.base_agent import Animat

# For testing, we'll create a simplified version
class TestAnimat:
    """Simplified Animat class for testing"""
    
    def __init__(self, genome=None):
        self.position = np.array([100.0, 100.0])
        self.radius = MockSettings.ANIMAT_SIZE
        self.active = True
        self.type = MockEntityType.ANIMAT
        self.survival_time = 0.0
        self.direction = np.array([1.0, 0.0])
        self.batteries = [MockSettings.BATTERY_MAX, MockSettings.BATTERY_MAX]
        self.wheel_speeds = [0.0, 0.0]
        
        if genome is None:
            self.create_test_genome()
        else:
            self.genome = genome.copy()
            
        # Extract sigmoid thresholds
        self.sigmoid_thresholds = [
            self._scale_sigmoid_genome_value(self.genome[-2]),
            self._scale_sigmoid_genome_value(self.genome[-1])
        ]
        
        self.parse_genome()
    
    def _scale_genome_value(self, val, min_val, max_val):
        """Scales a genome integer (0-99) to the range [min_val, max_val]."""
        return (val / 99.0) * (max_val - min_val) + min_val

    def _scale_sigmoid_genome_value(self, val):
        """Scales a genome integer (0-99) to the range [-3.0, 3.0]."""
        return (val / 99.0) * 6.0 - 3.0
    
    def create_test_genome(self):
        """Create a test genome with known values for validation"""
        self.genome = np.zeros(MockSettings.GENOTYPE_SIZE, dtype=int)
        
        # Link 0: Simple rising function
        self.genome[0] = 10   # offset -> -80
        self.genome[1] = 75   # grad1 -> positive slope
        self.genome[2] = 30   # thresh1 -> -40
        self.genome[3] = 75   # grad2 -> positive slope
        self.genome[4] = 70   # thresh2 -> 40
        self.genome[5] = 75   # grad3 -> positive slope
        self.genome[6] = 50   # slope_mod -> 0.5
        self.genome[7] = 60   # offset_mod -> 0.2
        self.genome[8] = 0    # battery 0
        
        # Link 1: V-shaped function
        self.genome[9] = 80   # offset -> 60
        self.genome[10] = 25  # grad1 -> negative slope
        self.genome[11] = 50  # thresh1 -> 0
        self.genome[12] = 75  # grad2 -> positive slope
        self.genome[13] = 80  # thresh2 -> 60
        self.genome[14] = 75  # grad3 -> positive slope
        self.genome[15] = 30  # slope_mod -> 0.3
        self.genome[16] = 40  # offset_mod -> -0.2
        self.genome[17] = 1   # battery 1
        
        # Link 2: Step function
        self.genome[18] = 50  # offset -> 0
        self.genome[19] = 50  # grad1 -> ~0 slope
        self.genome[20] = 60  # thresh1 -> 20
        self.genome[21] = 10  # grad2 -> very negative
        self.genome[22] = 80  # thresh2 -> 60
        self.genome[23] = 50  # grad3 -> ~0 slope
        self.genome[24] = 20  # slope_mod -> 0.2
        self.genome[25] = 50  # offset_mod -> 0
        self.genome[26] = 0   # battery 0
        
        # Fill remaining links with simple patterns
        for i in range(3, 9):
            base_idx = i * 9
            self.genome[base_idx:base_idx+9] = [50, 60, 40, 60, 80, 60, 25, 50, i % 2]
        
        # Sigmoid thresholds
        self.genome[81] = 50  # 0.0
        self.genome[82] = 50  # 0.0
    
    def parse_genome(self):
        """Parse genome into links (simplified version)"""
        self.links = []
        
        # Parse 9 encoded links
        encoded_links = []
        for i in range(0, MockSettings.NUM_LINKS * MockSettings.LINK_PARAM_COUNT, MockSettings.LINK_PARAM_COUNT):
            offset_val = self._scale_genome_value(self.genome[i], -100.0, 100.0)
            
            angle1 = (self.genome[i + 1] / 99.0) * np.pi - (np.pi / 2.0)
            angle1 = np.clip(angle1, -np.pi/2.0 * 0.99, np.pi/2.0 * 0.99)
            grad1_val = np.tan(angle1)
            
            thresh1_val = self._scale_genome_value(self.genome[i + 2], -100.0, 100.0)
            
            angle2 = (self.genome[i + 3] / 99.0) * np.pi - (np.pi / 2.0)
            angle2 = np.clip(angle2, -np.pi/2.0 * 0.99, np.pi/2.0 * 0.99)
            grad2_val = np.tan(angle2)
            
            thresh2_raw_scaled = self._scale_genome_value(self.genome[i + 4], -100.0, 100.0)
            thresh2_val = max(thresh1_val, thresh2_raw_scaled)
            
            angle3 = (self.genome[i + 5] / 99.0) * np.pi - (np.pi / 2.0)
            angle3 = np.clip(angle3, -np.pi/2.0 * 0.99, np.pi/2.0 * 0.99)
            grad3_val = np.tan(angle3)
            
            slope_mod_val = self.genome[i + 6] / 99.0
            offset_mod_val = self._scale_genome_value(self.genome[i + 7], -1.0, 1.0)
            battery_val = self.genome[i + 8]
            
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
        
        # Create 18 links with symmetry
        for sensor_idx in range(6):
            for link_offset in range(3):
                if sensor_idx % 2 == 0:  # Left sensor
                    encoded_idx = (sensor_idx // 2) * 3 + link_offset
                    self.links.append(encoded_links[encoded_idx].copy())
                else:  # Right sensor
                    left_sensor_idx = sensor_idx - 1
                    encoded_idx = (left_sensor_idx // 2) * 3 + link_offset
                    self.links.append(encoded_links[encoded_idx].copy())
    
    def compute_sensor_to_wheel_output(self, sensor_value, link_index):
        """Compute link output with battery modulation"""
        link = self.links[link_index]
        battery_level = self.batteries[link['battery']]
        
        # Battery modulation
        offset_modulation = (battery_level / 2.0) * link['offset_mod']
        slope_modulation_factor = ((battery_level - 100.0) / 100.0) * link['slope_mod']
        
        # Piecewise linear function
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
        
        return max(-1.0, min(1.0, output/100))
    
    def sigmoid_to_wheel_speed(self, x, threshold):
        """Convert summed link outputs to wheel speed"""
        try:
            sigmoid_01 = 1.0 / (1.0 + np.exp(-(x - threshold)))
            wheel_speed = (sigmoid_01 * 2.0 - 1.0) * 10.0
            return wheel_speed
        except OverflowError:
            return 10.0 if x - threshold > 500 else -10.0
    
    def compute_wheel_speed(self, sensor_readings):
        """Compute wheel speeds from sensor readings"""
        sensor_values = [
            sensor_readings['food_left'],
            sensor_readings['food_right'],
            sensor_readings['water_left'],
            sensor_readings['water_right'],
            sensor_readings['trap_left'],
            sensor_readings['trap_right']
        ]
        
        left_wheel_sum = 0.0
        right_wheel_sum = 0.0
        
        # Store link outputs for analysis
        self.link_outputs = []
        
        for link_idx in range(18):
            sensor_idx = link_idx // 3
            sensor_value = sensor_values[sensor_idx]
            
            output = self.compute_sensor_to_wheel_output(sensor_value, link_idx)
            self.link_outputs.append(output)
            
            if link_idx < 9:
                left_wheel_sum += output
            else:
                right_wheel_sum += output
        
        left_wheel = self.sigmoid_to_wheel_speed(left_wheel_sum, self.sigmoid_thresholds[0])
        right_wheel = self.sigmoid_to_wheel_speed(right_wheel_sum, self.sigmoid_thresholds[1])
        
        self.wheel_speeds = [left_wheel, right_wheel]
        self.wheel_sums = [left_wheel_sum, right_wheel_sum]
        
        return self.wheel_speeds

def test_piecewise_functions():
    """Test and plot piecewise activation functions"""
    print("=" * 60)
    print("TESTING PIECEWISE ACTIVATION FUNCTIONS")
    print("=" * 60)
    
    # Create test animat
    animat = TestAnimat()
    
    # Test different battery levels
    battery_levels = [200, 100, 0]
    sensor_range = np.linspace(0, 100, 1000)
    
    # Plot first 3 links to show variety
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for link_idx in range(3):
        ax = axes[link_idx]
        
        for battery_level in battery_levels:
            # Set battery level
            original_battery = animat.batteries[animat.links[link_idx]['battery']]
            animat.batteries[animat.links[link_idx]['battery']] = battery_level
            
            outputs = []
            for sensor_value in sensor_range:
                output = animat.compute_sensor_to_wheel_output(sensor_value, link_idx)
                outputs.append(output)
            
            # Restore battery
            animat.batteries[animat.links[link_idx]['battery']] = original_battery
            
            ax.plot(sensor_range, outputs, label=f'Battery {battery_level}', linewidth=2)
        
        # Plot details
        link = animat.links[link_idx]
        ax.set_title(f'Link {link_idx} (Battery {link["battery"]})')
        ax.set_xlabel('Sensor Input (0-100)')
        ax.set_ylabel('Link Output')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-1.1, 1.1)
        
        # Add threshold markers
        ax.axvline(link['thresh1'], color='red', linestyle='--', alpha=0.5, label='Thresh1')
        ax.axvline(link['thresh2'], color='orange', linestyle='--', alpha=0.5, label='Thresh2')
        
        print(f"Link {link_idx} parameters:")
        print(f"  Offset: {link['offset']:.2f}")
        print(f"  Gradients: {link['grad1']:.2f}, {link['grad2']:.2f}, {link['grad3']:.2f}")
        print(f"  Thresholds: {link['thresh1']:.2f}, {link['thresh2']:.2f}")
        print(f"  Battery: {link['battery']}")
        print()
    
    plt.tight_layout()
    plt.savefig('piecewise_functions_test.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_wheel_speed_computation():
    """Test complete sensor-to-wheel processing pipeline"""
    print("=" * 60)
    print("TESTING WHEEL SPEED COMPUTATION PIPELINE")
    print("=" * 60)
    
    # Create test animat
    animat = TestAnimat()
    
    # Test sensor readings
    test_cases = [
        {
            'name': 'No stimuli',
            'sensors': {'food_left': 0, 'food_right': 0, 'water_left': 0, 
                       'water_right': 0, 'trap_left': 0, 'trap_right': 0}
        },
        {
            'name': 'Food on left',
            'sensors': {'food_left': 80, 'food_right': 10, 'water_left': 0, 
                       'water_right': 0, 'trap_left': 0, 'trap_right': 0}
        },
        {
            'name': 'Water on right',
            'sensors': {'food_left': 0, 'food_right': 0, 'water_left': 10, 
                       'water_right': 70, 'trap_left': 0, 'trap_right': 0}
        },
        {
            'name': 'Trap detected',
            'sensors': {'food_left': 20, 'food_right': 20, 'water_left': 20, 
                       'water_right': 20, 'trap_left': 60, 'trap_right': 30}
        },
        {
            'name': 'Mixed stimuli',
            'sensors': {'food_left': 40, 'food_right': 60, 'water_left': 30, 
                       'water_right': 50, 'trap_left': 20, 'trap_right': 10}
        }
    ]
    
    print("Test Cases:")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{i+1}. {test_case['name']}")
        print(f"   Sensors: {test_case['sensors']}")
        
        # Compute wheel speeds
        wheel_speeds = animat.compute_wheel_speed(test_case['sensors'])
        
        print(f"   Link outputs: {[f'{x:.2f}' for x in animat.link_outputs]}")
        print(f"   Wheel sums: Left={animat.wheel_sums[0]:.2f}, Right={animat.wheel_sums[1]:.2f}")
        print(f"   Sigmoid thresholds: {animat.sigmoid_thresholds}")
        print(f"   Final wheel speeds: Left={wheel_speeds[0]:.2f}, Right={wheel_speeds[1]:.2f}")
        
        # Interpret movement
        left, right = wheel_speeds
        if abs(left) < 0.1 and abs(right) < 0.1:
            movement = "Stationary"
        elif left > 0 and right > 0:
            if abs(left - right) < 0.5:
                movement = "Forward"
            elif left > right:
                movement = "Forward + Right turn"
            else:
                movement = "Forward + Left turn"
        elif left < 0 and right < 0:
            movement = "Backward"
        elif left > 0 > right:
            movement = "Spin right"
        elif right > 0 > left:
            movement = "Spin left"
        else:
            movement = "Complex movement"
            
        print(f"   Movement: {movement}")

def test_battery_effects():
    """Test how battery levels affect behavior"""
    print("=" * 60)
    print("TESTING BATTERY LEVEL EFFECTS")
    print("=" * 60)
    
    animat = TestAnimat()
    
    # Test sensor input
    test_sensors = {
        'food_left': 50, 'food_right': 30, 'water_left': 40, 
        'water_right': 60, 'trap_left': 20, 'trap_right': 10
    }
    
    battery_levels = [200, 150, 100, 50, 0]
    
    print("Battery Level Effects:")
    print("-" * 40)
    
    for battery_level in battery_levels:
        # Set both batteries to test level
        animat.batteries = [battery_level, battery_level]
        
        wheel_speeds = animat.compute_wheel_speed(test_sensors)
        
        print(f"Battery {battery_level:3d}: Left={wheel_speeds[0]:6.2f}, Right={wheel_speeds[1]:6.2f}, "
              f"Sum=({animat.wheel_sums[0]:5.2f}, {animat.wheel_sums[1]:5.2f})")

def run_all_tests():
    """Run all validation tests"""
    print("ANIMAT VALIDATION TEST SUITE")
    print("=" * 60)
    
    try:
        test_piecewise_functions()
        test_wheel_speed_computation()
        test_battery_effects()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("Check 'piecewise_functions_test.png' for function plots.")
        print("=" * 60)
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()