"""
Logger for tracking animat behavior, battery levels, and genetic algorithm progress.
"""
import os
import csv
import time
import json
import numpy as np
from datetime import datetime

class Logger:
    """Logger for the animat simulation."""
    
    def __init__(self, log_dir="logs"):
        """Initialize logger with a directory for saving logs.
        
        Args:
            log_dir: Directory to save logs in
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.battery_log_file = os.path.join(log_dir, f"battery_log_{self.timestamp}.csv")
        self.behavior_log_file = os.path.join(log_dir, f"behavior_log_{self.timestamp}.csv")
        self.simulation_log_file = os.path.join(log_dir, f"simulation_log_{self.timestamp}.json")
        self.speed_log_file = os.path.join(log_dir, f"speed_log_{self.timestamp}.csv")
        
        # Initialize CSV files with headers
        with open(self.battery_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Animat ID", "Battery1", "Battery2", "Position X", "Position Y"])
            
        with open(self.behavior_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Animat ID", "Left Wheel", "Right Wheel", "Direction X", "Direction Y", "Speed"])
            
        with open(self.speed_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "AverageSpeed"])
        
        self.simulation_data = {
            "start_time": time.time(),
            "settings": {},
            "generations": [],
            "metrics": {}
        }
    
    def log_battery(self, animat_id, battery1, battery2, position):
        """Log an animat's battery levels and position.
        
        Args:
            animat_id: ID of the animat
            battery1: Level of battery 1
            battery2: Level of battery 2
            position: (x, y) position of the animat
        """
        with open(self.battery_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), animat_id, battery1, battery2, position[0], position[1]])
    
    def log_behavior(self, animat_id, wheel_speeds, direction):
        """Log an animat's behavior (wheel speeds and direction).
        
        Args:
            animat_id: ID of the animat
            wheel_speeds: [left_wheel, right_wheel] speeds
            direction: (x, y) direction vector
        """
        # Calculate the forward speed as the average of the two wheel speeds
        forward_speed = (wheel_speeds[0] + wheel_speeds[1]) / 2.0
        
        with open(self.behavior_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), animat_id, wheel_speeds[0], wheel_speeds[1], direction[0], direction[1], forward_speed])
    
    def log_generation(self, generation_num, fitness_scores, best_genome, avg_fitness):
        """Log data about a completed generation in the genetic algorithm.
        
        Args:
            generation_num: Generation number
            fitness_scores: List of fitness scores for the population
            best_genome: Genome of the best individual
            avg_fitness: Average fitness of the population
        """
        # Convert numpy array to list for JSON serialization
        if isinstance(best_genome, np.ndarray):
            best_genome = best_genome.tolist()
        
        generation_data = {
            "generation": generation_num,
            "best_fitness": max(fitness_scores) if fitness_scores else 0,
            "avg_fitness": avg_fitness,
            "best_genome": best_genome
        }
        self.simulation_data["generations"].append(generation_data)
        
        # Save after each generation to ensure data is not lost
        self._save_simulation_data()
    
    def log_metrics(self, metric_name, value):
        """Log a simulation metric.
        
        Args:
            metric_name: Name of the metric
            value: Value of the metric
        """
        if metric_name not in self.simulation_data["metrics"]:
            self.simulation_data["metrics"][metric_name] = []
        
        self.simulation_data["metrics"][metric_name].append({
            "time": time.time(),
            "value": value
        })
    
    def set_settings(self, settings):
        """Store the simulation settings.
        
        Args:
            settings: Dict of simulation settings
        """
        # Filter out non-JSON serializable objects
        json_safe_settings = {}
        for key, value in settings.items():
            # Only include primitive types that are JSON serializable
            if isinstance(value, (str, int, float, bool, list, dict)) and not key.startswith('__'):
                json_safe_settings[key] = value
        
        self.simulation_data["settings"] = json_safe_settings
        self._save_simulation_data()
    
    def finalize(self):
        """Finalize the log files and add summary information."""
        self.simulation_data["end_time"] = time.time()
        self.simulation_data["duration"] = self.simulation_data["end_time"] - self.simulation_data["start_time"]
        
        self._save_simulation_data()
        
        print(f"Logs saved to {self.log_dir}")
    
    def _save_simulation_data(self):
        """Save the simulation data to the JSON file."""
        with open(self.simulation_log_file, 'w') as f:
            json.dump(self.simulation_data, f, indent=2)
    
    def log_average_speed(self, generation_num, avg_speed):
        """Log the average speed for a generation."""
        with open(self.speed_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([generation_num, avg_speed]) 