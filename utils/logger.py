"""
Logger for tracking agent behavior, stress levels, and simulation metrics.
"""
import os
import csv
import time
import json
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs"):
        """Initialize logger with a directory for saving logs."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stress_log_file = os.path.join(log_dir, f"stress_log_{self.timestamp}.csv")
        self.behavior_log_file = os.path.join(log_dir, f"behavior_log_{self.timestamp}.csv")
        self.simulation_log_file = os.path.join(log_dir, f"simulation_log_{self.timestamp}.json")
        
        # Initialize CSV files with headers
        with open(self.stress_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Agent ID", "Stress Level", "Position X", "Position Y"])
            
        with open(self.behavior_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Agent ID", "Action", "Heading", "Speed", "Stimulus"])
            
        self.simulation_data = {
            "start_time": time.time(),
            "settings": {},
            "generations": [],
            "metrics": {}
        }
    
    def log_stress(self, agent_id, stress_level, position):
        """Log an agent's stress level and position."""
        with open(self.stress_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), agent_id, stress_level, position[0], position[1]])
    
    def log_behavior(self, agent_id, action, heading, speed, stimulus=None):
        """Log an agent's behavior (action taken)."""
        with open(self.behavior_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), agent_id, action, heading, speed, stimulus])
    
    def log_generation(self, generation_num, fitness_scores, best_genome, avg_fitness):
        """Log data about a completed generation in evolutionary algorithms."""
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
        """Log a simulation metric."""
        if metric_name not in self.simulation_data["metrics"]:
            self.simulation_data["metrics"][metric_name] = []
        
        self.simulation_data["metrics"][metric_name].append({
            "time": time.time(),
            "value": value
        })
    
    def set_settings(self, settings):
        """Store the simulation settings."""
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