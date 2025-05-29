#!/usr/bin/env python3
"""
Test script to demonstrate the new trajectory and capture marker features.
"""

from core.simulator import Simulator
from config import settings

def test_trajectory_and_markers():
    """Test the complete trajectory and capture markers functionality."""
    
    # Create a simulator with visualization enabled
    simulator = Simulator(headless=False)
    
    print("Testing Seth's model with complete trajectory and capture markers...")
    print("- The trajectory will show the entire path without disappearing")
    print("- Green X marks will appear when food is captured")
    print("- Blue X marks will appear when water is captured")
    print("- Press ESC to exit the simulation")
    
    # Run Seth's model for 60 seconds with moderate speed
    simulator.run_seth_model(max_time=60, speed_multiplier=1.0)
    
    print(f"Simulation completed!")
    print(f"Total trajectory points: {len(simulator.trajectory_to_draw)}")
    print(f"Total capture events: {len(simulator.capture_markers)}")
    
    # Count food vs water captures
    food_captures = sum(1 for marker in simulator.capture_markers if marker[2].name == 'FOOD')
    water_captures = sum(1 for marker in simulator.capture_markers if marker[2].name == 'WATER')
    
    print(f"Food captures: {food_captures}")
    print(f"Water captures: {water_captures}")

if __name__ == "__main__":
    test_trajectory_and_markers() 