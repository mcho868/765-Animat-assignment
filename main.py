"""
Main entry point for the Animat simulation.
Evolves animats with sensorimotor links using a genetic algorithm.
"""
import argparse
import random
import numpy as np
from config import settings
from core.simulator import Simulator

def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Animat Evolution Simulation")
    
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no visualization)")
    parser.add_argument("--generations", type=int, default=settings.NUM_GENERATIONS, 
                        help="Number of generations to evolve")
    parser.add_argument("--population", type=int, default=settings.POPULATION_SIZE,
                        help="Population size for genetic algorithm")
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED,
                        help="Random seed for reproducibility (None for random)")
    parser.add_argument("--visualize-evolution", action="store_true",
                        help="Visualize evolution in progress instead of headless mode")
    
    return parser.parse_args()

def main():
    """Main function to set up and run the simulation."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Override settings with command line arguments
    settings.HEADLESS_MODE = args.headless
    settings.POPULATION_SIZE = args.population
    settings.NUM_GENERATIONS = args.generations
    
    # Create simulator
    simulator = Simulator(headless=settings.HEADLESS_MODE)
    
    # Run evolution
    print(f"Starting evolution with population size {settings.POPULATION_SIZE} "
            f"for {settings.NUM_GENERATIONS} generations")
    
    if args.visualize_evolution:
        # Visualization specific settings for run_evolution_with_visualization
        # These could be hardcoded or moved to config.py if they are fixed
        parallel_viz_default = 100 # Default if not specified elsewhere
        speed_multiplier_default = 1000 # Default if not specified elsewhere
        
        print(f"Visualizing evolution in progress with {parallel_viz_default} parallel animats (speed: {speed_multiplier_default}x)")
        best_genome, best_fitness = simulator.run_evolution_with_visualization(
            args.generations, parallel_count=parallel_viz_default, speed_multiplier=speed_multiplier_default)
        
        # Optionally, run the best animat after visualization
        # This part can be kept or removed based on desired behavior after visualized evolution
        print("Running simulation with best evolved animat after visualization")
        simulator.run_best_animat(best_genome, max_time=6000, speed_multiplier=10) # Example parameters
    
    print(f"Evolution complete! Best fitness: {best_fitness:.2f}")
    
    # Plot statistics
    simulator.plot_stats()
    
    # Clean up
    simulator.cleanup()
    
if __name__ == "__main__":
    main() 