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
    parser.add_argument("--run-best", action="store_true",
                        help="Run simulation with the best evolved animat")
    parser.add_argument("--visualize-only", action="store_true",
                        help="Only visualize, don't evolve")
    parser.add_argument("--visualize-evolution", action="store_true",
                        help="Visualize evolution in progress instead of headless mode")
    parser.add_argument("--parallel-viz", type=int, default=1,
                        help="Number of animats to visualize in parallel (default: 1)")
    parser.add_argument("--speed-multiplier", type=float, default=1.0,
                        help="Speed multiplier for simulation (higher = faster, default: 1.0)")
    
    return parser.parse_args()

def main():
    """Main function to set up and run the simulation."""
    # Parse command line arguments
    try:
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
        
        if args.visualize_only:
            # Just visualize a random animat
            simulator.environment.initialize_random_environment()
            
            # Create and add a random animat
            from agents.base_agent import Animat
            center_pos = (simulator.environment.width/2, simulator.environment.height/2)
            animat = Animat(center_pos)
            simulator.environment.add_entity(animat)
            
            # Run visualization
            print("Running visualization with a random animat")
            simulator.run_evolution_with_visualization()
            # simulator.run_best_animat(animat.genome, max_time=60, speed_multiplier=args.speed_multiplier)
            
        else:
            # Run evolution
            print(f"Starting evolution with population size {settings.POPULATION_SIZE} "
                f"for {settings.NUM_GENERATIONS} generations")
            
            if args.visualize_evolution:
                print(f"Visualizing evolution in progress with {args.parallel_viz} parallel animats (speed: {args.speed_multiplier}x)")
                best_genome, best_fitness = simulator.run_evolution_with_visualization(
                    args.generations, parallel_count=args.parallel_viz, speed_multiplier=args.speed_multiplier)
            else:
                best_genome, best_fitness = simulator.run_evolution(args.generations)
            
            print(f"Evolution complete! Best fitness: {best_fitness:.2f}")
            
            # Plot statistics
            simulator.plot_stats()
            
            # Run simulation with best genome if requested
            if args.run_best:
                print("Running simulation with best evolved animat")
                simulator.run_best_animat(best_genome, max_time=60, speed_multiplier=args.speed_multiplier)
    except Exception as e:           # ← 指定 Exception 并打印 traceback
        import traceback
        traceback.print_exc()
        raise
    # Clean up
    finally:
        if 'simulator' in locals():
            simulator.logger.finalize()
            simulator.cleanup()
    
if __name__ == "__main__":
    main() 