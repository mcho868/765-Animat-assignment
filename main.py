"""
Main entry point for the Braitenberg-inspired simulation environment.
This file sets up and runs the simulation with specified agents.
"""
import random
import argparse
from config import settings
from core.simulator import Simulator
from agents.simple_agent import SimpleAgent
from agents.stress_agent import StressAgent

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Braitenberg-inspired simulation")
    
    parser.add_argument("--headless", action="store_true", help="Run simulation without visualization")
    parser.add_argument("--agents", type=int, default=settings.AGENT_COUNT, help="Number of agents to simulate")
    parser.add_argument("--obstacles", type=int, default=settings.OBSTACLE_COUNT, help="Number of obstacles in environment")
    parser.add_argument("--lights", type=int, default=settings.LIGHT_SOURCE_COUNT, help="Number of light sources")
    parser.add_argument("--food", type=int, default=settings.FOOD_SOURCE_COUNT, help="Number of food sources")
    parser.add_argument("--time", type=float, default=None, help="Maximum simulation time in seconds")
    parser.add_argument("--simple-ratio", type=float, default=0.5, 
                        help="Ratio of simple agents to stress agents (0-1)")
    
    return parser.parse_args()

def create_agents(simulator, count, simple_ratio):
    """Create and add agents to the simulator."""
    # Calculate counts
    simple_count = int(count * simple_ratio)
    stress_count = count - simple_count
    
    agents = []
    
    # Create simple agents with different behaviors
    for i in range(simple_count):
        # Random position avoiding obstacles
        valid_position = False
        while not valid_position:
            x = random.randint(50, simulator.width - 50)
            y = random.randint(50, simulator.height - 50)
            position = (x, y)
            collision, _ = simulator.environment.check_collision(position, settings.AGENT_SIZE, None)
            valid_position = not collision
            
        # Assign behavior type (1, 2, or 3)
        behavior_type = random.choice([1, 2, 3])
        
        # Create the agent
        agent = SimpleAgent(i, position, behavior_type)
        agents.append(agent)
        
    # Create stress agents
    for i in range(simple_count, count):
        # Random position avoiding obstacles
        valid_position = False
        while not valid_position:
            x = random.randint(50, simulator.width - 50)
            y = random.randint(50, simulator.height - 50)
            position = (x, y)
            collision, _ = simulator.environment.check_collision(position, settings.AGENT_SIZE, None)
            valid_position = not collision
            
        # Create the agent
        agent = StressAgent(i, position)
        agents.append(agent)
    
    # Add all agents to simulator
    simulator.add_agents(agents)
    
    return agents

def main():
    """Main function to set up and run the simulation."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Override settings with command line arguments
    if args.headless:
        settings.HEADLESS_MODE = True
    
    # Create simulator
    simulator = Simulator(headless=settings.HEADLESS_MODE)
    
    # Initialize environment with entities
    simulator.environment.initialize_random_environment(
        args.obstacles, args.lights, args.food
    )
    
    # Create and add agents
    agents = create_agents(simulator, args.agents, args.simple_ratio)
    
    print(f"Starting simulation with {len(agents)} agents")
    print(f"Environment has {len(simulator.environment.obstacles)} obstacles, "
          f"{len(simulator.environment.light_sources)} light sources, and "
          f"{len(simulator.environment.food_sources)} food sources")
    
    # Run the simulation
    simulator.run(max_time=args.time)
    
if __name__ == "__main__":
    main() 