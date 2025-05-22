"""
Main simulation loop and rendering for the Animat environment.
"""
import time
import pygame
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from config import settings
from core.environment import Environment, EntityType
from agents.base_agent import Animat
from agents.agent_logic import GeneticAlgorithm, simulate_animat
from utils.logger import Logger

class Simulator:
    """
    Main simulator class that handles:
    - Running the simulation loop
    - Rendering the simulation using PyGame
    - Managing the genetic algorithm
    - Logging results
    """
    
    def __init__(self, width=settings.WINDOW_WIDTH, height=settings.WINDOW_HEIGHT, 
                 headless=settings.HEADLESS_MODE):
        """Initialize the simulator.
        
        Args:
            width: Window width for visualization
            height: Window height for visualization
            headless: Whether to run in headless mode (no visualization)
        """
        self.width = width
        self.height = height
        self.headless = headless
        
        # Initialize the environment
        self.environment = Environment()
        
        # Initialize the genetic algorithm
        self.ga = GeneticAlgorithm()
        
        # Initialize logging
        self.logger = Logger()
        
        # Initialize pygame if not headless
        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(settings.WINDOW_TITLE)
            self.clock = pygame.time.Clock()
            
            # Load or create assets
            self.font = pygame.font.SysFont('Arial', 12)
            self.colors = {
                EntityType.FOOD: (0, 255, 0),          # Green
                EntityType.WATER: (0, 0, 255),         # Blue
                EntityType.TRAP: (255, 0, 0),          # Red
                EntityType.ANIMAT: (255, 255, 0),      # Yellow
            }
            
        self.is_running = False
        self.generation = 0
        self.simulation_time = 0
        self.fps = settings.FPS
        
        # Statistics for plotting
        self.generation_stats = {
            'generation': [],
            'max_fitness': [],
            'avg_fitness': [],
            'min_fitness': []
        }
        
    def initialize_ga(self):
        """Initialize the genetic algorithm."""
        self.ga.initialize_population()
        
    def handle_events(self):
        """Handle pygame events."""
        if self.headless:
            return True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                    
        return True
        
    def update(self, dt):
        """Update the simulation by one step.
        
        Args:
            dt: Time delta in seconds
        """
        # Update environment
        self.environment.update(dt)
        
        # Update simulation time
        self.simulation_time += dt
        
    def render(self):
        """Render the current state of the simulation."""
        if self.headless:
            return
            
        # Clear the screen
        self.screen.fill((255, 255, 255))
        
        # Calculate scale factors to map from environment coordinates to screen coordinates
        scale_x = self.width / self.environment.width
        scale_y = self.height / self.environment.height
        
        # Render entities
        for entity in self.environment.entities:
            if not entity.active:
                continue
                
            color = self.colors.get(entity.type, (0, 0, 0))
            
            # Draw entity
            screen_x = int(entity.position[0] * scale_x)
            screen_y = int(entity.position[1] * scale_y)
            screen_radius = int(entity.radius * scale_x)  # Use scale_x for consistent scaling
            
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), screen_radius)
            
            # Draw entity direction if it's an animat
            if entity.type == EntityType.ANIMAT:
                # Draw a line showing direction
                direction_end = (
                    screen_x + int(entity.radius * 2 * entity.direction[0]),
                    screen_y + int(entity.radius * 2 * entity.direction[1])
                )
                pygame.draw.line(self.screen, (0, 0, 0), (screen_x, screen_y), direction_end, 2)
                
                # Draw battery levels
                battery1_width = int(entity.batteries[0] / settings.BATTERY_MAX * 20)
                battery2_width = int(entity.batteries[1] / settings.BATTERY_MAX * 20)
                
                pygame.draw.rect(self.screen, (255, 0, 0), 
                                (screen_x - 10, screen_y - entity.radius - 10, battery1_width, 3))
                pygame.draw.rect(self.screen, (0, 0, 255), 
                                (screen_x - 10, screen_y - entity.radius - 5, battery2_width, 3))
        
        # Draw performance stats
        fps_text = self.font.render(f"FPS: {int(self.clock.get_fps())}", True, (0, 0, 0))
        time_text = self.font.render(f"Time: {self.simulation_time:.1f}s", True, (0, 0, 0))
        gen_text = self.font.render(f"Generation: {self.generation}/{settings.NUM_GENERATIONS}", True, (0, 0, 0))
        
        if self.ga.best_fitness > 0:
            fitness_text = self.font.render(f"Best Fitness: {self.ga.best_fitness:.1f}", True, (0, 0, 0))
            self.screen.blit(fitness_text, (10, 70))
        
        self.screen.blit(fps_text, (10, 10))
        self.screen.blit(time_text, (10, 30))
        self.screen.blit(gen_text, (10, 50))
        
        # Update the display
        pygame.display.flip()
        
    def render_environment(self, environment, x, y, width, height):
        """Render an environment in a specific section of the screen.
        
        Args:
            environment: The environment to render
            x, y: Top-left coordinates of the section
            width, height: Dimensions of the section
        """
        # Calculate scale factors to map from environment coordinates to screen coordinates
        scale_x = width / environment.width
        scale_y = height / environment.height
        
        # Render entities
        for entity in environment.entities:
            if not entity.active:
                continue
                
            color = self.colors.get(entity.type, (0, 0, 0))
            
            # Draw entity
            screen_x = int(x + entity.position[0] * scale_x)
            screen_y = int(y + entity.position[1] * scale_y)
            screen_radius = int(entity.radius * scale_x)  # Use scale_x for consistent scaling
            
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), screen_radius)
            
            # Draw entity direction if it's an animat
            if entity.type == EntityType.ANIMAT:
                # Draw a line showing direction
                direction_end = (
                    screen_x + int(entity.radius * 2 * entity.direction[0]),
                    screen_y + int(entity.radius * 2 * entity.direction[1])
                )
                pygame.draw.line(self.screen, (0, 0, 0), (screen_x, screen_y), direction_end, 2)
                
                # Draw battery levels
                battery1_width = int(entity.batteries[0] / settings.BATTERY_MAX * 20)
                battery2_width = int(entity.batteries[1] / settings.BATTERY_MAX * 20)
                
                pygame.draw.rect(self.screen, (255, 0, 0), 
                                (screen_x - 10, screen_y - entity.radius - 10, battery1_width, 3))
                pygame.draw.rect(self.screen, (0, 0, 255), 
                                (screen_x - 10, screen_y - entity.radius - 5, battery2_width, 3))
        
        # Draw section border for clarity
        pygame.draw.rect(self.screen, (200, 200, 200), (x, y, width, height), 1)
        
    def run_evolution(self,
                       num_generations=settings.NUM_GENERATIONS, 
                       parallel_count=1, speed_multiplier=1.0):
        """Run the evolutionary algorithm.
        
        Args:
            num_generations: Number of generations to evolve
            
        Returns:
            Tuple of (best_genome, best_fitness)
        """

        #keep track of original headless 
        old_headless = self.headless

        self.initialize_ga()
        

        for gen in range(num_generations):
            self.generation = gen

            print(f"Generation {gen+1}/{num_generations}")
            
            # Process animats in parallel batches
            batch_size = min(parallel_count, settings.POPULATION_SIZE)
            num_batches = (settings.POPULATION_SIZE + batch_size - 1) // batch_size
            
            fitnesses = []
            best_fitness = 0.0
            best_genome = None
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, settings.POPULATION_SIZE)
                batch_genomes = self.ga.population[start_idx:end_idx]
                
                print(f"  Processing batch {batch+1}/{num_batches} (animats {start_idx+1}-{end_idx})")
                
                # Create environments and animats for this batch
                environments ,animats = [],[]
                
                for genome in batch_genomes:
                    # Create a new environment for each animat
                    env = Environment()
                    env.initialize_random_environment()
                    
                    # Create animat with the genome
                    center_pos = (env.width/2, env.height/2)
                    animat = Animat(center_pos, genome)
                    env.add_entity(animat)
                    
                    environments.append(env)
                    animats.append(animat)
                
                # Run simulation loop 
                self.is_running = True
                self.simulation_time = 0
                last_time = time.time()
                steps = 0
                max_steps = settings.ANIMAT_MAX_LIFESPAN
                
                # Track which animats are still active
                active_animats = [True] * len(animats)
                any_active = True
                
                while self.is_running and any_active and steps < max_steps:
                    # Calculate delta time
                    current_time = time.time()
                    dt = min(current_time - last_time, 0.2) * speed_multiplier
                    last_time = current_time
                    any_active = False


                    for i, (env, animat) in enumerate(zip(environments, animats)):
                        if not active_animats[i]:
                            continue
                            
                        # Update this environment
                        env.update(dt)
                        
                        # mark inactive animates
                        if not animat.active:
                            active_animats[i] = False
                        else:
                            any_active = True

                         # log battery & behavior for this animat
                        
                        if animat.active:
                            self.logger.log_battery(
                                animat_id = start_idx + i,                 
                                battery1  = float(animat.batteries[0]),    
                                battery2  = float(animat.batteries[1]),
                                position  = tuple(map(float, animat.position))
                            )

                            self.logger.log_behavior(
                                animat_id    = start_idx + i,
                                wheel_speeds = list(map(float, animat.wheel_speeds)),
                                direction    = tuple(map(float, animat.direction))
                            )   

                        step_increment = max(1, int(speed_multiplier))
                        steps += step_increment

                        #Collect fitness for this batch 
                    for i,a in enumerate(animats):
                        f = a.get_fitness()
                        fitnesses.append(f)
                        if f > best_fitness:
                            best_fitness = f
                            best_genome = batch_genomes[i].copy() 

                #pad fitness list if population not multiple of batch_size
            if len(fitnesses) < settings.POPULATION_SIZE:
                fitnesses += [0.0] * (settings.POPULATION_SIZE - len(fitnesses))

            # Log stats
            avg_fit = sum(fitnesses) / len(fitnesses)
            self.generation_stats['generation'].append(gen)
            self.generation_stats['max_fitness'].append(best_fitness)
            self.generation_stats['avg_fitness'].append(avg_fit)
            self.generation_stats['min_fitness'].append(min(fitnesses))
            
            print(f"  Max Fitness: {best_fitness:.2f}")
            print(f"  Avg Fitness: {avg_fit:.2f}")
            print(f"  Min Fitness: {min(fitnesses):.2f}")
            
            self.logger.log_generation(gen, fitnesses, best_genome, avg_fit)
            
            # Update GA's best genome
            if best_genome is not None and (self.ga.best_genome is None or best_fitness > self.ga.best_fitness):
                self.ga.best_genome = best_genome.copy()
                self.ga.best_fitness = best_fitness
            
            # Evolve next generation (except for last generation)
            if gen < num_generations - 1:
                self.ga.evolve_generation()
                
        # Restore headless setting
        self.headless = old_headless
        # return self.ga.get_best_genome()
        return self.ga.get_best_genome(), self.ga.best_fitness

        
    def run_evolution_with_visualization(self, num_generations=settings.NUM_GENERATIONS, parallel_count=1, speed_multiplier=1.0):
        """Run the evolutionary algorithm with visualization.
        
        Args:
            num_generations: Number of generations to evolve
            parallel_count: Number of animats to visualize in parallel
            speed_multiplier: Speed multiplier for simulation (higher = faster)
            
        Returns:
            Tuple of (best_genome, best_fitness)
        """
        self.initialize_ga()
        
        
        # Ensure visualization is enabled
        old_headless = self.headless
        self.headless = False
        
        # Initialize PyGame if needed
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(settings.WINDOW_TITLE)
            self.clock = pygame.time.Clock()
            
            # Load or create assets
            self.font = pygame.font.SysFont('Arial', 12)
            self.colors = {
                EntityType.FOOD: (0, 255, 0),          # Green
                EntityType.WATER: (0, 0, 255),         # Blue
                EntityType.TRAP: (255, 0, 0),          # Red
                EntityType.ANIMAT: (255, 255, 0),      # Yellow
            }
        
        for gen in range(num_generations):
            self.generation = gen
            print(f"Generation {gen+1}/{num_generations}")
            
            # Process animats in parallel batches
            batch_size = min(parallel_count, settings.POPULATION_SIZE)
            num_batches = (settings.POPULATION_SIZE + batch_size - 1) // batch_size
            
            fitnesses = []
            best_fitness = 0
            best_genome = None
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, settings.POPULATION_SIZE)
                batch_genomes = self.ga.population[start_idx:end_idx]
                
                print(f"  Processing batch {batch+1}/{num_batches} (animats {start_idx+1}-{end_idx})")
                
                # Create environments and animats for this batch
                environments = []
                animats = []
                
                for genome in batch_genomes:
                    # Create a new environment for each animat
                    env = Environment()
                    env.initialize_random_environment()
                    
                    # Create animat with the genome
                    center_pos = (env.width/2, env.height/2)
                    animat = Animat(center_pos, genome)
                    env.add_entity(animat)
                    
                    environments.append(env)
                    animats.append(animat)
                
                # Run simulation loop with visualization for this batch
                self.is_running = True
                self.simulation_time = 0
                last_time = time.time()
                steps = 0
                max_steps = settings.ANIMAT_MAX_LIFESPAN
                
                # Display which batch is being simulated
                title_text = f"{settings.WINDOW_TITLE} - Gen {gen+1}/{num_generations}, Batch {batch+1}/{num_batches}"
                pygame.display.set_caption(title_text)
                
                # Track which animats are still active
                active_animats = [True] * len(animats)
                any_active = True
                
                while self.is_running and any_active and steps < max_steps:
                    # Calculate delta time
                    current_time = time.time()
                    dt = current_time - last_time
                    last_time = current_time
                    
                    # Apply speed multiplier
                    dt = dt * speed_multiplier
                    
                    # Cap dt to avoid large jumps (increased for speed)
                    dt = min(dt, 0.2 * speed_multiplier)
                    
                    # Handle events
                    if not self.handle_events():
                        self.is_running = False
                        break
                    
                    # Clear the screen
                    self.screen.fill((255, 255, 255))
                    
                    # Update and render each environment
                    any_active = False
                    
                    for i, (env, animat) in enumerate(zip(environments, animats)):
                        if not active_animats[i]:
                            continue
                            
                        # Update this environment
                        env.update(dt)
                        
                        # Check if animat is still active
                        if not animat.active:
                            active_animats[i] = False
                        else:
                            any_active = True
                        
                        if animat.active:    
                            self.logger.log_battery(
                                animat_id = start_idx + i,                 
                                battery1  = float(animat.batteries[0]),    
                                battery2  = float(animat.batteries[1]),
                                position  = tuple(map(float, animat.position))
                            )
                            self.logger.log_behavior(
                                animat_id    = start_idx + i,
                                wheel_speeds = list(map(float, animat.wheel_speeds)),
                                direction    = tuple(map(float, animat.direction))
                            )   

                        # Calculate the grid layout dimensions
                        grid_cols = int(np.ceil(np.sqrt(batch_size)))
                        grid_rows = int(np.ceil(batch_size / grid_cols))
                        
                        # Calculate the section dimensions
                        section_width = self.width // grid_cols
                        section_height = self.height // grid_rows
                        
                        # Calculate the position in the grid
                        grid_x = i % grid_cols
                        grid_y = i // grid_cols
                        
                        # Calculate the section coordinates
                        section_x = grid_x * section_width
                        section_y = grid_y * section_height
                        
                        # Render this environment in its section
                        self.render_environment(env, section_x, section_y, section_width, section_height)
                        
                        # Draw animat ID
                        animat_id_text = self.font.render(f"Animat {start_idx + i + 1}", True, (0, 0, 0))
                        self.screen.blit(animat_id_text, (section_x + 10, section_y + 10))
                    
                    # Draw overall stats
                    gen_text = self.font.render(f"Generation: {gen+1}/{num_generations}, Batch: {batch+1}/{num_batches}", 
                                               True, (0, 0, 0))
                    step_text = self.font.render(f"Step: {steps}/{max_steps}", True, (0, 0, 0))
                    
                    self.screen.blit(gen_text, (self.width - 300, 10))
                    self.screen.blit(step_text, (self.width - 300, 30))
                    
                    # Update the display
                    #pygame.display.flip()
                    
                    # Calculate delay based on speed multiplier (lower = faster)
                    delay = max(1, int(10 / speed_multiplier))
                    pygame.time.delay(delay)
                    
                    # Increase step count proportionally to speed
                    step_increment = max(1, int(speed_multiplier))
                    steps += step_increment
                
                # Calculate fitness for each animat in the batch
                for i, animat in enumerate(animats):
                    fitness = animat.get_fitness()
                    fitnesses.append(fitness)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_genome = batch_genomes[i].copy()
            
            # Ensure fitnesses list has the right length
            while len(fitnesses) < settings.POPULATION_SIZE:
                fitnesses.append(0)
            
            # Log stats
            avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0
            max_fitness = max(fitnesses) if fitnesses else 0
            min_fitness = min(fitnesses) if fitnesses else 0
            
            self.generation_stats['generation'].append(gen)
            self.generation_stats['max_fitness'].append(max_fitness)
            self.generation_stats['avg_fitness'].append(avg_fitness)
            self.generation_stats['min_fitness'].append(min_fitness)
            
            print(f"  Max Fitness: {max_fitness:.2f}")
            print(f"  Avg Fitness: {avg_fitness:.2f}")
            print(f"  Min Fitness: {min_fitness:.2f}")
            
            self.logger.log_generation(gen, fitnesses, best_genome, avg_fitness)
            
            # Update GA's best genome
            if best_genome is not None and (self.ga.best_genome is None or best_fitness > self.ga.best_fitness):
                self.ga.best_genome = best_genome.copy()
                self.ga.best_fitness = best_fitness
            
            # Evolve next generation (except for last generation)
            if gen < num_generations - 1:
                self.ga.evolve_generation()
                
        # Restore headless setting
        self.headless = old_headless
        # self.logger.finalize()
        # Return the best genome
        return self.ga.get_best_genome()
        
    def run_best_animat(self, genome, max_time=60, speed_multiplier=1.0):
        """Run a simulation with the best animat.
        
        Args:
            genome: Genome to use for the animat
            max_time: Maximum simulation time in seconds
            speed_multiplier: Speed multiplier for simulation (higher = faster)
        """
        # Reset the environment
        self.environment = Environment()
        self.environment.initialize_random_environment()
        
        # Create animat with the best genome
        center_pos = (self.environment.width/2, self.environment.height/2)
        animat = Animat(center_pos, genome)
        self.environment.add_entity(animat)
        
        # Run simulation loop
        self.is_running = True
        self.simulation_time = 0
        last_time = time.time()
        
        while self.is_running:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Apply speed multiplier
            dt = dt * speed_multiplier
            
            # Cap dt to avoid large jumps
            dt = min(dt, 0.2 * speed_multiplier)
            
            # Handle events
            if not self.handle_events():
                self.is_running = False
                break
                
            # Update simulation
            self.update(dt)
            
            # Render
            self.render()
            
            # Check if max time reached or animat died
            if max_time is not None and self.simulation_time >= max_time:
                self.is_running = False
                break
                
            if not animat.active:
                print("Animat died")
                self.is_running = False
                break
                
            # Cap frame rate
            if not self.headless:
                self.clock.tick(self.fps)
    
    def plot_stats(self):
        """Plot the evolution statistics."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.generation_stats['generation'], self.generation_stats['max_fitness'], label='Max Fitness')
        plt.plot(self.generation_stats['generation'], self.generation_stats['avg_fitness'], label='Avg Fitness')
        plt.plot(self.generation_stats['generation'], self.generation_stats['min_fitness'], label='Min Fitness')
        
        plt.title('Evolution Progress')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.legend()
        
        plt.savefig('evolution_stats.png')
        print("Evolution statistics saved to evolution_stats.png")
        
        if not self.headless:
            plt.show()
        
    def cleanup(self):
        """Clean up resources."""
        if not self.headless:
            pygame.quit()
            
    def reset(self):
        """Reset the simulation."""
        self.environment = Environment()
        self.simulation_time = 0 