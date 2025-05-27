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
        self.trajectory_to_draw = [] # For drawing agent trajectory
        
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
        
        # Draw trajectory if available
        if len(self.trajectory_to_draw) >= 2:
            scaled_points = []
            for point in self.trajectory_to_draw:
                screen_px = int(point[0] * scale_x)
                screen_py = int(point[1] * scale_y)
                scaled_points.append((screen_px, screen_py))
            pygame.draw.lines(self.screen, (180, 180, 180), False, scaled_points, 2) # Light grey color, 2 pixels thick
        
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
                
                pygame.draw.rect(self.screen, self.colors[EntityType.FOOD], 
                                (screen_x - 10, screen_y - entity.radius - 10, battery1_width, 3))
                pygame.draw.rect(self.screen, self.colors[EntityType.WATER], 
                                (screen_x - 10, screen_y - entity.radius - 5, battery2_width, 3))
                
                # Display battery percentages next to the bars
                battery1_percent = int((entity.batteries[0] / settings.BATTERY_MAX) * 100)
                battery2_percent = int((entity.batteries[1] / settings.BATTERY_MAX) * 100)
                battery_text = self.font.render(f"L:{battery1_percent}% R:{battery2_percent}%", True, (0, 0, 0))
                self.screen.blit(battery_text, (screen_x + 10, screen_y - entity.radius - 10))
                
                # Display animat speed directly below the animat in the main render view
                forward_speed = entity.get_forward_speed()
                speed_text_surface = self.font.render(f"Speed: {forward_speed:.2f}", True, (0, 0, 0))
                text_width = speed_text_surface.get_width()
                self.screen.blit(speed_text_surface, (screen_x - text_width // 2, screen_y + screen_radius + 5))
        
        # Draw performance stats
        fps_text = self.font.render(f"FPS: {int(self.clock.get_fps())}", True, (0, 0, 0))
        time_text = self.font.render(f"Time: {self.simulation_time:.1f}s", True, (0, 0, 0))
        gen_text = self.font.render(f"Generation: {self.generation}/{settings.NUM_GENERATIONS}", True, (0, 0, 0))
        
        # Display speed for each animat
        y_offset = 70
        for entity in self.environment.entities:
            if entity.type == EntityType.ANIMAT:
                # Calculate the actual speed (magnitude of the velocity)
                forward_speed = entity.get_forward_speed()
                speed_text = self.font.render(f"Animat {id(entity) % 1000} Speed: {forward_speed:.2f}", True, (0, 0, 0))
                self.screen.blit(speed_text, (10, y_offset))
                y_offset += 20
                
                # Display battery percentages in the panel
                battery1_percent = int((entity.batteries[0] / settings.BATTERY_MAX) * 100)
                battery2_percent = int((entity.batteries[1] / settings.BATTERY_MAX) * 100)
                battery_text = self.font.render(f"Batteries: Left: {battery1_percent}% Right: {battery2_percent}%", True, (0, 0, 0))
                self.screen.blit(battery_text, (10, y_offset))
                y_offset += 20
        
        if self.ga.best_fitness > 0:
            fitness_text = self.font.render(f"Best Fitness: {self.ga.best_fitness:.1f}", True, (0, 0, 0))
            self.screen.blit(fitness_text, (10, y_offset))
        
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
                
                pygame.draw.rect(self.screen, self.colors[EntityType.FOOD], 
                                (screen_x - 10, screen_y - entity.radius - 10, battery1_width, 3))
                pygame.draw.rect(self.screen, self.colors[EntityType.WATER], 
                                (screen_x - 10, screen_y - entity.radius - 5, battery2_width, 3))
                
                # Display battery percentages next to the bars
                battery1_percent = int((entity.batteries[0] / settings.BATTERY_MAX) * 100)
                battery2_percent = int((entity.batteries[1] / settings.BATTERY_MAX) * 100)
                battery_text = self.font.render(f"L:{battery1_percent}% R:{battery2_percent}%", True, (0, 0, 0))
                self.screen.blit(battery_text, (screen_x + 10, screen_y - entity.radius - 10))
                
                # Display animat speed directly below the animat in the main render view
                forward_speed = entity.get_forward_speed()
                speed_text_surface = self.font.render(f"Speed: {forward_speed:.2f}", True, (0, 0, 0))
                text_width = speed_text_surface.get_width()
                self.screen.blit(speed_text_surface, (screen_x - text_width // 2, screen_y + screen_radius + 5))
        
        # Draw section border for clarity
        pygame.draw.rect(self.screen, (200, 200, 200), (x, y, width, height), 1)
        
    def track_animat_deaths(self, animats, active_animats, generation):
        """Track animat deaths and check if 50% of population has died.
        
        Args:
            animats: List of animat objects
            active_animats: List of boolean values indicating if each animat is active
            generation: Current generation number
            
        Returns:
            bool: True if 50% or more of the population has died
        """
        total_population = len(animats)
        dead_count = sum(1 for animat in animats if not animat.active)
        death_percentage = (dead_count / total_population) * 100
        
        if death_percentage >= settings.SIMULATION_END_PERCENTAGE:
            print(f"  Generation {generation + 1}: {dead_count}/{total_population} animats died ({death_percentage:.1f}%) - Terminating generation early")
            return True
        
        return False
        
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
            
            # Process animats in parallel batches (assume batch_size = population_size)
            batch_size = settings.POPULATION_SIZE  # Force batch_size to equal population_size
            num_batches = 1  # Only one batch since batch_size = population_size
            
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
                self.simulation_time = 0 # This is for display, not direct loop control here
                
                max_total_sim_steps = settings.ANIMAT_MAX_LIFESPAN # Total 0.1s ticks for an animat
                accumulated_sim_steps = 0 # Tracks total 0.1s ticks simulated for this batch
                SIMULATION_TIMESTEP = 1 # Fixed duration of one simulation update

                # Determine num_sim_steps_this_frame and visual_delay_ms
                num_sim_steps_this_frame = 1
                visual_delay_ms = 1 # Default to minimal for very high speeds
                effective_sm = max(1.0, speed_multiplier) # Ensure speed_multiplier is at least 1

                if effective_sm <= 100.0: # Threshold for rendering every sim step
                    # Visual delay aims to make 1 sim step (0.1s) take (0.1s / effective_sm) real time
                    visual_delay_ms = max(1, int((SIMULATION_TIMESTEP * 1000) / effective_sm))
                    num_sim_steps_this_frame = 1
                else: # speed_multiplier > 100.0, run multiple sim steps per rendered frame
                    # Number of sim steps per rendered frame increases with speed_multiplier
                    num_sim_steps_this_frame = max(1, int(effective_sm / 100.0)) 
                    visual_delay_ms = 1 # Minimal delay, let computation be the limit.
                
                # Display which batch is being simulated
                title_text = f"{settings.WINDOW_TITLE} - Gen {gen+1}/{num_generations}, Batch {batch+1}/{num_batches}"
                pygame.display.set_caption(title_text)
                
                # Track which animats are still active
                active_animats = [True] * len(animats)
                any_active = True # Initially true if there are animats
                
                # Outer loop for rendered frames; continues as long as simulation steps are left
                while self.is_running and any_active and accumulated_sim_steps < max_total_sim_steps:
                    # Handle Pygame events (once per rendered frame)
                    if not self.handle_events():
                        self.is_running = False
                        break # Exit outer while loop

                    # Perform a batch of simulation steps before rendering
                    for _ in range(num_sim_steps_this_frame):
                        if not any_active or accumulated_sim_steps >= max_total_sim_steps:
                            break # Stop sim steps if all died or max steps reached for the batch

                        current_batch_still_active = False # Tracks if any animat is active in *this* sim step
                        for i, (env_obj, animat_obj) in enumerate(zip(environments, animats)):
                            if active_animats[i]: # If this animat was considered active for this sim step
                                env_obj.update(SIMULATION_TIMESTEP) # Update with fixed 0.1s dt
                                if not animat_obj.active: # Check if it became inactive
                                    active_animats[i] = False 
                                else:
                                    current_batch_still_active = True # At least one animat is still running
                        
                        any_active = current_batch_still_active # Update overall status for the batch
                        accumulated_sim_steps += 1
                        
                        # Check if 50% of animats have died (assuming batch_size = population_size)
                        if self.track_animat_deaths(animats, active_animats, gen):
                            any_active = False  # Force termination of the generation
                            break
                        
                        if not any_active: # if all animats died in this simulation step, break from inner (sim_steps) loop
                            break
                    
                    # --- Rendering part (happens once after a number of simulation steps) ---
                    self.screen.fill((255, 255, 255))
                    
                    # Render all animat environments based on their latest state
                    for i_render, (env_render, animat_render_obj) in enumerate(zip(environments, animats)):
                        # Calculate the grid layout dimensions
                        grid_cols = int(np.ceil(np.sqrt(batch_size)))
                        grid_rows = int(np.ceil(batch_size / grid_cols))
                        
                        # Calculate the section dimensions
                        section_width = self.width // grid_cols
                        section_height = self.height // grid_rows
                        
                        # Calculate the position in the grid
                        grid_x = i_render % grid_cols
                        grid_y = i_render // grid_cols
                        
                        # Calculate the section coordinates
                        section_x = grid_x * section_width
                        section_y = grid_y * section_height
                        
                        # Render this environment in its section
                        self.render_environment(env_render, section_x, section_y, section_width, section_height)
                        
                        # Draw animat ID
                        animat_id_text = self.font.render(f"Animat {start_idx + i_render + 1}", True, (0, 0, 0))
                        self.screen.blit(animat_id_text, (section_x + 10, section_y + 10))
                        
                        # Display battery percentages
                        battery1_percent = int((animat_render_obj.batteries[0] / settings.BATTERY_MAX) * 100)
                        battery2_percent = int((animat_render_obj.batteries[1] / settings.BATTERY_MAX) * 100)
                        battery_text = self.font.render(f"L: {battery1_percent}% R: {battery2_percent}%", True, (0, 0, 0))
                        self.screen.blit(battery_text, (section_x + 10, section_y + 30))
                    
                    # Draw overall stats
                    gen_display_text = self.font.render(f"Generation: {gen+1}/{num_generations}, Batch: {batch+1}/{num_batches}", 
                                                       True, (0, 0, 0))
                    # Display the accumulated_sim_steps
                    step_display_text = self.font.render(f"Step: {accumulated_sim_steps}/{max_total_sim_steps}", True, (0, 0, 0))
                    
                    self.screen.blit(gen_display_text, (self.width - 300, 10))
                    self.screen.blit(step_display_text, (self.width - 300, 30))
                    
                    # Update the display
                    pygame.display.flip()
                    
                    # Visual delay
                    pygame.time.delay(visual_delay_ms)

                    if not any_active: # If, after the sim steps and rendering, no animats are active in batch
                        break # End visualization for this batch early
                
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
        self.trajectory_to_draw = [] # Clear previous trajectory
        MAX_TRAJECTORY_POINTS = 500 # Max points to store for trajectory
        
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
            
            # Add current animat position to trajectory
            if animat.active:
                if len(self.trajectory_to_draw) >= MAX_TRAJECTORY_POINTS:
                    self.trajectory_to_draw.pop(0) # Remove oldest point
                self.trajectory_to_draw.append(tuple(animat.position))

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

    def run_seth_model(self, max_time=60, speed_multiplier=1.0):
        """Run a simulation with Seth's specific animat model from the paper.
        
        Args:
            max_time: Maximum simulation time in seconds
            speed_multiplier: Speed multiplier for simulation (higher = faster)
        """
        print("Running Seth's model with specific link configurations from the paper...")
        
        # Reset the environment
        self.environment = Environment()
        self.environment.initialize_random_environment()
        
        # Create animat with Seth's specific genome
        center_pos = (self.environment.width/2, self.environment.height/2)
        animat = Animat(center_pos, "seth")  # Special string to trigger seth genome
        self.environment.add_entity(animat)
        
        print("Seth's animat created with the following link configuration:")
        print("- Links 1-3: Food sensors (mix of battery 1 & 2)")
        print("- Links 4-6: Water sensors (mix of battery 1 & 2)")  
        print("- Links 7-9: Trap sensors (mix of battery 1 & 2)")
        print("- Each sensor type has varied transfer functions as shown in the paper")
        
        # Run simulation loop
        self.is_running = True
        self.simulation_time = 0
        last_time = time.time()
        self.trajectory_to_draw = []  # Clear previous trajectory
        MAX_TRAJECTORY_POINTS = 500
        
        # Use fixed timestep like in the evolution simulation
        SIMULATION_TIMESTEP = 0.1  # Fixed 0.1 second timestep
        step_count = 0
        max_steps = int(max_time / SIMULATION_TIMESTEP)
        
        while self.is_running and step_count < max_steps:
            # Handle events
            if not self.handle_events():
                self.is_running = False
                break
                
            # Update simulation with fixed timestep
            self.environment.update(SIMULATION_TIMESTEP)
            self.simulation_time += SIMULATION_TIMESTEP
            step_count += 1
            
            # Add current animat position to trajectory
            if animat.active:
                if len(self.trajectory_to_draw) >= MAX_TRAJECTORY_POINTS:
                    self.trajectory_to_draw.pop(0)  # Remove oldest point
                self.trajectory_to_draw.append(tuple(animat.position))

            # Render
            self.render()
            
            # Check if animat died
            if not animat.active:
                print(f"Seth's animat died after {self.simulation_time:.1f} seconds")
                print(f"Final battery levels: Battery 1: {animat.batteries[0]:.1f}, Battery 2: {animat.batteries[1]:.1f}")
                print(f"Final fitness: {animat.get_fitness():.3f}")
                self.is_running = False
                break
                
            # Cap frame rate
            if not self.headless:
                self.clock.tick(self.fps)
                
        # Print final statistics
        if animat.active:
            print(f"Seth's animat survived the full simulation!")
            print(f"Final battery levels: Battery 1: {animat.batteries[0]:.1f}, Battery 2: {animat.batteries[1]:.1f}")
            print(f"Final fitness: {animat.get_fitness():.3f}") 