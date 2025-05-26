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
            'generation': []
            ,'max_fitness': []
            ,'avg_fitness': []
            ,'min_fitness': []
            ,'best_of_gen':[]
            
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
                
                pygame.draw.rect(self.screen, self.colors[EntityType.FOOD], 
                                (screen_x - 10, screen_y - entity.radius - 10, battery1_width, 3))
                pygame.draw.rect(self.screen, self.colors[EntityType.WATER], 
                                (screen_x - 10, screen_y - entity.radius - 5, battery2_width, 3))
                
                # Display battery percentages next to the bars
                battery1_percent = int((entity.batteries[0] / settings.BATTERY_MAX) * 100)
                battery2_percent = int((entity.batteries[1] / settings.BATTERY_MAX) * 100)
                battery_text = self.font.render(f"L:{battery1_percent}% R:{battery2_percent}%", True, (0, 0, 0))
                self.screen.blit(battery_text, (screen_x + 10, screen_y - entity.radius - 10))
        
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
                
                # Draw speed for this animat
                forward_speed = entity.get_forward_speed()
                speed_text = self.font.render(f"Speed: {forward_speed:.2f}", True, (0, 0, 0))
                self.screen.blit(speed_text, (screen_x - 10, screen_y + entity.radius + 5))
        
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
                
                # print(f"  Processing batch {batch+1}/{num_batches} (animats {start_idx+1}-{end_idx})")
                
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

                #Collect fitness after this batch 
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
            best_of_gen = max(fitnesses) if fitnesses else 0

            self.generation_stats['generation'].append(gen)
            self.generation_stats['max_fitness'].append(best_fitness)
            self.generation_stats['avg_fitness'].append(avg_fit)
            self.generation_stats['min_fitness'].append(min(fitnesses))
            self.generation_stats['best_of_gen'].append(best_of_gen)

            # print(f"  Max Fitness: {best_fitness:.2f}")
            print(f"  Avg Fitness: {avg_fit:.2f}")
            print(f"  Min Fitness: {min(fitnesses):.2f}")
            print(f"  Best of gen: {best_of_gen:.2f}")
            

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
        
        print("evo1")
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
                    #pygame.display.flip()
                    
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
            
            avg_fit = sum(fitnesses) / len(fitnesses)
            best_of_gen = max(fitnesses) if fitnesses else 0
            max_fitness = max(fitnesses) if fitnesses else 0
            min_fitness = min(fitnesses) if fitnesses else 0
            
            self.generation_stats['generation'].append(gen)
            self.generation_stats['max_fitness'].append(best_fitness)
            self.generation_stats['avg_fitness'].append(avg_fit)
            self.generation_stats['min_fitness'].append(min(fitnesses))
            self.generation_stats['best_of_gen'].append(best_of_gen)

            # print(f"  Max Fitness: {best_fitness:.2f}")
            print(f"  Avg Fitness: {avg_fit:.2f}")
            print(f"  Min Fitness: {min(fitnesses):.2f}")
            print(f"  Best of gen: {best_of_gen:.2f}")
            
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
        gens = self.generation_stats['generation']
        best_of_gens = np.array(self.generation_stats['best_of_gen'])

        if len(best_of_gens) == 0:          # 防止再次为空
            print("No data to plot — best_of_gen is empty.")
            return

        # running-max 做阶梯
        best_so_far = np.maximum.accumulate(best_of_gens)

        plt.figure(figsize=(10, 6))
        plt.step(gens, best_so_far, where='post', linewidth=2,
                label='Best so-far (阶梯)')
        plt.plot(gens, best_of_gens, '--', alpha=.6,
                label='Best / generation')

        plt.plot(gens, self.generation_stats['avg_fitness'], label='Avg Fitness')
        plt.plot(gens, self.generation_stats['min_fitness'], label='Min Fitness')

        plt.title('Evolution Progress')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True, alpha=.3)
        plt.legend()
        plt.tight_layout()
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