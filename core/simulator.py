"""
Main simulation loop and rendering for the Animat environment.
"""
import time
import pygame
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
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
        
        # Camera tracking for following animats in unbounded environment
        self.camera_offset = [0.0, 0.0]
        
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
        self.capture_markers = []  # For marking food/water capture events [(x, y, type), ...]
        
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
        
        # Update camera to follow animat if needed
        self.update_camera()
        
        # Calculate scale factors to map from environment coordinates to screen coordinates
        # Use object_area_size as the reference for scaling
        scale_x = self.width / self.environment.object_area_size
        scale_y = self.height / self.environment.object_area_size
        
        # Draw trajectory if available
        if len(self.trajectory_to_draw) >= 2:
            scaled_points = []
            for point in self.trajectory_to_draw:
                # Apply camera offset to trajectory points
                screen_px = int((point[0] - self.camera_offset[0]) * scale_x)
                screen_py = int((point[1] - self.camera_offset[1]) * scale_y)
                # Only add points that are visible on screen
                if -50 <= screen_px <= self.width + 50 and -50 <= screen_py <= self.height + 50:
                    scaled_points.append((screen_px, screen_py))
            if len(scaled_points) >= 2:
                pygame.draw.lines(self.screen, (180, 180, 180), False, scaled_points, 2) # Light grey color, 2 pixels thick
        
        # Draw capture markers (food/water capture events)
        for marker in self.capture_markers:
            marker_x, marker_y, marker_type = marker
            # Apply camera offset to marker positions
            screen_mx = int((marker_x - self.camera_offset[0]) * scale_x)
            screen_my = int((marker_y - self.camera_offset[1]) * scale_y)
            
            # Only draw markers that are visible on screen
            if -20 <= screen_mx <= self.width + 20 and -20 <= screen_my <= self.height + 20:
                if marker_type == EntityType.FOOD:
                    # Draw a green X mark for food capture
                    pygame.draw.line(self.screen, (0, 200, 0), (screen_mx - 5, screen_my - 5), (screen_mx + 5, screen_my + 5), 3)
                    pygame.draw.line(self.screen, (0, 200, 0), (screen_mx - 5, screen_my + 5), (screen_mx + 5, screen_my - 5), 3)
                elif marker_type == EntityType.WATER:
                    # Draw a blue X mark for water capture
                    pygame.draw.line(self.screen, (0, 0, 200), (screen_mx - 5, screen_my - 5), (screen_mx + 5, screen_my + 5), 3)
                    pygame.draw.line(self.screen, (0, 0, 200), (screen_mx - 5, screen_my + 5), (screen_mx + 5, screen_my - 5), 3)
        
        # Render entities
        for entity in self.environment.entities:
            if not entity.active:
                continue
                
            color = self.colors.get(entity.type, (0, 0, 0))
            
            # Apply camera offset to entity positions
            screen_x = int((entity.position[0] - self.camera_offset[0]) * scale_x)
            screen_y = int((entity.position[1] - self.camera_offset[1]) * scale_y)
            screen_radius = int(entity.radius * scale_x)  # Use scale_x for consistent scaling
            
            # Only render entities that are visible on screen (with some margin)
            margin = 100
            if (-margin <= screen_x <= self.width + margin and 
                -margin <= screen_y <= self.height + margin):
                
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
        
        # Display camera position info
        camera_text = self.font.render(f"Camera: ({self.camera_offset[0]:.1f}, {self.camera_offset[1]:.1f})", True, (0, 0, 0))
        self.screen.blit(camera_text, (10, 70))
        
        # Display speed for each animat
        y_offset = 90
        for entity in self.environment.entities:
            if entity.type == EntityType.ANIMAT:
                # Calculate the actual speed (magnitude of the velocity)
                forward_speed = entity.get_forward_speed()
                speed_text = self.font.render(f"Animat {id(entity) % 1000} Speed: {forward_speed:.2f}", True, (0, 0, 0))
                self.screen.blit(speed_text, (10, y_offset))
                y_offset += 20
                
                # Display position
                pos_text = self.font.render(f"Position: ({entity.position[0]:.1f}, {entity.position[1]:.1f})", True, (0, 0, 0))
                self.screen.blit(pos_text, (10, y_offset))
                y_offset += 20
                
                # Display distance from center
                center_pos = self.environment.object_area_size / 2
                distance_from_center = np.linalg.norm(entity.position - np.array([center_pos, center_pos]))
                dist_text = self.font.render(f"Distance from center: {distance_from_center:.1f}", True, (0, 0, 0))
                self.screen.blit(dist_text, (10, y_offset))
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
        # Calculate camera offset for this environment
        camera_offset = [0.0, 0.0]
        
        # Find the animat in this environment
        animat = None
        for entity in environment.entities:
            if entity.type == EntityType.ANIMAT and entity.active:
                animat = entity
                break
        
        if animat:
            # Check if animat is outside the original object area
            object_area_boundary = environment.object_area_size
            animat_pos = animat.position
            
            # If animat is outside the object area, center camera on animat
            if (animat_pos[0] < 0 or animat_pos[0] > object_area_boundary or 
                animat_pos[1] < 0 or animat_pos[1] > object_area_boundary):
                
                camera_offset[0] = animat_pos[0] - environment.object_area_size / 2
                camera_offset[1] = animat_pos[1] - environment.object_area_size / 2
        
        # Calculate scale factors to map from environment coordinates to screen coordinates
        # Use object_area_size as the reference for scaling
        scale_x = width / environment.object_area_size
        scale_y = height / environment.object_area_size
        
        # Render entities
        for entity in environment.entities:
            if not entity.active:
                continue
                
            color = self.colors.get(entity.type, (0, 0, 0))
            
            # Apply camera offset to entity positions
            screen_x = int(x + (entity.position[0] - camera_offset[0]) * scale_x)
            screen_y = int(y + (entity.position[1] - camera_offset[1]) * scale_y)
            screen_radius = int(entity.radius * scale_x)  # Use scale_x for consistent scaling
            
            # Only render entities that are visible within this section
            if (x <= screen_x <= x + width and y <= screen_y <= y + height):
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
                    
                    # Create animat with the genome at a random position
                    spawn_pos = env.get_random_spawn_position()
                    animat = Animat(spawn_pos, genome)
                    env.add_entity(animat)
                    
                    environments.append(env)
                    animats.append(animat)
                
                # Run simulation loop with visualization for this batch
                self.is_running = True
                self.simulation_time = 0 # This is for display, not direct loop control here
                
                max_total_sim_steps = settings.ANIMAT_MAX_LIFESPAN # Total 0.1s ticks for an animat
                accumulated_sim_steps = 0 # Tracks total 0.1s ticks simulated for this batch
                SIMULATION_TIMESTEP = 0.1 # Fixed duration of one simulation update

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
        
    def save_screenshot(self, filename_prefix="screenshot", reason="death"):
        """Save a screenshot of the current pygame screen.
        
        Args:
            filename_prefix: Prefix for the filename
            reason: Reason for taking the screenshot (e.g., "death", "end")
        """
        if self.headless:
            return
            
        # Create screenshots directory if it doesn't exist
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{filename_prefix}_{reason}_{timestamp}.png"
        filepath = os.path.join(screenshots_dir, filename)
        
        # Save the screenshot
        pygame.image.save(self.screen, filepath)
        print(f"Screenshot saved: {filepath}")
        
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
        
        # Create animat with the best genome at a random position
        spawn_pos = self.environment.get_random_spawn_position()
        animat = Animat(spawn_pos, genome)
        self.environment.add_entity(animat)
        
        # Run simulation loop
        self.is_running = True
        self.simulation_time = 0
        last_time = time.time()
        self.trajectory_to_draw = [] # Clear previous trajectory
        self.capture_markers = []  # Clear previous capture markers
        
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
            
            # Store previous battery levels to detect capture events
            prev_battery1 = animat.batteries[0]
            prev_battery2 = animat.batteries[1]
                
            # Update simulation
            self.update(dt)
            
            # Check for food/water capture events by detecting battery level changes
            if animat.active and (animat.batteries[0] > prev_battery1 or animat.batteries[1] > prev_battery2):
                if animat.batteries[0] > prev_battery1:
                    # Food was captured (battery 1 recharged)
                    self.capture_markers.append((animat.position[0], animat.position[1], EntityType.FOOD))
                if animat.batteries[1] > prev_battery2:
                    # Water was captured (battery 2 recharged)
                    self.capture_markers.append((animat.position[0], animat.position[1], EntityType.WATER))
            
            # Add current animat position to trajectory (no length limit now)
            if animat.active:
                self.trajectory_to_draw.append(tuple(animat.position))

            # Render
            self.render()
            
            # Check if max time reached or animat died
            if max_time is not None and self.simulation_time >= max_time:
                print(f"Simulation ended after {max_time} seconds")
                self.save_screenshot("best_animat", "timeout")
                self.is_running = False
                break
                
            if not animat.active:
                print(f"Animat died after {self.simulation_time:.1f} seconds")
                print(f"Final battery levels: Battery 1: {animat.batteries[0]:.1f}, Battery 2: {animat.batteries[1]:.1f}")
                print(f"Final fitness: {animat.get_fitness():.3f}")
                print(f"Total food/water captures: {len(self.capture_markers)}")
                self.save_screenshot("best_animat", "death")
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
        
        # Create animat with Seth's specific genome at a random position
        spawn_pos = self.environment.get_random_spawn_position()
        animat = Animat(spawn_pos, "seth")  # Special string to trigger seth genome
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
        self.capture_markers = []  # Clear previous capture markers
        
        # Use fixed timestep like in the evolution simulation
        SIMULATION_TIMESTEP = 1  # Fixed 0.1 second timestep
        step_count = 0
        max_steps = int(max_time / SIMULATION_TIMESTEP)
        
        while self.is_running and step_count < max_steps:
            # Handle events
            if not self.handle_events():
                self.is_running = False
                break
            
            # Store previous battery levels to detect capture events
            prev_battery1 = animat.batteries[0]
            prev_battery2 = animat.batteries[1]
                
            # Update simulation with fixed timestep
            self.environment.update(SIMULATION_TIMESTEP)
            self.simulation_time += SIMULATION_TIMESTEP
            step_count += 1
            
            # Check for food/water capture events by detecting battery level changes
            if animat.active and (animat.batteries[0] > prev_battery1 or animat.batteries[1] > prev_battery2):
                if animat.batteries[0] > prev_battery1:
                    # Food was captured (battery 1 recharged)
                    self.capture_markers.append((animat.position[0], animat.position[1], EntityType.FOOD))
                if animat.batteries[1] > prev_battery2:
                    # Water was captured (battery 2 recharged)
                    self.capture_markers.append((animat.position[0], animat.position[1], EntityType.WATER))
            
            # Add current animat position to trajectory (no length limit now)
            if animat.active:
                self.trajectory_to_draw.append(tuple(animat.position))

            # Render
            self.render()
            
            # Check if animat died
            if not animat.active:
                print(f"Seth's animat died after {self.simulation_time:.1f} seconds")
                print(f"Final battery levels: Battery 1: {animat.batteries[0]:.1f}, Battery 2: {animat.batteries[1]:.1f}")
                print(f"Final fitness: {animat.get_fitness():.3f}")
                self.save_screenshot("seth_animat", "death")
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
            self.save_screenshot("seth_animat", "survived")

    def update_camera(self):
        """Update the camera position to follow the animat if it goes outside the object area."""
        # Find the first active animat
        active_animat = None
        for entity in self.environment.entities:
            if entity.type == EntityType.ANIMAT and entity.active:
                active_animat = entity
                break
        
        if not active_animat:
            return
        
        # Check if animat is outside the original object area
        object_area_boundary = self.environment.object_area_size
        animat_pos = active_animat.position
        
        # Calculate desired camera position to center the animat
        desired_camera_x = animat_pos[0] - self.environment.object_area_size / 2
        desired_camera_y = animat_pos[1] - self.environment.object_area_size / 2
        
        # If animat is outside the object area, start following
        if (animat_pos[0] < 0 or animat_pos[0] > object_area_boundary or 
            animat_pos[1] < 0 or animat_pos[1] > object_area_boundary):
            
            # Smooth camera following
            camera_speed = 0.1  # Adjust for smoother/faster following
            self.camera_offset[0] += (desired_camera_x - self.camera_offset[0]) * camera_speed
            self.camera_offset[1] += (desired_camera_y - self.camera_offset[1]) * camera_speed
        else:
            # If animat is back in the object area, gradually return camera to origin
            camera_speed = 0.05
            self.camera_offset[0] += (0 - self.camera_offset[0]) * camera_speed
            self.camera_offset[1] += (0 - self.camera_offset[1]) * camera_speed

    def follow_animat(self, animat):
        """Follow a specific animat."""
        self.followed_animat = animat

    def unfollow_animat(self):
        """Unfollow the current animat."""
        self.followed_animat = None 