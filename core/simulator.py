"""
Main simulation loop and PyGame rendering for the Braitenberg-inspired environment.
"""
import time
import pygame
import sys
import random
from config import settings
from core.environment import Environment, EntityType
from utils.logger import Logger

class Simulator:
    """
    The main simulator class that handles:
    - Initializing the environment
    - Running the simulation loop
    - Rendering the simulation (using pygame)
    - Tick updating of all entities
    """
    def __init__(self, width=settings.WINDOW_WIDTH, height=settings.WINDOW_HEIGHT, 
                 headless=settings.HEADLESS_MODE):
        self.width = width
        self.height = height
        self.headless = headless
        
        # Initialize the environment
        self.environment = Environment(width, height)
        
        # Initialize logging
        self.logger = Logger()
        self.logger.set_settings(vars(settings))
        
        # Initialize pygame if not headless
        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(settings.WINDOW_TITLE)
            self.clock = pygame.time.Clock()
            
            # Load or create assets
            self.font = pygame.font.SysFont('Arial', 12)
            self.colors = {
                EntityType.OBSTACLE: (100, 100, 100),      # Gray
                EntityType.LIGHT_SOURCE: (255, 255, 0),    # Yellow
                EntityType.FOOD_SOURCE: (0, 255, 0),       # Green
                EntityType.AGENT: (255, 0, 0),             # Red
            }
            
        self.is_running = False
        self.tick_counter = 0
        self.simulation_time = 0
        self.fps = settings.FPS
        
    def initialize_environment(self):
        """Initialize the environment with entities."""
        self.environment.initialize_random_environment(
            settings.OBSTACLE_COUNT,
            settings.LIGHT_SOURCE_COUNT,
            settings.FOOD_SOURCE_COUNT
        )
        
    def add_agents(self, agents):
        """Add agents to the environment."""
        for agent in agents:
            self.environment.add_entity(agent)
            
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
        """Update the simulation by one step."""
        # Update environment
        self.environment.update(dt)
        
        # Log data
        if settings.LOGGING_ENABLED and self.tick_counter % 10 == 0:
            for agent in self.environment.agents:
                self.logger.log_stress(agent.id, agent.stress, agent.position)
                
        self.tick_counter += 1
        self.simulation_time += dt
        
    def render(self):
        """Render the current state of the simulation."""
        if self.headless:
            return
            
        # Clear the screen
        self.screen.fill((255, 255, 255))
        
        # Render entities
        for entity in self.environment.entities:
            if not entity.active:
                continue
                
            color = self.colors.get(entity.type, (0, 0, 0))
            
            # Adjust color based on entity properties
            if entity.type == EntityType.LIGHT_SOURCE:
                # Make color brighter based on intensity
                intensity_factor = entity.intensity / 150.0
                color = (min(255, int(color[0] * intensity_factor)),
                        min(255, int(color[1] * intensity_factor)),
                        color[2])
            elif entity.type == EntityType.FOOD_SOURCE:
                # Adjust green based on energy level
                energy_factor = entity.energy / entity.max_energy
                color = (color[0], 
                        int(color[1] * energy_factor),
                        color[2])
            elif entity.type == EntityType.AGENT:
                # Adjust red based on stress level
                stress_factor = 1.0 - (entity.stress / settings.MAX_STRESS)
                color = (color[0], 
                        int(color[1] + 150 * stress_factor),
                        int(color[2] + 150 * stress_factor))
                
            # Draw entity
            pygame.draw.circle(self.screen, color, 
                              (int(entity.position[0]), int(entity.position[1])), 
                              int(entity.radius))
            
            # Draw entity ID
            text = self.font.render(str(entity.id), True, (0, 0, 0))
            self.screen.blit(text, (entity.position[0] - 5, entity.position[1] - 5))
            
            # Draw agent direction if it's an agent
            if entity.type == EntityType.AGENT:
                # Draw a line showing direction
                direction_vector = (
                    entity.position[0] + entity.radius * 1.5 * entity.direction[0],
                    entity.position[1] + entity.radius * 1.5 * entity.direction[1]
                )
                pygame.draw.line(self.screen, (0, 0, 0), 
                                entity.position, direction_vector, 2)
        
        # Draw performance stats
        fps_text = self.font.render(f"FPS: {int(self.clock.get_fps())}", True, (0, 0, 0))
        time_text = self.font.render(f"Time: {self.simulation_time:.1f}s", True, (0, 0, 0))
        agents_text = self.font.render(f"Agents: {len(self.environment.agents)}", True, (0, 0, 0))
        
        self.screen.blit(fps_text, (10, 10))
        self.screen.blit(time_text, (10, 30))
        self.screen.blit(agents_text, (10, 50))
        
        # Update the display
        pygame.display.flip()
        
    def run(self, max_time=None):
        """Run the simulation loop."""
        self.is_running = True
        last_time = time.time()
        
        while self.is_running:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Cap dt to avoid large jumps
            dt = min(dt, 0.1)
            
            # Handle events
            if not self.handle_events():
                self.is_running = False
                break
                
            # Update simulation
            self.update(dt)
            
            # Render
            self.render()
            
            # Check if max time reached
            if max_time is not None and self.simulation_time >= max_time:
                self.is_running = False
                break
                
            # Cap frame rate
            if not self.headless:
                self.clock.tick(self.fps)
        
        # Clean up
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources."""
        self.logger.finalize()
        
        if not self.headless:
            pygame.quit()
            
    def reset(self):
        """Reset the simulation."""
        self.environment = Environment(self.width, self.height)
        self.initialize_environment()
        self.tick_counter = 0
        self.simulation_time = 0 