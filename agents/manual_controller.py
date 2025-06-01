"""
Manual controller for animats with automatic survival behavior when not manually controlled.
Allows keyboard control of an animat to manually navigate and survive in the environment.
Enhanced with smooth movement, proper respawning, collision prevention, and optimized UI rendering.
"""
import pygame
import numpy as np
import time
import random
from config import settings
from agents.base_agent import Animat
from core.environment import EntityType

class ManualController:
    """Manual controller for animat simulation with automatic survival AI."""
    
    def __init__(self, simulator):
        """Initialize the manual controller.
        
        Args:
            simulator: The main simulator instance
        """
        self.simulator = simulator
        self.animat = None
        self.is_running = False
        self.move_speed = 150.0  # Pixels per second for smooth movement
        self.turn_speed = 3.0  # Radians per second for smooth turning
        self.manual_mode = False  # Whether player is actively controlling
        self.last_manual_input = 0  # Time since last manual input
        self.manual_timeout = 1.0  # Seconds before switching to auto mode
        
        # Respawn management with entity property tracking
        self.food_respawn_timer = 0.0
        self.water_respawn_timer = 0.0
        self.respawn_delay = 2.0  # Seconds before respawning consumed resources
        self.pending_food_respawns = []  # Store entity info for proper respawn
        self.pending_water_respawns = []  # Store entity info for proper respawn
        
        # Smooth movement interpolation
        self.target_position = None
        self.movement_smoothing = 0.8  # Lower = smoother but slower response
        
        # UI caching to prevent flickering
        self.text_cache = {}
        self.ui_surfaces = {}
        
        # Force non-headless mode for manual control
        self.simulator.headless = False
        
        # Initialize pygame properly to avoid conflicts
        self._initialize_display()
        
        # Create UI surfaces and cache text after display initialization
        self._create_ui_surfaces()
    
    def _initialize_display(self):
        """Initialize pygame display without conflicts."""
        if not pygame.get_init():
            pygame.init()
        
        # Only create screen if it doesn't exist or is incompatible
        if not hasattr(self.simulator, 'screen') or self.simulator.screen is None:
            self.simulator.screen = pygame.display.set_mode(
                (self.simulator.width, self.simulator.height), 
                pygame.DOUBLEBUF | pygame.HWSURFACE
            )
            pygame.display.set_caption(settings.WINDOW_TITLE + " - Manual Mode")
        
        if not hasattr(self.simulator, 'clock') or self.simulator.clock is None:
            self.simulator.clock = pygame.time.Clock()
            
        # Load font and colors if not already loaded
        if not hasattr(self.simulator, 'font'):
            self.simulator.font = pygame.font.SysFont('Arial', 14)
        if not hasattr(self.simulator, 'colors'):
            self.simulator.colors = {
                EntityType.FOOD: (0, 255, 0),          # Green
                EntityType.WATER: (0, 0, 255),         # Blue
                EntityType.TRAP: (255, 0, 0),          # Red
                EntityType.ANIMAT: (255, 255, 0),      # Yellow
            }
    
    def _create_ui_surfaces(self):
        """Pre-create UI surfaces to prevent flickering."""
        # Create base overlays
        self.ui_surfaces['instruction_overlay'] = pygame.Surface((400, 220), pygame.SRCALPHA)
        self.ui_surfaces['instruction_overlay'].fill((255, 255, 255, 180))
        
        self.ui_surfaces['status_overlay'] = pygame.Surface((400, 140), pygame.SRCALPHA)
        self.ui_surfaces['status_overlay'].fill((255, 255, 255, 180))
        
        self.ui_surfaces['game_over_overlay'] = pygame.Surface((600, 100), pygame.SRCALPHA)
        self.ui_surfaces['game_over_overlay'].fill((255, 255, 255, 200))
        
        # Pre-render static text
        self._cache_static_text()
    
    def _cache_static_text(self):
        """Cache static text that doesn't change."""
        static_texts = [
            ("wasd_instruction", "WASD or Arrow Keys - Manual Control", self.simulator.font, (0, 0, 0)),
            ("auto_instruction", "Auto mode when idle for 1 second", self.simulator.font, (0, 0, 0)),
            ("food_instruction", "Green circles - Food (restores left battery)", self.simulator.font, (0, 0, 0)),
            ("water_instruction", "Blue circles - Water (restores right battery)", self.simulator.font, (0, 0, 0)),
            ("trap_instruction", "Red circles - Traps (AVOID!)", self.simulator.font, (0, 0, 0)),
            ("controls_instruction", "ESC - Quit | R - Restart (when dead)", self.simulator.font, (0, 0, 0)),
            ("status_header", "=== ANIMAT STATUS ===", pygame.font.SysFont('Arial', 16, bold=True), (0, 0, 150)),
            ("game_over", "GAME OVER", pygame.font.SysFont('Arial', 24, bold=True), (255, 0, 0)),
            ("restart_prompt", "Press R to restart or ESC to quit", self.simulator.font, (0, 0, 0))
        ]
        
        for key, text, font, color in static_texts:
            self.text_cache[key] = font.render(text, True, color)
    
    def _get_cached_text(self, key, text, font, color):
        """Get or create cached text surface."""
        cache_key = f"{key}_{text}_{color}"
        if cache_key not in self.text_cache:
            self.text_cache[cache_key] = font.render(text, True, color)
        return self.text_cache[cache_key]
    
    def _find_safe_spawn_position(self, min_distance=30):
        """Find a safe position that doesn't overlap with existing entities.
        
        Args:
            min_distance: Minimum distance from other entities
            
        Returns:
            np.array: Safe position coordinates
        """
        max_attempts = 100
        area_size = self.simulator.environment.object_area_size
        
        for _ in range(max_attempts):
            # Generate random position within bounds
            x = random.uniform(min_distance, area_size - min_distance)
            y = random.uniform(min_distance, area_size - min_distance)
            position = np.array([x, y])
            
            # Check if position is safe (only check active entities)
            safe = True
            for entity in self.simulator.environment.entities:
                if entity.active:  # Only check active entities
                    distance = np.linalg.norm(entity.position - position)
                    if distance < min_distance:
                        safe = False
                        break
            
            if safe:
                return position
        
        # Fallback to random position if no safe position found
        return np.array([
            random.uniform(min_distance, area_size - min_distance),
            random.uniform(min_distance, area_size - min_distance)
        ])
    
    def create_manual_animat(self):
        """Create an animat with survival AI when not manually controlled.
        
        Returns:
            Animat: The manually controlled animat
        """
        # Find a safe starting position away from traps
        start_position = self._find_safe_spawn_position(min_distance=50)
        
        # Create animat
        animat = Animat(start_position)
        self.target_position = start_position.copy()
        
        # Clear existing animats and add our manual one
        self.simulator.environment.animats.clear()
        self.simulator.environment.entities = [e for e in self.simulator.environment.entities 
                                            if e.type != EntityType.ANIMAT]
        self.simulator.environment.add_entity(animat)
        
        return animat
    
    def respawn_entity_safely(self, entity_type, original_radius=None):
        """Respawn an entity at a safe location with proper size.
        
        Args:
            entity_type: Type of entity to respawn
            original_radius: Original radius of the entity (if known)
        """
        # Remove inactive entities from the list first
        self.simulator.environment.entities = [e for e in self.simulator.environment.entities if e.active]
        
        safe_position = self._find_safe_spawn_position()
        
        # Use original radius or default based on entity type
        if original_radius is None:
            # Use default sizes from config or environment
            if entity_type == EntityType.FOOD:
                radius = getattr(settings, 'FOOD_RADIUS', 10)
            elif entity_type == EntityType.WATER:
                radius = getattr(settings, 'WATER_RADIUS', 10)
            else:
                radius = 8
        else:
            radius = original_radius
        
        # Create new entity at safe position
        if entity_type == EntityType.FOOD:
            new_entity = type('Entity', (), {
                'position': safe_position,
                'type': EntityType.FOOD,
                'active': True,
                'radius': radius
            })()
        elif entity_type == EntityType.WATER:
            new_entity = type('Entity', (), {
                'position': safe_position,
                'type': EntityType.WATER,
                'active': True,
                'radius': radius
            })()
        else:
            return
        
        # Add to environment
        self.simulator.environment.entities.append(new_entity)
    
    def update_respawn_timers(self, dt):
        """Update respawn timers and handle respawning.
        
        Args:
            dt: Time delta
        """
        # Update food respawn timer
        if self.pending_food_respawns:
            self.food_respawn_timer += dt
            if self.food_respawn_timer >= self.respawn_delay:
                for entity_info in self.pending_food_respawns:
                    self.respawn_entity_safely(
                        entity_info['type'], 
                        original_radius=entity_info['radius']
                    )
                self.pending_food_respawns.clear()
                self.food_respawn_timer = 0.0
        
        # Update water respawn timer
        if self.pending_water_respawns:
            self.water_respawn_timer += dt
            if self.water_respawn_timer >= self.respawn_delay:
                for entity_info in self.pending_water_respawns:
                    self.respawn_entity_safely(
                        entity_info['type'], 
                        original_radius=entity_info['radius']
                    )
                self.pending_water_respawns.clear()
                self.water_respawn_timer = 0.0
    
    def find_nearest_entity(self, entity_type):
        """Find the nearest entity of a specific type.
        
        Args:
            entity_type: The type of entity to find
            
        Returns:
            tuple: (entity, distance) or (None, float('inf'))
        """
        if not self.animat:
            return None, float('inf')
        
        nearest_entity = None
        min_distance = float('inf')
        
        for entity in self.simulator.environment.entities:
            if entity.type == entity_type and entity.active:
                distance = np.linalg.norm(entity.position - self.animat.position)
                if distance < min_distance:
                    min_distance = distance
                    nearest_entity = entity
        
        return nearest_entity, min_distance
    
    def automatic_behavior(self, dt):
        """Implement automatic survival behavior when not manually controlled.
        
        Args:
            dt: Time delta
        """
        if not self.animat or not self.animat.active or self.manual_mode:
            return
        
        # Determine what the animat needs most
        battery1_pct = self.animat.batteries[0] / settings.BATTERY_MAX
        battery2_pct = self.animat.batteries[1] / settings.BATTERY_MAX
        
        # Emergency thresholds
        critical_threshold = 0.2
        
        target_entity = None
        target_type = None
        
        # Priority: avoid immediate death, then seek what's needed most
        if battery1_pct < critical_threshold or battery2_pct < critical_threshold:
            # Emergency mode - go for whatever is closer
            food, food_dist = self.find_nearest_entity(EntityType.FOOD)
            water, water_dist = self.find_nearest_entity(EntityType.WATER)
            
            if battery1_pct < battery2_pct and food:
                target_entity = food
                target_type = EntityType.FOOD
            elif water:
                target_entity = water
                target_type = EntityType.WATER
            elif food:  # Fallback if no water
                target_entity = food
                target_type = EntityType.FOOD
        else:
            # Normal mode - go for what's needed more
            if battery1_pct < battery2_pct:
                target_entity, _ = self.find_nearest_entity(EntityType.FOOD)
                target_type = EntityType.FOOD
            else:
                target_entity, _ = self.find_nearest_entity(EntityType.WATER)
                target_type = EntityType.WATER
        
        # If no target found, find the nearest resource
        if not target_entity:
            food, food_dist = self.find_nearest_entity(EntityType.FOOD)
            water, water_dist = self.find_nearest_entity(EntityType.WATER)
            if food_dist < water_dist:
                target_entity = food
                target_type = EntityType.FOOD
            else:
                target_entity = water
                target_type = EntityType.WATER
        
        # Avoid traps
        trap, trap_dist = self.find_nearest_entity(EntityType.TRAP)
        
        # Movement logic with smooth interpolation
        if target_entity:
            # Calculate direction to target
            to_target = target_entity.position - self.animat.position
            target_distance = np.linalg.norm(to_target)
            
            if target_distance > 0:
                target_direction = to_target / target_distance
                
                # Check for nearby traps and adjust direction
                if trap and trap_dist < 60:  # If trap is close
                    to_trap = trap.position - self.animat.position
                    trap_direction = to_trap / np.linalg.norm(to_trap)
                    
                    # Add avoidance vector
                    avoidance_strength = max(0, (60 - trap_dist) / 60)
                    avoidance_vector = -trap_direction * avoidance_strength
                    
                    # Combine target seeking with trap avoidance
                    final_direction = target_direction + avoidance_vector * 2.5
                    if np.linalg.norm(final_direction) > 0:
                        final_direction = final_direction / np.linalg.norm(final_direction)
                else:
                    final_direction = target_direction
                
                # Smooth turning towards target
                current_angle = np.arctan2(self.animat.direction[1], self.animat.direction[0])
                target_angle = np.arctan2(final_direction[1], final_direction[0])
                
                angle_diff = target_angle - current_angle
                # Normalize angle difference to [-pi, pi]
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                
                # Smooth turn towards target
                max_turn = self.turn_speed * dt
                turn_amount = np.clip(angle_diff, -max_turn, max_turn)
                new_angle = current_angle + turn_amount
                self.animat.direction = np.array([np.cos(new_angle), np.sin(new_angle)])
                
                # Move forward if facing roughly the right direction
                if abs(angle_diff) < np.pi / 2:  # Within 90 degrees
                    move_distance = self.move_speed * dt * 0.8  # Auto mode slightly slower
                    movement = self.animat.direction * move_distance
                    new_position = self.animat.position + movement
                    
                    # Check collision and move smoothly
                    self._handle_smooth_movement(new_position, dt)
    
    def handle_manual_input(self, keys, dt):
        """Handle keyboard input for manual control with smooth movement.
        
        Args:
            keys: Pygame key state
            dt: Time delta
        """
        if not self.animat or not self.animat.active:
            return
        
        # Check for any manual input
        manual_input_detected = (
            keys[pygame.K_w] or keys[pygame.K_UP] or 
            keys[pygame.K_s] or keys[pygame.K_DOWN] or 
            keys[pygame.K_a] or keys[pygame.K_LEFT] or 
            keys[pygame.K_d] or keys[pygame.K_RIGHT]
        )
        
        if manual_input_detected:
            self.manual_mode = True
            self.last_manual_input = time.time()
            
            # Movement controls
            forward_input = 0
            turn_input = 0
            
            # WASD or Arrow key controls
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                forward_input = 1
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                forward_input = -0.5  # Reverse slower
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                turn_input = -1
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                turn_input = 1
            
            # Apply smooth turning
            if turn_input != 0:
                angle_change = turn_input * self.turn_speed * dt
                current_angle = np.arctan2(self.animat.direction[1], self.animat.direction[0])
                new_angle = current_angle + angle_change
                self.animat.direction = np.array([np.cos(new_angle), np.sin(new_angle)])
            
            # Apply smooth manual movement
            if forward_input != 0:
                move_distance = forward_input * self.move_speed * dt
                movement = self.animat.direction * move_distance
                new_position = self.animat.position + movement
                self._handle_smooth_movement(new_position, dt)
        else:
            # Check if we should switch to auto mode
            if time.time() - self.last_manual_input > self.manual_timeout:
                self.manual_mode = False
    
    def _handle_smooth_movement(self, new_position, dt):
        """Handle movement with smooth interpolation and collision detection.
        
        Args:
            new_position: The target position to move to
            dt: Time delta
        """
        # Check for collisions before moving
        collision, entity = self.simulator.environment.check_collision(
            new_position, self.animat.radius, exclude_entity=self.animat)
        
        if not collision:
            # Smooth movement interpolation
            if self.target_position is None:
                self.target_position = self.animat.position.copy()
            
            # Update target position
            self.target_position = new_position
            
            # Interpolate towards target position
            direction_to_target = self.target_position - self.animat.position
            distance_to_target = np.linalg.norm(direction_to_target)
            
            if distance_to_target > 1.0:  # Only interpolate if there's significant distance
                # Smooth interpolation factor
                lerp_factor = min(1.0, self.movement_smoothing * dt * 10)
                self.animat.position = self.animat.position + direction_to_target * lerp_factor
            else:
                self.animat.position = self.target_position
        else:
            # Handle collision with entity
            if entity.type == EntityType.FOOD and entity.active:
                # Consume food - restore battery 1
                old_battery = self.animat.batteries[0]
                self.animat.batteries[0] = min(settings.BATTERY_MAX, 
                                            self.animat.batteries[0] + 80)
                entity.active = False
                
                # Schedule respawn with original properties
                self.pending_food_respawns.append({
                    'type': EntityType.FOOD,
                    'radius': entity.radius  # Preserve original radius
                })
                print(f"Food consumed! Battery 1: {old_battery:.1f} -> {self.animat.batteries[0]:.1f}")
                self.animat.position = new_position
                self.target_position = new_position
            
            elif entity.type == EntityType.WATER and entity.active:
                # Consume water - restore battery 2
                old_battery = self.animat.batteries[1]
                self.animat.batteries[1] = min(settings.BATTERY_MAX, 
                                            self.animat.batteries[1] + 80)
                entity.active = False
                
                # Schedule respawn with original properties
                self.pending_water_respawns.append({
                    'type': EntityType.WATER,
                    'radius': entity.radius  # Preserve original radius
                })
                print(f"Water consumed! Battery 2: {old_battery:.1f} -> {self.animat.batteries[1]:.1f}")
                self.animat.position = new_position
                self.target_position = new_position
            
            elif entity.type == EntityType.TRAP:
                # Hit a trap - animat dies
                self.animat.active = False
                print("Hit a trap! Game Over!")
                print(f"Survival time: {self.animat.survival_time:.1f} seconds")
                return
    
    def update_animat(self, dt):
        """Update the manually controlled animat.
        
        Args:
            dt: Time delta
        """
        if not self.animat or not self.animat.active:
            return
        
        # Update survival time
        self.animat.survival_time += dt
        
        # Decrease batteries over time
        self.animat.batteries[0] = max(0, self.animat.batteries[0] - 
                                    settings.BATTERY_DECAY_RATE * dt)
        self.animat.batteries[1] = max(0, self.animat.batteries[1] - 
                                    settings.BATTERY_DECAY_RATE * dt)
        
        # Check if animat dies from lack of energy
        if self.animat.batteries[0] <= 0 or self.animat.batteries[1] <= 0:
            self.animat.active = False
            print("Ran out of energy! Game Over!")
            print(f"Survival time: {self.animat.survival_time:.1f} seconds")
            return
    
    def render_ui(self):
        """Render the manual mode UI elements with reduced flicker using cached surfaces."""
        if self.simulator.headless or not self.animat:
            return
        
        # Use pre-created overlays
        self.simulator.screen.blit(self.ui_surfaces['instruction_overlay'], (10, 10))
        
        # Render mode header (only this changes frequently)
        mode_text = "MANUAL" if self.manual_mode else "AUTO"
        mode_color = (255, 0, 0) if self.manual_mode else (0, 150, 0)
        mode_surface = self._get_cached_text(
            "mode", 
            f"=== {mode_text} MODE ===", 
            pygame.font.SysFont('Arial', 16, bold=True), 
            mode_color
        )
        self.simulator.screen.blit(mode_surface, (20, 20))
        
        # Render static instructions using cached text
        y_offset = 45
        static_keys = [
            "wasd_instruction", "auto_instruction", "food_instruction",
            "water_instruction", "trap_instruction", "controls_instruction"
        ]
        for key in static_keys:
            self.simulator.screen.blit(self.text_cache[key], (20, y_offset))
            y_offset += 20
        
        # Status overlay
        self.simulator.screen.blit(self.ui_surfaces['status_overlay'], (10, self.simulator.height - 150))
        
        # Render animat status
        if self.animat.active:
            # Calculate battery percentages and create colored bars
            battery1_pct = (self.animat.batteries[0] / settings.BATTERY_MAX) * 100
            battery2_pct = (self.animat.batteries[1] / settings.BATTERY_MAX) * 100
            
            # Render status header using cached text
            self.simulator.screen.blit(self.text_cache["status_header"], (20, self.simulator.height - 140))
            
            # Dynamic status text (these change frequently so cache with values)
            status_texts = [
                (f"Survival Time: {self.animat.survival_time:.1f}s", (0, 0, 0)),
                (f"Left Battery (Food): {self.animat.batteries[0]:.1f}/{settings.BATTERY_MAX} ({battery1_pct:.1f}%)", (0, 0, 0)),
                (f"Right Battery (Water): {self.animat.batteries[1]:.1f}/{settings.BATTERY_MAX} ({battery2_pct:.1f}%)", (0, 0, 0)),
                (f"Position: ({self.animat.position[0]:.1f}, {self.animat.position[1]:.1f})", (0, 0, 0)),
                (f"Food respawning: {len(self.pending_food_respawns)} | Water respawning: {len(self.pending_water_respawns)}", (0, 0, 0))
            ]
            
            y_offset = self.simulator.height - 120
            for text, color in status_texts:
                text_surface = self._get_cached_text("dynamic_status", text, self.simulator.font, color)
                self.simulator.screen.blit(text_surface, (20, y_offset))
                y_offset += 20
            
            # Draw battery bars with smooth gradients
            bar_width = 150
            bar_height = 10
            
            # Left battery bar
            battery1_width = int((self.animat.batteries[0] / settings.BATTERY_MAX) * bar_width)
            pygame.draw.rect(self.simulator.screen, (200, 200, 200), 
                        (180, self.simulator.height - 88, bar_width, bar_height))
            battery1_color = (0, 255, 0) if battery1_pct > 30 else (255, 165, 0) if battery1_pct > 10 else (255, 0, 0)
            if battery1_width > 0:
                pygame.draw.rect(self.simulator.screen, battery1_color, 
                            (180, self.simulator.height - 88, battery1_width, bar_height))
            
            # Right battery bar  
            battery2_width = int((self.animat.batteries[1] / settings.BATTERY_MAX) * bar_width)
            pygame.draw.rect(self.simulator.screen, (200, 200, 200), 
                        (180, self.simulator.height - 68, bar_width, bar_height))
            battery2_color = (0, 255, 0) if battery2_pct > 30 else (255, 165, 0) if battery2_pct > 10 else (255, 0, 0)
            if battery2_width > 0:
                pygame.draw.rect(self.simulator.screen, battery2_color, 
                            (180, self.simulator.height - 68, battery2_width, bar_height))
            
        else:
            # Game over screen using cached surfaces and text
            game_over_rect = self.ui_surfaces['game_over_overlay'].get_rect(
                center=(self.simulator.width // 2, self.simulator.height // 2)
            )
            self.simulator.screen.blit(self.ui_surfaces['game_over_overlay'], game_over_rect)
            
            # Use cached game over text
            game_over_rect = self.text_cache["game_over"].get_rect(
                center=(self.simulator.width // 2, self.simulator.height // 2 - 20)
            )
            restart_rect = self.text_cache["restart_prompt"].get_rect(
                center=(self.simulator.width // 2, self.simulator.height // 2 + 10)
            )
            
            self.simulator.screen.blit(self.text_cache["game_over"], game_over_rect)
            self.simulator.screen.blit(self.text_cache["restart_prompt"], restart_rect)
    
    def restart_game(self):
        """Restart the manual mode game."""
        # Clear pending respawns
        self.pending_food_respawns.clear()
        self.pending_water_respawns.clear()
        self.food_respawn_timer = 0.0
        self.water_respawn_timer = 0.0
        
        # Clear text cache for dynamic content
        keys_to_remove = [k for k in self.text_cache.keys() if k.startswith("dynamic_")]
        for key in keys_to_remove:
            del self.text_cache[key]
        
        # Reinitialize environment
        self.simulator.environment.initialize_random_environment()
        
        # Create new manual animat
        self.animat = self.create_manual_animat()
        
        # Reset manual mode
        self.manual_mode = False
        self.last_manual_input = time.time()
        
        # Follow the new animat
        self.simulator.follow_animat(self.animat)
        
        print("Game restarted!")
    
    def run(self):
        """Run the manual mode simulation with improved performance."""
        self.is_running = True
        
        # Initialize the environment
        self.simulator.environment.initialize_random_environment()
        
        # Create the manual animat
        self.animat = self.create_manual_animat()
        
        # Initialize manual mode state
        self.manual_mode = False
        self.last_manual_input = time.time()
        
        # Follow the animat with camera
        self.simulator.follow_animat(self.animat)
        
        print("Manual mode started. Use WASD or arrow keys to control.")
        print("Animat will automatically survive when not controlled.")
        print("Food and water will respawn safely after consumption.")
        
        # Main game loop with vsync and optimized rendering
        while self.is_running:
            dt = self.simulator.clock.tick(self.simulator.fps) / 1000.0
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False
                    elif event.key == pygame.K_r and not self.animat.active:
                        self.restart_game()
            
            # Get current key states
            keys = pygame.key.get_pressed()
            
            # Handle manual input
            self.handle_manual_input(keys, dt)
            
            # Run automatic behavior when not in manual mode
            if not self.manual_mode:
                self.automatic_behavior(dt)
            
            # Update animat
            self.update_animat(dt)
            
            # Update respawn timers
            self.update_respawn_timers(dt)
            
            # Update environment
            self.simulator.environment.update(dt)
            
            # Update simulation time
            self.simulator.simulation_time += dt
            
            # Clear screen once
            self.simulator.screen.fill((0, 0, 0))
            
            # Render everything
            self.simulator.render()
            self.render_ui()
            
            # Single display update to reduce flicker
            pygame.display.flip()
        
        print("Manual mode ended.")