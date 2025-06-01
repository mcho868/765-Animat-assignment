"""
Multi-agent manual controller for animats with automatic survival behavior when not manually controlled.
Allows keyboard control of multiple animats with the ability to switch between them.
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
    """Multi-agent manual controller for animat simulation with automatic survival AI."""
    
    def __init__(self, simulator, num_agents=3):
        """Initialize the multi-agent manual controller.
        
        Args:
            simulator: The main simulator instance
            num_agents: Number of agents to create and manage
        """
        self.simulator = simulator
        self.num_agents = num_agents
        self.animats = []  # List of all animats
        self.active_animat_index = 0  # Index of currently controlled animat
        self.is_running = False
        self.move_speed = 150.0  # Pixels per second for smooth movement
        self.turn_speed = 3.0  # Radians per second for smooth turning
        self.manual_mode = False  # Whether player is actively controlling
        self.last_manual_input = 0  # Time since last manual input
        self.manual_timeout = 1.0  # Seconds before switching to auto mode
        
        # Respawn management
        self.food_respawn_timer = 0.0
        self.water_respawn_timer = 0.0
        self.respawn_delay = 2.0  # Seconds before respawning consumed resources
        self.pending_food_respawns = []
        self.pending_water_respawns = []
        
        # Smooth movement interpolation (per animat)
        self.target_positions = {}  # Dictionary mapping animat to target position
        self.movement_smoothing = 0.8  # Lower = smoother but slower response
        
        # UI caching to prevent flickering
        self.text_cache = {}
        self.ui_surfaces = {}
        
        # Agent colors for visual distinction
        self.agent_colors = [
            (255, 255, 0),   # Yellow - Agent 1
            (255, 0, 255),   # Magenta - Agent 2  
            (0, 255, 255),   # Cyan - Agent 3
            (255, 128, 0),   # Orange - Agent 4
            (128, 255, 0),   # Lime - Agent 5
            (255, 0, 128),   # Pink - Agent 6
            (128, 0, 255),   # Purple - Agent 7
            (0, 128, 255),   # Light Blue - Agent 8
        ]
        
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
            pygame.display.set_caption(settings.WINDOW_TITLE + " - Multi-Agent Manual Mode")
        
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
                EntityType.ANIMAT: (255, 255, 0),      # Yellow (will be overridden per agent)
            }
    
    def _create_ui_surfaces(self):
        """Pre-create UI surfaces to prevent flickering."""
        # Create base overlays (larger to accommodate multi-agent info)
        self.ui_surfaces['instruction_overlay'] = pygame.Surface((450, 280), pygame.SRCALPHA)
        self.ui_surfaces['instruction_overlay'].fill((255, 255, 255, 180))
        
        self.ui_surfaces['status_overlay'] = pygame.Surface((600, 200), pygame.SRCALPHA)
        self.ui_surfaces['status_overlay'].fill((255, 255, 255, 180))
        
        self.ui_surfaces['game_over_overlay'] = pygame.Surface((700, 150), pygame.SRCALPHA)
        self.ui_surfaces['game_over_overlay'].fill((255, 255, 255, 200))
        
        # Pre-render static text
        self._cache_static_text()
    
    def _cache_static_text(self):
        """Cache static text that doesn't change."""
        static_texts = [
            ("wasd_instruction", "WASD or Arrow Keys - Manual Control", self.simulator.font, (0, 0, 0)),
            ("auto_instruction", "Auto mode when idle for 1 second", self.simulator.font, (0, 0, 0)),
            ("switch_instruction", "1-8 Keys - Switch Active Agent", self.simulator.font, (0, 0, 0)),
            ("tab_instruction", "TAB - Next Agent | SHIFT+TAB - Previous Agent", self.simulator.font, (0, 0, 0)),
            ("food_instruction", "Green circles - Food (restores left battery)", self.simulator.font, (0, 0, 0)),
            ("water_instruction", "Blue circles - Water (restores right battery)", self.simulator.font, (0, 0, 0)),
            ("trap_instruction", "Red circles - Traps (AVOID!)", self.simulator.font, (0, 0, 0)),
            ("controls_instruction", "ESC - Quit | R - Restart (when all dead)", self.simulator.font, (0, 0, 0)),
            ("status_header", "=== MULTI-AGENT STATUS ===", pygame.font.SysFont('Arial', 16, bold=True), (0, 0, 150)),
            ("game_over", "ALL AGENTS DEAD", pygame.font.SysFont('Arial', 24, bold=True), (255, 0, 0)),
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
    
    def _find_safe_spawn_position(self, min_distance=50, avoid_other_animats=True):
        """Find a safe position that doesn't overlap with existing entities.
        
        Args:
            min_distance: Minimum distance from other entities
            avoid_other_animats: Whether to avoid other animats when spawning
            
        Returns:
            np.array: Safe position coordinates
        """
        max_attempts = 200
        area_size = self.simulator.environment.object_area_size
        
        for _ in range(max_attempts):
            # Generate random position within bounds
            x = random.uniform(min_distance, area_size - min_distance)
            y = random.uniform(min_distance, area_size - min_distance)
            position = np.array([x, y])
            
            # Check if position is safe (only check active entities)
            safe = True
            for entity in self.simulator.environment.entities:
                if entity.active:
                    distance = np.linalg.norm(entity.position - position)
                    if distance < min_distance:
                        safe = False
                        break
            
            # Also check against other animats if requested
            if safe and avoid_other_animats:
                for animat in self.animats:
                    if animat.active:
                        distance = np.linalg.norm(animat.position - position)
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
    
    def create_multiple_animats(self):
        """Create multiple animats with survival AI.
        
        Returns:
            list: List of created animats
        """
        self.animats = []
        
        # Clear existing animats
        self.simulator.environment.animats.clear()
        self.simulator.environment.entities = [e for e in self.simulator.environment.entities 
                                            if e.type != EntityType.ANIMAT]
        
        # Create multiple animats at safe positions
        for i in range(self.num_agents):
            start_position = self._find_safe_spawn_position(min_distance=60)
            
            # Create animat
            animat = Animat(start_position)
            
            # Assign unique color and ID
            animat.agent_id = i
            animat.color = self.agent_colors[i % len(self.agent_colors)]
            
            # Initialize target position
            self.target_positions[animat] = start_position.copy()
            
            # Add to our list and environment
            self.animats.append(animat)
            self.simulator.environment.add_entity(animat)
        
        return self.animats
    
    def get_active_animat(self):
        """Get the currently active (controlled) animat."""
        if 0 <= self.active_animat_index < len(self.animats):
            return self.animats[self.active_animat_index]
        return None
    
    def switch_active_animat(self, new_index):
        """Switch to a different active animat."""
        if 0 <= new_index < len(self.animats):
            old_index = self.active_animat_index
            self.active_animat_index = new_index
            active_animat = self.get_active_animat()
            
            if active_animat and active_animat.active:
                # Follow the new active animat
                self.simulator.follow_animat(active_animat)
                print(f"Switched control: Agent {old_index + 1} -> Agent {new_index + 1}")
                return True
        return False
    
    def switch_to_next_active_animat(self):
        """Switch to the next living animat."""
        start_index = self.active_animat_index
        attempts = 0
        
        while attempts < len(self.animats):
            self.active_animat_index = (self.active_animat_index + 1) % len(self.animats)
            if self.animats[self.active_animat_index].active:
                self.simulator.follow_animat(self.animats[self.active_animat_index])
                print(f"Switched to Agent {self.active_animat_index + 1}")
                return True
            attempts += 1
        
        # No active animats found, stay on current
        self.active_animat_index = start_index
        return False
    
    def switch_to_previous_active_animat(self):
        """Switch to the previous living animat."""
        start_index = self.active_animat_index
        attempts = 0
        
        while attempts < len(self.animats):
            self.active_animat_index = (self.active_animat_index - 1) % len(self.animats)
            if self.animats[self.active_animat_index].active:
                self.simulator.follow_animat(self.animats[self.active_animat_index])
                print(f"Switched to Agent {self.active_animat_index + 1}")
                return True
            attempts += 1
        
        # No active animats found, stay on current
        self.active_animat_index = start_index
        return False
    
    def respawn_entity_safely(self, entity_type, original_radius=None):
        """Respawn an entity at a safe location with proper size."""
        # Remove inactive entities from the list first
        self.simulator.environment.entities = [e for e in self.simulator.environment.entities if e.active]
        
        safe_position = self._find_safe_spawn_position(min_distance=30)
        
        # Use original radius or default based on entity type
        if original_radius is None:
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
        """Update respawn timers and handle respawning."""
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
    
    def find_nearest_entity(self, animat, entity_type):
        """Find the nearest entity of a specific type for a given animat."""
        if not animat:
            return None, float('inf')
        
        nearest_entity = None
        min_distance = float('inf')
        
        for entity in self.simulator.environment.entities:
            if entity.type == entity_type and entity.active:
                distance = np.linalg.norm(entity.position - animat.position)
                if distance < min_distance:
                    min_distance = distance
                    nearest_entity = entity
        
        return nearest_entity, min_distance
    
    def automatic_behavior(self, animat, dt):
        """Implement automatic survival behavior for a specific animat when not manually controlled."""
        if not animat or not animat.active or (animat == self.get_active_animat() and self.manual_mode):
            return
        
        # Determine what the animat needs most
        battery1_pct = animat.batteries[0] / settings.BATTERY_MAX
        battery2_pct = animat.batteries[1] / settings.BATTERY_MAX
        
        # Emergency thresholds
        critical_threshold = 0.2
        
        target_entity = None
        target_type = None
        
        # Priority: avoid immediate death, then seek what's needed most
        if battery1_pct < critical_threshold or battery2_pct < critical_threshold:
            # Emergency mode - go for whatever is closer
            food, food_dist = self.find_nearest_entity(animat, EntityType.FOOD)
            water, water_dist = self.find_nearest_entity(animat, EntityType.WATER)
            
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
                target_entity, _ = self.find_nearest_entity(animat, EntityType.FOOD)
                target_type = EntityType.FOOD
            else:
                target_entity, _ = self.find_nearest_entity(animat, EntityType.WATER)
                target_type = EntityType.WATER
        
        # If no target found, find the nearest resource
        if not target_entity:
            food, food_dist = self.find_nearest_entity(animat, EntityType.FOOD)
            water, water_dist = self.find_nearest_entity(animat, EntityType.WATER)
            if food_dist < water_dist:
                target_entity = food
                target_type = EntityType.FOOD
            else:
                target_entity = water
                target_type = EntityType.WATER
        
        # Avoid traps
        trap, trap_dist = self.find_nearest_entity(animat, EntityType.TRAP)
        
        # Movement logic with smooth interpolation
        if target_entity:
            # Calculate direction to target
            to_target = target_entity.position - animat.position
            target_distance = np.linalg.norm(to_target)
            
            if target_distance > 0:
                target_direction = to_target / target_distance
                
                # Check for nearby traps and adjust direction
                if trap and trap_dist < 60:  # If trap is close
                    to_trap = trap.position - animat.position
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
                current_angle = np.arctan2(animat.direction[1], animat.direction[0])
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
                animat.direction = np.array([np.cos(new_angle), np.sin(new_angle)])
                
                # Move forward if facing roughly the right direction
                if abs(angle_diff) < np.pi / 2:  # Within 90 degrees
                    move_distance = self.move_speed * dt * 0.6  # Auto mode slower than manual
                    movement = animat.direction * move_distance
                    new_position = animat.position + movement
                    
                    # Check collision and move smoothly
                    self._handle_smooth_movement(animat, new_position, dt)
    
    def handle_manual_input(self, keys, dt):
        """Handle keyboard input for manual control with smooth movement."""
        active_animat = self.get_active_animat()
        if not active_animat or not active_animat.active:
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
                current_angle = np.arctan2(active_animat.direction[1], active_animat.direction[0])
                new_angle = current_angle + angle_change
                active_animat.direction = np.array([np.cos(new_angle), np.sin(new_angle)])
            
            # Apply smooth manual movement
            if forward_input != 0:
                move_distance = forward_input * self.move_speed * dt
                movement = active_animat.direction * move_distance
                new_position = active_animat.position + movement
                self._handle_smooth_movement(active_animat, new_position, dt)
        else:
            # Check if we should switch to auto mode
            if time.time() - self.last_manual_input > self.manual_timeout:
                self.manual_mode = False
    
    def _handle_smooth_movement(self, animat, new_position, dt):
        """Handle movement with smooth interpolation and collision detection."""
        # Check for collisions before moving
        collision, entity = self.simulator.environment.check_collision(
            new_position, animat.radius, exclude_entity=animat)
        
        if not collision:
            # Smooth movement interpolation
            if animat not in self.target_positions:
                self.target_positions[animat] = animat.position.copy()
            
            # Update target position
            self.target_positions[animat] = new_position
            
            # Interpolate towards target position
            direction_to_target = self.target_positions[animat] - animat.position
            distance_to_target = np.linalg.norm(direction_to_target)
            
            if distance_to_target > 1.0:  # Only interpolate if there's significant distance
                # Smooth interpolation factor
                lerp_factor = min(1.0, self.movement_smoothing * dt * 10)
                animat.position = animat.position + direction_to_target * lerp_factor
            else:
                animat.position = self.target_positions[animat]
        else:
            # Handle collision with entity
            if entity.type == EntityType.FOOD and entity.active:
                # Consume food - restore battery 1
                old_battery = animat.batteries[0]
                animat.batteries[0] = min(settings.BATTERY_MAX, 
                                        animat.batteries[0] + 80)
                entity.active = False
                
                # Schedule respawn with original properties
                self.pending_food_respawns.append({
                    'type': EntityType.FOOD,
                    'radius': entity.radius
                })
                print(f"Agent {animat.agent_id + 1} consumed food! Battery 1: {old_battery:.1f} -> {animat.batteries[0]:.1f}")
                animat.position = new_position
                self.target_positions[animat] = new_position
            
            elif entity.type == EntityType.WATER and entity.active:
                # Consume water - restore battery 2
                old_battery = animat.batteries[1]
                animat.batteries[1] = min(settings.BATTERY_MAX, 
                                        animat.batteries[1] + 80)
                entity.active = False
                
                # Schedule respawn with original properties
                self.pending_water_respawns.append({
                    'type': EntityType.WATER,
                    'radius': entity.radius
                })
                print(f"Agent {animat.agent_id + 1} consumed water! Battery 2: {old_battery:.1f} -> {animat.batteries[1]:.1f}")
                animat.position = new_position
                self.target_positions[animat] = new_position
            
            elif entity.type == EntityType.TRAP:
                # Hit a trap - animat dies
                animat.active = False
                print(f"Agent {animat.agent_id + 1} hit a trap! Agent eliminated!")
                print(f"Agent {animat.agent_id + 1} survival time: {animat.survival_time:.1f} seconds")
                
                # Switch to next living animat if current one died
                if animat == self.get_active_animat():
                    self.switch_to_next_active_animat()
    
    def update_animats(self, dt):
        """Update all animats."""
        for animat in self.animats:
            if not animat.active:
                continue
            
            # Update survival time
            animat.survival_time += dt
            
            # Decrease batteries over time
            animat.batteries[0] = max(0, animat.batteries[0] - 
                                    settings.BATTERY_DECAY_RATE * dt)
            animat.batteries[1] = max(0, animat.batteries[1] - 
                                    settings.BATTERY_DECAY_RATE * dt)
            
            # Check if animat dies from lack of energy
            if animat.batteries[0] <= 0 or animat.batteries[1] <= 0:
                animat.active = False
                print(f"Agent {animat.agent_id + 1} ran out of energy! Agent eliminated!")
                print(f"Agent {animat.agent_id + 1} survival time: {animat.survival_time:.1f} seconds")
                
                # Switch to next living animat if current one died
                if animat == self.get_active_animat():
                    self.switch_to_next_active_animat()
    
    def all_animats_dead(self):
        """Check if all animats are dead."""
        return all(not animat.active for animat in self.animats)
    
    def get_living_animats_count(self):
        """Get the number of living animats."""
        return sum(1 for animat in self.animats if animat.active)
    
    def render_ui(self):
        """Render the multi-agent manual mode UI elements with reduced flicker using cached surfaces."""
        if self.simulator.headless:
            return
        
        # Use pre-created overlays
        self.simulator.screen.blit(self.ui_surfaces['instruction_overlay'], (10, 10))
        
        # Render mode header (only this changes frequently)
        active_animat = self.get_active_animat()
        if active_animat and active_animat.active:
            mode_text = f"MANUAL - Agent {self.active_animat_index + 1}" if self.manual_mode else f"AUTO - Agent {self.active_animat_index + 1}"
            mode_color = (255, 0, 0) if self.manual_mode else (0, 150, 0)
        else:
            mode_text = "ALL AGENTS DEAD"
            mode_color = (255, 0, 0)
        
        mode_surface = self._get_cached_text(
            "mode", 
            f"=== {mode_text} ===", 
            pygame.font.SysFont('Arial', 16, bold=True), 
            mode_color
        )
        self.simulator.screen.blit(mode_surface, (20, 20))
        
        # Render static instructions using cached text
        y_offset = 45
        static_keys = [
            "wasd_instruction", "auto_instruction", "switch_instruction", "tab_instruction",
            "food_instruction", "water_instruction", "trap_instruction", "controls_instruction"
        ]
        for key in static_keys:
            self.simulator.screen.blit(self.text_cache[key], (20, y_offset))
            y_offset += 20
        
        # Status overlay
        self.simulator.screen.blit(self.ui_surfaces['status_overlay'], (10, self.simulator.height - 210))
        
        # Render multi-agent status
        if not self.all_animats_dead():
            # Render status header using cached text
            self.simulator.screen.blit(self.text_cache["status_header"], (20, self.simulator.height - 200))
            
            # Show living agents count
            living_count = self.get_living_animats_count()
            living_text = f"Living Agents: {living_count}/{self.num_agents}"
            living_surface = self._get_cached_text("living_count", living_text, self.simulator.font, (0, 0, 0))
            self.simulator.screen.blit(living_surface, (20, self.simulator.height - 180))
            
            # Show status for each living animat
            y_offset = self.simulator.height - 160
            for i, animat in enumerate(self.animats):
                if not animat.active:
                    continue
                
                # Agent indicator with color
                agent_color = animat.color
                active_indicator = ">>>" if i == self.active_animat_index else "   "
                
                battery1_pct = (animat.batteries[0] / settings.BATTERY_MAX) * 100
                battery2_pct = (animat.batteries[1] / settings.BATTERY_MAX) * 100
                
                agent_text = f"{active_indicator} Agent {i + 1}: L:{battery1_pct:.0f}% R:{battery2_pct:.0f}% T:{animat.survival_time:.1f}s"
                text_color = agent_color if i == self.active_animat_index else (100, 100, 100)
                
                agent_surface = self._get_cached_text(f"agent_{i}", agent_text, self.simulator.font, text_color)
                self.simulator.screen.blit(agent_surface, (20, y_offset))
                
                # Draw mini battery bars for active agent
                if i == self.active_animat_index:
                    bar_width = 100
                    bar_height = 6
                    
                    # Left battery bar
                    battery1_width = int((animat.batteries[0] / settings.BATTERY_MAX) * bar_width)
                    pygame.draw.rect(self.simulator.screen, (200, 200, 200), 
                                (380, y_offset + 2, bar_width, bar_height))
                    battery1_color = (0, 255, 0) if battery1_pct > 30 else (255, 165, 0) if battery1_pct > 10 else (255, 0, 0)
                    if battery1_width > 0:
                        pygame.draw.rect(self.simulator.screen, battery1_color, 
                                    (380, y_offset + 2, battery1_width, bar_height))
                    
                    # Right battery bar  
                    battery2_width = int((animat.batteries[1] / settings.BATTERY_MAX) * bar_width)
                    pygame.draw.rect(self.simulator.screen, (200, 200, 200), 
                                (380, y_offset + 10, bar_width, bar_height))
                    battery2_color = (0, 255, 0) if battery2_pct > 30 else (255, 165, 0) if battery2_pct > 10 else (255, 0, 0)
                    if battery2_width > 0:
                        pygame.draw.rect(self.simulator.screen, battery2_color, 
                                    (380, y_offset + 10, battery2_width, bar_height))
                
                y_offset += 20
                
        else:
            # Game over screen using cached surfaces and text
            game_over_rect = self.ui_surfaces['game_over_overlay'].get_rect(
                center=(self.simulator.width // 2, self.simulator.height // 2)
            )
            self.simulator.screen.blit(self.ui_surfaces['game_over_overlay'], game_over_rect)
            
            # Use cached game over text
            game_over_rect = self.text_cache["game_over"].get_rect(
                center=(self.simulator.width // 2, self.simulator.height // 2 - 30)
            )
            restart_rect = self.text_cache["restart_prompt"].get_rect(
                center=(self.simulator.width // 2, self.simulator.height // 2)
            )
            
            # Show final survival times
            final_stats = "Final Survival Times:"
            final_surface = self._get_cached_text("final_stats", final_stats, self.simulator.font, (0, 0, 0))
            final_rect = final_surface.get_rect(center=(self.simulator.width // 2, self.simulator.height // 2 + 20))
            
            self.simulator.screen.blit(self.text_cache["game_over"], game_over_rect)
            self.simulator.screen.blit(self.text_cache["restart_prompt"], restart_rect)
            self.simulator.screen.blit(final_surface, final_rect)
            
            # Show individual agent survival times
            y_offset = self.simulator.height // 2 + 40
            for i, animat in enumerate(self.animats):
                time_text = f"Agent {i + 1}: {animat.survival_time:.1f}s"
                time_surface = self._get_cached_text(f"final_time_{i}", time_text, self.simulator.font, animat.color)
                time_rect = time_surface.get_rect(center=(self.simulator.width // 2, y_offset))
                self.simulator.screen.blit(time_surface, time_rect)
                y_offset += 18
    
    def restart_game(self):
        """Restart the multi-agent manual mode game."""
        # Clear pending respawns
        self.pending_food_respawns.clear()
        self.pending_water_respawns.clear()
        self.food_respawn_timer = 0.0
        self.water_respawn_timer = 0.0
        
        # Clear target positions
        self.target_positions.clear()
        
        # Clear dynamic text cache
        keys_to_remove = [k for k in self.text_cache.keys() if k.startswith(("dynamic_", "agent_", "final_"))]
        for key in keys_to_remove:
            del self.text_cache[key]
        
        # Reinitialize environment
        self.simulator.environment.initialize_random_environment()
        
        # Create new animats
        self.animats = self.create_multiple_animats()
        self.active_animat_index = 0
        
        # Reset manual mode
        self.manual_mode = False
        self.last_manual_input = time.time()
        
        # Follow the first animat
        if self.animats:
            self.simulator.follow_animat(self.animats[0])
        
        print(f"Multi-agent game restarted with {self.num_agents} agents!")
    
    def run(self):
        """Run the multi-agent manual mode simulation with improved performance."""
        self.is_running = True
        
        # Initialize the environment
        self.simulator.environment.initialize_random_environment()
        
        # Create the animats
        self.animats = self.create_multiple_animats()
        
        # Initialize manual mode state
        self.manual_mode = False
        self.last_manual_input = time.time()
        
        # Follow the first animat with camera
        if self.animats:
            self.simulator.follow_animat(self.animats[0])
        
        print(f"Multi-agent manual mode started with {self.num_agents} agents.")
        print("Use WASD or arrow keys to control the active agent.")
        print("Use 1-8 keys or TAB to switch between agents.")
        print("Agents will automatically survive when not controlled.")
        
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
                    elif event.key == pygame.K_r and self.all_animats_dead():
                        self.restart_game()
                    elif event.key == pygame.K_TAB:
                        # Check for shift modifier
                        if pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]:
                            self.switch_to_previous_active_animat()
                        else:
                            self.switch_to_next_active_animat()
                    # Number keys for direct agent switching
                    elif pygame.K_1 <= event.key <= pygame.K_8:
                        agent_index = event.key - pygame.K_1
                        if agent_index < len(self.animats) and self.animats[agent_index].active:
                            self.switch_active_animat(agent_index)
            
            # Get current key states
            keys = pygame.key.get_pressed()
            
            # Handle manual input for active animat
            self.handle_manual_input(keys, dt)
            
            # Run automatic behavior for all animats
            for animat in self.animats:
                self.automatic_behavior(animat, dt)
            
            # Update all animats
            self.update_animats(dt)
            
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
        
        print("Multi-agent manual mode ended.")