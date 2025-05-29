"""
Environment module for the Animat simulation.
Defines the 2D world, objects, and physics.
"""
import numpy as np
import random
from enum import Enum, auto
from config import settings

class EntityType(Enum):
    """Types of entities in the environment."""
    FOOD = auto()
    WATER = auto()
    TRAP = auto()
    ANIMAT = auto()

class Entity:
    """Base class for all entities in the environment."""
    next_id = 0
    
    def __init__(self, position, entity_type, radius=5):
        """Initialize an entity.
        
        Args:
            position: Tuple (x, y) coordinates
            entity_type: EntityType enum value
            radius: Collision radius
        """
        self.id = Entity.next_id
        Entity.next_id += 1
        self.position = np.array(position, dtype=float)
        self.type = entity_type
        self.radius = radius
        self.active = True

class FoodSource(Entity):
    """Food source that restores battery 1."""
    def __init__(self, position):
        super().__init__(position, EntityType.FOOD, radius=settings.SOURCE_SIZE)

class WaterSource(Entity):
    """Water source that restores battery 2."""
    def __init__(self, position):
        super().__init__(position, EntityType.WATER, radius=settings.SOURCE_SIZE)

class Trap(Entity):
    """Trap that kills animats on collision."""
    def __init__(self, position):
        super().__init__(position, EntityType.TRAP, radius=settings.SOURCE_SIZE)

class Environment:
    """The 2D environment containing all entities."""

    def place_entity_without_overlap(self, entity_class):
        """Place an entity in a random location avoiding overlap with existing entities."""
        max_attempts = 1000
        padding = settings.OBJECT_PLACEMENT_PADDING
        radius = settings.SOURCE_SIZE
        for _ in range(max_attempts):
            x = random.randint(padding, self.width - padding)
            y = random.randint(padding, self.height - padding)
            position = np.array((x, y), dtype=float)

            # Check for overlap
            overlaps = False
            for e in self.entities:
                dist = np.linalg.norm(position - e.position)
                if dist < (radius + e.radius + 2):  # 2 is small buffer distance
                    overlaps = True
                    break

            if not overlaps:
                return entity_class(position)

        raise RuntimeError(f"Couldn't place entity without overlap after {max_attempts} attempts.")
    
    def __init__(self, width=settings.ENV_SIZE, height=settings.ENV_SIZE):
        """Initialize the environment.
        
        Args:
            width: Width of the environment
            height: Height of the environment
        """
        self.width = width
        self.height = height
        self.entities = []
        self.animats = []
        self.food_sources = []
        self.water_sources = []
        self.traps = []
        
    def add_entity(self, entity):
        """Add an entity to the environment."""
        self.entities.append(entity)
        
        # Also add to type-specific lists for faster access
        if entity.type == EntityType.FOOD:
            self.food_sources.append(entity)
        elif entity.type == EntityType.WATER:
            self.water_sources.append(entity)
        elif entity.type == EntityType.TRAP:
            self.traps.append(entity)
        elif entity.type == EntityType.ANIMAT:
            self.animats.append(entity)
    
    def initialize_random_environment(self):
        """Initialize the environment with random placement of objects."""
        # Clear any existing entities
        self.entities = []
        self.animats = []
        self.food_sources = []
        self.water_sources = []
        self.traps = []
        
        padding = settings.OBJECT_PLACEMENT_PADDING

        # Add food sources
        for _ in range(settings.FOOD_COUNT):
            food = self.place_entity_without_overlap(FoodSource)
            self.add_entity(food)

        # Add water sources
        for _ in range(settings.WATER_COUNT):
            water = self.place_entity_without_overlap(WaterSource)
            self.add_entity(water)

        # Add traps
        for _ in range(settings.TRAP_COUNT):
            trap = self.place_entity_without_overlap(Trap)
            self.add_entity(trap)
    
    def check_collision(self, position, radius, exclude_entity=None):
        """Check if a position collides with any entity.
        
        Args:
            position: (x, y) tuple to check
            radius: Collision radius
            exclude_entity: Optional entity to exclude from collision check
            
        Returns:
            Tuple (collision_detected, entity_collided_with)
        """
        for entity in self.entities:
            if entity == exclude_entity or not entity.active:
                continue
                
            # Simple distance-based collision
            distance = np.linalg.norm(np.array(position) - entity.position)
            if distance < (radius + entity.radius):
                return True, entity
                
        return False, None
    
    def get_sensor_readings(self, animat):
        """Get sensor readings for an animat based on its position.
        
        Args:
            animat: The animat to get sensor readings for
            
        Returns:
            Dict of sensor readings for each sensor type and side
        """
        readings = {
            'food_left': 0.0, 'food_right': 0.0,
            'water_left': 0.0, 'water_right': 0.0,
            'trap_left': 0.0, 'trap_right': 0.0
        }
        
        # Define the range vector for each side (assuming animat has direction)
        # Rotate the direction vector 90 degrees for left/right
        left_vec = np.array([-animat.direction[1], animat.direction[0]])
        right_vec = np.array([animat.direction[1], -animat.direction[0]])
        
        # Calculate readings for each entity type
        for entity in self.entities:
            if not entity.active or entity.type == EntityType.ANIMAT:
                continue
                
            # Vector from animat to entity
            vec_to_entity = entity.position - animat.position
            distance = np.linalg.norm(vec_to_entity)
            
            # Skip if outside sensor range
            if distance > settings.SENSOR_RANGE:
                continue
                
            # Normalize direction to entity
            if distance > 0:
                direction_to_entity = vec_to_entity / distance
            else:
                continue  # Skip if at same position
                
            # Calculate dot product with left and right vectors to determine side
            left_dot = np.dot(direction_to_entity, left_vec)
            right_dot = np.dot(direction_to_entity, right_vec)
            
            # Calculate sensor value (stronger when closer and more directly to one side)
            # Sensor output is between 0 and 100, with 100 being at the source (distance=0)
            # and decreasing as distance increases
            sensor_value = max(0, 100 * (1 - distance / settings.SENSOR_RANGE))
            
            # Apply to the appropriate sensor based on entity type and side
            if entity.type == EntityType.FOOD:
                if left_dot > 0:
                    readings['food_left'] = max(readings['food_left'], sensor_value * 1.2 * left_dot)
                if right_dot > 0:
                    readings['food_right'] = max(readings['food_right'], sensor_value * 1.2 * right_dot)
            
            elif entity.type == EntityType.WATER:
                if left_dot > 0:
                    readings['water_left'] = max(readings['water_left'], sensor_value * 1.2 * left_dot)
                if right_dot > 0:
                    readings['water_right'] = max(readings['water_right'], sensor_value * 1.2 * right_dot)
            
            elif entity.type == EntityType.TRAP:
                if left_dot > 0:
                    readings['trap_left'] = max(readings['trap_left'], sensor_value * 1.2 * left_dot)
                if right_dot > 0:
                    readings['trap_right'] = max(readings['trap_right'], sensor_value * 1.2 * right_dot)
        
        return readings
    
    def respawn_entity(self, entity):
        """Respawn an entity at a random location.
        
        Args:
            entity: The entity to respawn
        """
        # Generate new random position
        padding = settings.OBJECT_PLACEMENT_PADDING
        new_position = (
            random.randint(padding, self.width - padding),
            random.randint(padding, self.height - padding)
        )
        
        # Update entity position
        entity.position = np.array(new_position, dtype=float)
        
    def wrap_position(self, entity):
        """Wrap entity position if it goes outside environment boundaries.
        
        Args:
            entity: The entity to wrap
        """
        # Wrap x-coordinate
        if entity.position[0] < 0:
            entity.position[0] = self.width
        elif entity.position[0] > self.width:
            entity.position[0] = 0
            
        # Wrap y-coordinate
        if entity.position[1] < 0:
            entity.position[1] = self.height
        elif entity.position[1] > self.height:
            entity.position[1] = 0
    
    def update(self, dt):
        """Update the environment for one timestep.
        
        Args:
            dt: Time delta in seconds
        """
        # Update animats (movement, collisions, battery depletion)
        for animat in self.animats[:]:  # Use a copy to allow removal
            if not animat.active:
                continue
                
            # Update animat position
            animat.update(dt, self)
            
            # Wrap position if animat moves outside boundaries
            self.wrap_position(animat)
            
            # Check for collisions with entities
            collision, entity = self.check_collision(animat.position, animat.radius, animat)
            
            if collision:
                if entity.type == EntityType.FOOD:
                    # Replenish battery 1
                    animat.batteries[0] = settings.BATTERY_MAX
                    # Make food disappear and reappear at random location
                    self.respawn_entity(entity)
                    animat.food_collected += 1
                    
                elif entity.type == EntityType.WATER:
                    # Replenish battery 2
                    animat.batteries[1] = settings.BATTERY_MAX
                    # Make water disappear and reappear at random location
                    self.respawn_entity(entity)
                    animat.water_collected += 1
                    
                elif entity.type == EntityType.TRAP:
                    # Animat dies
                    animat.active = False
                    animat.killed_by_trap = True
                    
            # Check if batteries are depleted
            if animat.batteries[0] <= 0 and animat.batteries[1] <= 0:
                animat.active = False 