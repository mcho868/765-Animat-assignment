"""
Environment module for the Animat simulation.
Defines the 2D unbounded world with objects placed in a 200x200 area.
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
    """The 2D unbounded environment with objects placed in a 200x200 area."""
    
    def __init__(self, object_area_size=settings.ENV_SIZE):
        """Initialize the unbounded environment.
        
        Args:
            object_area_size: Size of the area where objects are placed (200x200 as per paper)
        """
        self.object_area_size = object_area_size  # Objects appear within this area
        # No width/height limits - environment is unbounded
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
        """Initialize the environment with random placement of objects within the 200x200 area."""
        # Clear any existing entities
        self.entities = []
        self.animats = []
        self.food_sources = []
        self.water_sources = []
        self.traps = []
        
        padding = settings.OBJECT_PLACEMENT_PADDING
        
        # Add food sources within the object placement area
        for _ in range(settings.FOOD_COUNT):
            position = (
                random.randint(padding, self.object_area_size - padding),
                random.randint(padding, self.object_area_size - padding)
            )
            self.add_entity(FoodSource(position))
        
        # Add water sources within the object placement area
        for _ in range(settings.WATER_COUNT):
            position = (
                random.randint(padding, self.object_area_size - padding),
                random.randint(padding, self.object_area_size - padding)
            )
            self.add_entity(WaterSource(position))
        
        # Add traps within the object placement area
        for _ in range(settings.TRAP_COUNT):
            position = (
                random.randint(padding, self.object_area_size - padding),
                random.randint(padding, self.object_area_size - padding)
            )
            self.add_entity(Trap(position))
    
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
        """Respawn an entity at a random location within the object placement area.
        
        Args:
            entity: The entity to respawn
        """
        # Generate new random position within the object placement area
        padding = settings.OBJECT_PLACEMENT_PADDING
        new_position = (
            random.randint(padding, self.object_area_size - padding),
            random.randint(padding, self.object_area_size - padding)
        )
        
        # Update entity position
        entity.position = np.array(new_position, dtype=float)
        
    def get_random_spawn_position(self, radius=settings.ANIMAT_SIZE, max_attempts=50):
        """Get a random spawn position that doesn't overlap with existing entities.
        
        Args:
            radius: Radius of the entity to spawn
            max_attempts: Maximum attempts to find a non-overlapping position
            
        Returns:
            Tuple (x, y) of a valid spawn position
        """
        padding = settings.OBJECT_PLACEMENT_PADDING
        
        for _ in range(max_attempts):
            # Generate random position within the object placement area
            x = random.randint(padding + radius, self.object_area_size - padding - radius)
            y = random.randint(padding + radius, self.object_area_size - padding - radius)
            position = (x, y)
            
            # Check if this position collides with any existing entity
            collision, _ = self.check_collision(position, radius)
            
            if not collision:
                return position
        
        # If we couldn't find a non-overlapping position, return a position at the center
        # (this ensures we always return a valid position even in crowded environments)
        return (self.object_area_size / 2, self.object_area_size / 2)
    
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
            
            # No position wrapping - environment is unbounded!
            # Animats can move freely in unlimited space
            
            # Check for collisions with entities
            collision, entity = self.check_collision(animat.position, animat.radius, animat)
            
            if collision:
                if entity.type == EntityType.FOOD:
                    # Replenish battery 1
                    animat.batteries[0] = settings.BATTERY_MAX
                    # Make food disappear and reappear at random location within object area
                    self.respawn_entity(entity)
                    
                elif entity.type == EntityType.WATER:
                    # Replenish battery 2
                    animat.batteries[1] = settings.BATTERY_MAX
                    # Make water disappear and reappear at random location within object area
                    self.respawn_entity(entity)
                    
                elif entity.type == EntityType.TRAP:
                    # Animat dies
                    animat.active = False
                    animat.batteries[0] = 0
                    animat.batteries[1] = 0
                    
            # Check if batteries are depleted
            if animat.batteries[0] <= 0 and animat.batteries[1] <= 0:
                animat.active = False 