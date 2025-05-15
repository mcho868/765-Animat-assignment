"""
Environment module for setting up the simulation world with resources, obstacles, etc.
Based on the Braitenberg model as described in the proposal.
"""
import random
import numpy as np
from enum import Enum

class EntityType(Enum):
    """Enumeration of entity types in the environment."""
    OBSTACLE = 0
    LIGHT_SOURCE = 1
    FOOD_SOURCE = 2
    AGENT = 3

class Entity:
    """Base class for all entities in the environment."""
    def __init__(self, entity_id, entity_type, position, radius=10):
        self.id = entity_id
        self.type = entity_type
        self.position = position  # (x, y) tuple
        self.radius = radius
        self.active = True

    def update(self, dt):
        """Update the entity's state."""
        pass

class Obstacle(Entity):
    """Static obstacle that blocks agent movement."""
    def __init__(self, entity_id, position, radius=15):
        super().__init__(entity_id, EntityType.OBSTACLE, position, radius)
        
class LightSource(Entity):
    """Light source that agents can sense from a distance."""
    def __init__(self, entity_id, position, intensity=100, radius=5):
        super().__init__(entity_id, EntityType.LIGHT_SOURCE, position, radius)
        self.intensity = intensity
        self.max_range = intensity * 5  # How far the light reaches
        
    def get_intensity_at(self, position):
        """Calculate light intensity at a given position."""
        dist = ((position[0] - self.position[0])**2 + 
                (position[1] - self.position[1])**2)**0.5
        
        # Inverse square law for light intensity
        if dist < self.radius:
            return self.intensity
        elif dist > self.max_range:
            return 0
        else:
            return self.intensity * (self.radius / dist)**2

class FoodSource(Entity):
    """Food source that agents can consume."""
    def __init__(self, entity_id, position, energy=100, radius=8, respawn_time=30):
        super().__init__(entity_id, EntityType.FOOD_SOURCE, position, radius)
        self.max_energy = energy
        self.energy = energy
        self.respawn_time = respawn_time
        self.respawn_timer = 0
        
    def consume(self, amount):
        """Agent consumes some amount of energy from the food source."""
        if not self.active:
            return 0
            
        amount_consumed = min(self.energy, amount)
        self.energy -= amount_consumed
        
        # Deactivate when depleted
        if self.energy <= 0:
            self.active = False
            self.respawn_timer = self.respawn_time
            
        return amount_consumed
        
    def update(self, dt):
        """Update food source state, including respawning."""
        if not self.active:
            self.respawn_timer -= dt
            if self.respawn_timer <= 0:
                self.energy = self.max_energy
                self.active = True

class Environment:
    """Simulation environment containing all entities and physics."""
    def __init__(self, width, height, config=None):
        self.width = width
        self.height = height
        self.config = config
        self.entities = []
        self.agents = []
        self.obstacles = []
        self.light_sources = []
        self.food_sources = []
        self.entity_counter = 0
        
    def add_entity(self, entity):
        """Add an entity to the environment."""
        self.entities.append(entity)
        
        # Add to specific list based on type
        if entity.type == EntityType.AGENT:
            self.agents.append(entity)
        elif entity.type == EntityType.OBSTACLE:
            self.obstacles.append(entity)
        elif entity.type == EntityType.LIGHT_SOURCE:
            self.light_sources.append(entity)
        elif entity.type == EntityType.FOOD_SOURCE:
            self.food_sources.append(entity)
    
    def generate_obstacles(self, count, min_radius=10, max_radius=30):
        """Generate random obstacles in the environment."""
        for _ in range(count):
            valid_position = False
            while not valid_position:
                radius = random.randint(min_radius, max_radius)
                x = random.randint(radius, self.width - radius)
                y = random.randint(radius, self.height - radius)
                position = (x, y)
                
                # Check if the position overlaps with any existing entity
                valid_position = not any(
                    ((x - e.position[0])**2 + (y - e.position[1])**2)**0.5 < (radius + e.radius + 10)
                    for e in self.entities
                )
            
            obstacle = Obstacle(self.entity_counter, position, radius)
            self.entity_counter += 1
            self.add_entity(obstacle)
    
    def generate_light_sources(self, count, intensity_range=(50, 150)):
        """Generate random light sources in the environment."""
        for _ in range(count):
            valid_position = False
            while not valid_position:
                x = random.randint(20, self.width - 20)
                y = random.randint(20, self.height - 20)
                position = (x, y)
                
                # Check if the position overlaps with any existing entity
                valid_position = not any(
                    ((x - e.position[0])**2 + (y - e.position[1])**2)**0.5 < (5 + e.radius + 20)
                    for e in self.entities
                )
            
            intensity = random.randint(*intensity_range)
            light = LightSource(self.entity_counter, position, intensity)
            self.entity_counter += 1
            self.add_entity(light)
            
    def generate_food_sources(self, count, energy_range=(50, 200)):
        """Generate random food sources in the environment."""
        for _ in range(count):
            valid_position = False
            while not valid_position:
                x = random.randint(20, self.width - 20)
                y = random.randint(20, self.height - 20)
                position = (x, y)
                
                # Check if the position overlaps with any existing entity
                valid_position = not any(
                    ((x - e.position[0])**2 + (y - e.position[1])**2)**0.5 < (8 + e.radius + 15)
                    for e in self.entities
                )
            
            energy = random.randint(*energy_range)
            food = FoodSource(self.entity_counter, position, energy)
            self.entity_counter += 1
            self.add_entity(food)
    
    def update(self, dt):
        """Update all entities in the environment."""
        for entity in self.entities:
            entity.update(dt)
    
    def get_light_intensity_at(self, position):
        """Calculate the total light intensity at a given position."""
        total_intensity = 0
        for light in self.light_sources:
            if light.active:
                total_intensity += light.get_intensity_at(position)
        return total_intensity
    
    def get_entities_in_range(self, position, radius):
        """Get all entities within a certain range of a position."""
        in_range = []
        for entity in self.entities:
            dist = ((position[0] - entity.position[0])**2 + 
                    (position[1] - entity.position[1])**2)**0.5
            if dist < radius:
                in_range.append((entity, dist))
        return in_range
    
    def check_collision(self, position, radius, exclude_entity=None):
        """Check if a position with a given radius collides with any entity."""
        for entity in self.entities:
            if entity == exclude_entity:
                continue
                
            if entity.type == EntityType.AGENT or entity.type == EntityType.OBSTACLE:
                dist = ((position[0] - entity.position[0])**2 + 
                        (position[1] - entity.position[1])**2)**0.5
                if dist < radius + entity.radius:
                    return True, entity
        
        # Check boundaries
        if (position[0] < radius or position[0] > self.width - radius or
            position[1] < radius or position[1] > self.height - radius):
            return True, None
            
        return False, None
        
    def consume_food_at(self, position, radius, amount):
        """Agent attempts to consume food at a given position."""
        total_consumed = 0
        for food in self.food_sources:
            if not food.active:
                continue
                
            dist = ((position[0] - food.position[0])**2 + 
                    (position[1] - food.position[1])**2)**0.5
            if dist < radius + food.radius:
                consumed = food.consume(amount)
                total_consumed += consumed
                
        return total_consumed
        
    def initialize_random_environment(self, obstacle_count, light_count, food_count):
        """Initialize the environment with random entities."""
        self.generate_obstacles(obstacle_count)
        self.generate_light_sources(light_count)
        self.generate_food_sources(food_count) 