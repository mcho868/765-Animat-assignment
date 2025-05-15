"""
Configuration settings for the Braitenberg-inspired simulation environment.
Based on the 765 Proposal and research paper.
"""

# Window settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Braitenberg Vehicle Simulation"
FPS = 60

# Environment settings
GRID_SIZE = 100
OBSTACLE_COUNT = 10
LIGHT_SOURCE_COUNT = 5
FOOD_SOURCE_COUNT = 5
AGENT_COUNT = 20

# Agent settings
AGENT_SIZE = 20
AGENT_SPEED = 2
AGENT_TURN_RATE = 0.05
AGENT_VISION_RANGE = 100
AGENT_SENSOR_ANGLES = [-45, 0, 45]  # Degrees relative to heading
MAX_STRESS = 100
STRESS_DECAY_RATE = 0.1

# Genetic algorithm settings
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7

# Simulation mode
HEADLESS_MODE = False  # Set to True to run simulations without visualization
LOGGING_ENABLED = True 