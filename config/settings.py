"""
Configuration settings for the Animat model simulation.
"""

# Window settings
WINDOW_WIDTH = 1200  # 0 means use full screen width
WINDOW_HEIGHT = 800 # 0 means use full screen height
WINDOW_TITLE = "Animat Evolution Simulation"
FPS = 60

# Environment settings
ENV_SIZE = 200  # 200x200 area
OBJECT_PLACEMENT_PADDING = 10
FOOD_COUNT = 6
WATER_COUNT = 6
TRAP_COUNT = 2

# Animat settings
ANIMAT_SIZE = 10
ANIMAT_MAX_SPEED = 5.0
SENSOR_RANGE = 100
BATTERY_MAX = 100
BATTERY_DECAY_RATE = 1  # Per timestep

# Genetic algorithm settings
POPULATION_SIZE = 50
MUTATION_RATE = 0.004  # 0.4% per bit
CROSSOVER_RATE = 0.9
NUM_GENERATIONS = 100
TOURNAMENT_SIZE = 3

# Sensorimotor link settings
LINK_PARAM_COUNT = 9
NUM_LINKS = 18  # 3 sensors * 2 sides * 3 parallel links
GENOTYPE_SIZE = NUM_LINKS * LINK_PARAM_COUNT + 2  # +2 for sigmoid thresholds

# Simulation mode
HEADLESS_MODE = False  # Set to True to run simulations without visualization
LOGGING_ENABLED = True

# Random seed for reproducibility (set to None for random)
RANDOM_SEED = 42 