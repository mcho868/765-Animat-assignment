"""
Configuration settings for the Animat model simulation.
"""

#Window settings
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1400
WINDOW_TITLE = "Animat Evolution Simulation"
FPS = 60

#Environment settings
ENV_SIZE = 200  # 200x200 area
OBJECT_PLACEMENT_PADDING = 10
FOOD_COUNT = 3
WATER_COUNT = 3
TRAP_COUNT = 9

#Animat settings
ANIMAT_SIZE = 5
SOURCE_SIZE = 16
ANIMAT_MAX_SPEED = 2.8
SENSOR_RANGE = 100
BATTERY_MAX = 200
BATTERY_DECAY_RATE = 100  # Per timestep (adjusted to deplete in 200 steps with dt=0.1)
ANIMAT_MAX_LIFESPAN = 800 # Maximum simulation steps for fitness evaluation

#Genetic algorithm settings
POPULATION_SIZE = 100
MUTATION_RATE = 0.04  # 4% per bit (Note: current mutation is per-gene)
CROSSOVER_RATE = 0.5
NUM_GENERATIONS = 200
TOURNAMENT_SIZE = 2

#Sensorimotor link settings
LINK_PARAM_COUNT = 9
NUM_LINKS = 18  # 3 sensors * 2 sides * 3 parallel links
GENOTYPE_SIZE = NUM_LINKS * LINK_PARAM_COUNT + 2  # +2 for sigmoid thresholds

#Simulation mode
HEADLESS_MODE = True  # Set to True to run simulations without visualization
LOGGING_ENABLED = False
SIMULATION=1000
#Random seed for reproducibility (set to None for random)
RANDOM_SEED = 42