"""
Configuration settings for the Animat model simulation.
"""

#Window settings
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
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
SENSOR_RANGE = 200
BATTERY_MAX = 200  # Paper specifies maximum and initial level of 200
BATTERY_DECAY_RATE = 0.3  # Paper specifies decreases by 1 each time step
ANIMAT_MAX_LIFESPAN = 800 # Maximum simulation steps for fitness evaluation

#Genetic algorithm settings
POPULATION_SIZE = 100
MUTATION_RATE = 0.04  # 4% per bit (Note: current mutation is per-gene)
CROSSOVER_RATE = 0.5
NUM_GENERATIONS = 3
TOURNAMENT_SIZE = 5

#Sensorimotor link settings
LINK_PARAM_COUNT = 9
NUM_LINKS = 9  # Only 9 links need to be genetically specified due to left-right symmetry
GENOTYPE_SIZE = NUM_LINKS * LINK_PARAM_COUNT + 2  # 9*9 + 2 = 83 total integers

#Simulation mode
HEADLESS_MODE = True  # Set to True to run simulations without visualization
LOGGING_ENABLED = False
SIMULATION=1000
#Random seed for reproducibility (set to None for random)
RANDOM_SEED = None

SIMULATION_END_PERCENTAGE = 100
BATTERY_FITNESS_MODE = True