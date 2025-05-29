"""
Configuration settings for the Animat model simulation.
"""

#Window settings
WINDOW_WIDTH = 2000
WINDOW_HEIGHT = 2000
WINDOW_TITLE = "Animat Evolution Simulation"
FPS = 60

#Environment settings
ENV_SIZE = 200  # 200x200 area
OBJECT_PLACEMENT_PADDING = 10
FOOD_COUNT = 3
WATER_COUNT = 3
# TRAP_COUNT = 9
TRAP_COUNT = 3

#Animat settings
ANIMAT_SIZE = 5
SOURCE_SIZE = 16
ANIMAT_MAX_SPEED = 2.8
SENSOR_RANGE = 100
BATTERY_MAX = 200  # Paper specifies maximum and initial level of 200
BATTERY_DECAY_RATE = 1.0  # Paper specifies decreases by 1 each time step
ANIMAT_MAX_LIFESPAN = 800 # Maximum simulation steps for fitness evaluation

#Genetic algorithm settings
POPULATION_SIZE = 100
MUTATION_RATE = 0.04  # 4% per bit (Note: current mutation is per-gene)
CROSSOVER_RATE = 0.8
NUM_GENERATIONS = 200
TOURNAMENT_SIZE = 5

#Sensorimotor link settings
LINK_PARAM_COUNT = 2  # Only weight and bias per link
NUM_LINKS = 9  # Only 9 links need to be genetically specified due to left-right symmetry
GENOTYPE_SIZE = NUM_LINKS * LINK_PARAM_COUNT + 2  # 9*2 + 2 = 20 total integers
NUM_SENSORS = 6  # Default, will be updated

def update_sensor_settings(num_animats):
    if num_animats == 1:
        num_sensors = 6
    else:
        num_sensors = 8
    global NUM_LINKS, GENOTYPE_SIZE, NUM_SENSORS
    NUM_SENSORS = num_sensors
    NUM_LINKS = num_sensors * 3 // 2  # Only left-side links are encoded, right are mirrored
    GENOTYPE_SIZE = NUM_LINKS * LINK_PARAM_COUNT + 2  # Update for new LINK_PARAM_COUNT

#Simulation mode
HEADLESS_MODE = True  # Set to True to run simulations without visualization
LOGGING_ENABLED = False
SIMULATION=1000
#Random seed for reproducibility (set to None for random)
RANDOM_SEED = 42