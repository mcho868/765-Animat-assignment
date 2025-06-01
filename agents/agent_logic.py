"""
Genetic algorithm implementation for evolving animats.
"""
import numpy as np
import random
from config import settings
from agents.base_agent import Animat
from core.environment import Environment

class GeneticAlgorithm:
    """Genetic algorithm for evolving animat genomes."""
    
    def __init__(self):
        """Initialize the genetic algorithm."""
        self.population_size = settings.POPULATION_SIZE
        self.mutation_rate = settings.MUTATION_RATE
        self.crossover_rate = settings.CROSSOVER_RATE
        self.tournament_size = settings.TOURNAMENT_SIZE
        self.population = []
        self.fitnesses = []
        self.best_genome = None
        self.best_fitness = 0
        self.generation = 0
        
    def initialize_population(self):
        """Initialize a random population of genomes."""
        self.population = []
        
        for _ in range(self.population_size):
            # Create a random animat (this generates a random genome)
            animat = Animat((0, 0))
            # Add its genome to the population
            self.population.append(animat.genome)
            
        self.fitnesses = [0] * self.population_size

    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_population(self, population):
        self.population = population

        self.population_size = len(population)
            
    # Maybe sample without repitition - Vince
    def tournament_selection(self):
        """Select a genome using tournament selection.
        
        Returns:
            Selected genome
        """
        # Randomly choose tournament_size individuals
        tournament_indices = random.sample(range(self.population_size), self.tournament_size)
        
        # Find the best individual in the tournament
        best_idx = tournament_indices[0]
        best_fitness = self.fitnesses[best_idx]
        
        for idx in tournament_indices[1:]:
            if self.fitnesses[idx] > best_fitness:
                best_idx = idx
                best_fitness = self.fitnesses[idx]
                
        return self.population[best_idx].copy()
        
    def crossover(self, parent1, parent2):
        """Perform crossover between two parent genomes.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            Two child genomes
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        # One-point crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        
        # Enforce thresh2 >= thresh1 constraint after crossover for both children
        for child in [child1, child2]:
            for i in range(0, settings.NUM_LINKS * settings.LINK_PARAM_COUNT, settings.LINK_PARAM_COUNT):
                thresh1_pos = i + 2  # Position of thresh1
                thresh2_pos = i + 4  # Position of thresh2
                
                if child[thresh2_pos] < child[thresh1_pos]:
                    if child[thresh1_pos] >= 98:
                        child[thresh1_pos] = child[thresh1_pos] - random.randint(1,5)
                    child[thresh2_pos] = min(99, child[thresh1_pos] + random.randint(1, 10))

        return child1, child2
        
    def mutate(self, genome):
        """Mutate a genome with the given mutation rate.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        mutated_genome = genome.copy()
        
        for i in range(len(mutated_genome)):
            # Apply mutation with probability mutation_rate
            if random.random() < self.mutation_rate:
                # For other genes, add a random value between -10 and 10
                random_value = random.randint(-10, 10)
                while random_value == 0:
                    random_value = random.randint(-10, 10)
                mutated_genome[i] += random_value
                # Keep values in appropriate ranges
                mutated_genome[i] = mutated_genome[i]%100 # Wrap around
                # mutated_genome[i] = max(0, min(99, mutated_genome[i]))
                    
        # Enforce thresh2 >= thresh1 constraint after mutation
        for i in range(0, settings.NUM_LINKS * settings.LINK_PARAM_COUNT, settings.LINK_PARAM_COUNT):
            thresh1_pos = i + 2  # Position of thresh1
            thresh2_pos = i + 4  # Position of thresh2
            
            if mutated_genome[thresh2_pos] < mutated_genome[thresh1_pos]:
                if mutated_genome[thresh1_pos] >= 98:
                    mutated_genome[thresh1_pos] = mutated_genome[thresh1_pos] - random.randint(1, 5)
                mutated_genome[thresh2_pos] = min(99, mutated_genome[thresh1_pos] + random.randint(1, 10))

                    
        return mutated_genome
        
    def evolve_generation(self):
        """Evolve one generation of the population.
        
        Returns:
            List of new genomes
        """
        new_population = []
        
        # Always include the best genome (elitism)
        if self.best_genome is not None:
            new_population.append(self.best_genome.copy())
        
        # Generate new individuals until we reach population_size
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutate
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
                
        self.population = new_population
        self.generation += 1
        
        return new_population
        
    def get_best_genome(self):
        """Get the best genome found so far.
        
        Returns:
            Best genome and its fitness
        """
        return self.best_genome, self.best_fitness
        
def simulate_animat(genome, max_steps=settings.ANIMAT_MAX_LIFESPAN):
    """Simulate an animat with the given genome to evaluate fitness.
    
    Args:
        genome: Genome to evaluate
        max_steps: Maximum simulation steps
        
    Returns:
        Fitness score
    """
    # Create environment
    env = Environment()
    env.initialize_random_environment()
    
    # Create animat with the genome at a random spawn position
    spawn_position = env.get_random_spawn_position()
    animat = Animat(spawn_position, genome)
    env.add_entity(animat)
    
    # Run simulation
    step = 0
    while animat.active and step < max_steps:
        env.update(1)  # Time step per update
        step += 1
        
    # Calculate fitness
    return animat.get_fitness() 