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
        
    def evaluate_fitness(self, simulate_function):
        """Evaluate fitness of each genome in the population.
        
        Args:
            simulate_function: Function that takes a genome and returns its fitness
        """
        self.fitnesses = []
        
        for genome in self.population:
            fitness = simulate_function(genome)
            self.fitnesses.append(fitness)
            
        # Track best genome
        best_idx = np.argmax(self.fitnesses)
        if self.best_genome is None or self.fitnesses[best_idx] > self.best_fitness:
            self.best_genome = self.population[best_idx].copy()
            self.best_fitness = self.fitnesses[best_idx]
            
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
                # Battery genes (every 9th gene) can only be 0 or 1
                if (i % settings.LINK_PARAM_COUNT == 8) and (i < settings.NUM_LINKS * settings.LINK_PARAM_COUNT):
                    mutated_genome[i] = 1 - mutated_genome[i]  # Flip 0 to 1 or 1 to 0
                else:
                    # For other genes, add a random value between -10 and 10
                    mutated_genome[i] += random.randint(-10, 10)
                    # Keep values in appropriate ranges
                    mutated_genome[i] = max(-50, min(50, mutated_genome[i]))
                    
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
        
def simulate_animat(genome, max_steps=5000):
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
    
    # Create animat with the genome
    animat = Animat((env.width/2, env.height/2), genome)
    env.add_entity(animat)
    
    # Run simulation
    step = 0
    while animat.active and step < max_steps:
        env.update(0.1)  # 0.1 seconds per step
        step += 1
        
    # Calculate fitness
    return animat.get_fitness() 