import random
import time
import numpy as np

from functions import schaffer,sphere,rastrigin,rosenbrock

# ABC parameters
population_size = 50  # Number of solutions in the population
max_iter = 1000       # Number of iterations
dim = 2               # Dimension of the problem (can be adjusted)
lower_bound = -5.12   # Lower bound for the search space
upper_bound = 5.12    # Upper bound for the search space

# Initialize population
def initialize_population():
    return np.random.uniform(lower_bound, upper_bound, (population_size, dim))

# Evaluate solutions in population
def evaluate_population(population, func):
    return np.array([func(individual) for individual in population])

# ABC main loop
def abc_algorithm(func):
    population = initialize_population()
    fitness = evaluate_population(population, func)
    best_solution = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    
    # Tracking for performance evaluation
    start_time = time.time()
    
    for iteration in range(max_iter):
        for i in range(population_size):
            # Produce a new solution by modifying an existing one
            phi = np.random.uniform(-1, 1, dim)
            k = random.choice([j for j in range(population_size) if j != i])
            new_solution = population[i] + phi * (population[i] - population[k])
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            
            # Evaluate new solution
            new_fitness = func(new_solution)
            
            # Greedy selection
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        # Memorize the best solution
        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_solution = population[np.argmin(fitness)]
        
    end_time = time.time()
    
    return best_solution, best_fitness, end_time - start_time

# Running the algorithm on continuous functions
for func, name in [(schaffer, "Schaffer"), (sphere, "Sphere"), (rastrigin, "Rastrigin"), (rosenbrock, "Rosenbrock")]:
    best_solution, best_fitness, elapsed_time = abc_algorithm(func)
    print(f"{name} Function: Best Fitness = {best_fitness}, Time = {elapsed_time:.2f}s")
