import random
import time
import numpy as np

# ABC parameters
population_size = 100  # Number of solutions in the population
max_iter = 150       # Number of iterations

# Initialize population
def initialize_population(dims, bound):
    return np.random.uniform(-bound, bound, (population_size, dims))

# Evaluate solutions in population
def evaluate_population(population, func):
    return np.array([func(individual) for individual in population])

# ABC main loop
def abc_algorithm(func, dims, bound, iterations=max_iter, callback=None):
    population = initialize_population(dims, bound)
    fitness = evaluate_population(population, func)
    best_solution = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    
    # Tracking for performance evaluation
    start_time = time.time()
    
    for iteration in range(iterations):
        for i in range(population_size):
            # Produce a new solution by modifying an existing one
            phi = np.random.uniform(-1, 1, dims)
            k = random.choice([j for j in range(population_size) if j != i])
            new_solution = population[i] + phi * (population[i] - population[k])
            new_solution = np.clip(new_solution, -bound, bound)
            
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

        # Call the callback if provided, and check if it wants to stop early
        if callback and not callback(best_fitness, iteration):
            break
        
    end_time = time.time()
    
    return best_solution, best_fitness, end_time - start_time
