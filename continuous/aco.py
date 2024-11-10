import numpy as np
import time  # Import time module

# ACO parameters
ANT_COUNT = 50
ITERATIONS = 300
EVAPORATION_RATE = 0.5
ALPHA = 1  # Influence of pheromone
BETA = 2  # Influence of heuristic information (distance)

def create_pheromone_map(n, dims):
    return np.ones((n, dims))  # Initialize pheromone levels to 1

def fitness_function(x, func):
    return func(x)  # For ACO, minimize the function value directly

def choose_path(pheromone_map, distance_map, alpha=ALPHA, beta=BETA):
    pheromone = pheromone_map ** alpha
    heuristic = (1 / (distance_map + 1e-6)) ** beta
    probability = pheromone * heuristic
    probability /= np.sum(probability)
    return np.random.choice(len(probability), p=probability)

def update_pheromone(pheromone_map, ant_solutions, decay=EVAPORATION_RATE):
    pheromone_map *= (1 - decay)
    for ant in ant_solutions:
        pheromone_map += ant  # Add pheromone left by the ant

def ant_colony_optimization(func, dims, bound, iterations=ITERATIONS, ant_count=ANT_COUNT):
    start_time = time.time()  # Record the start time
    
    # Step 1: Initialize pheromone map
    pheromone_map = create_pheromone_map(ant_count, dims)
    
    best_solution = None
    best_fitness = float('inf')
    
    for it in range(iterations):
        ant_solutions = []
        ant_fitness = []
        
        # Step 2: Let ants search for food
        for _ in range(ant_count):
            solution = np.random.uniform(-bound, bound, dims)  # Ant chooses initial solution randomly
            fitness = fitness_function(solution, func)
            ant_solutions.append(solution)
            ant_fitness.append(fitness)
        
        # Step 3: Update pheromone map based on ants' solutions
        update_pheromone(pheromone_map, ant_solutions)

        # Step 4: Track the best solution
        best_ant_idx = np.argmin(ant_fitness)  # Minimize fitness
        if ant_fitness[best_ant_idx] < best_fitness:
            best_solution = ant_solutions[best_ant_idx]
            best_fitness = ant_fitness[best_ant_idx]
        
        # print(f"Iteration {it + 1}: Best fitness = {best_fitness}, Best solution = {best_solution}")
    
    elapsed_time = time.time() - start_time  # Calculate the elapsed time
    return best_solution, best_fitness, elapsed_time  # Return best solution, fitness, and elapsed time
