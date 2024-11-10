import numpy as np
import random

# Fitness function to evaluate the total distance of the path (including depot)
def fitness(solution, distance_matrix, depot=0):
    total_distance = 0
    # Ensure the path starts and ends at the depot
    for i in range(len(solution) - 1):
        total_distance += distance_matrix[solution[i], solution[i + 1]]
    total_distance += distance_matrix[solution[-1], depot]  # Return to depot
    return total_distance

# Initialize the population of solutions
def initialize_population(population_size, num_cities, depot):
    population = []
    for _ in range(population_size):
        solution = np.random.permutation(num_cities - 1).tolist() + [depot]
        solution[0] = solution[-1] = depot
        population.append(solution)
    return population

# Generate new solution by swapping two cities
def mutate(solution, depot=0):
    size = len(solution)
    idx1, idx2 = random.sample(range(1, size - 1), 2)  # Exclude depot
    solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
    return solution

# ABC algorithm for TSP
def abc_algorithm_tsp(distance_matrix, population_size=100, iterations=100, depot=0):
    num_cities = len(distance_matrix)
    
    # Initialize the population of solutions
    population = initialize_population(population_size, num_cities, depot)
    
    # Evaluate the fitness of the initial population
    fitness_values = np.array([fitness(solution, distance_matrix, depot) for solution in population])

    best_solution = population[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)

    # Store fitness history for plotting
    fitness_over_time = []

    # Main loop for the ABC algorithm
    for _ in range(iterations):
        # Employed bees phase (local search)
        for i in range(population_size):
            new_solution = mutate(population[i], depot)
            new_fitness = fitness(new_solution, distance_matrix, depot)
            if new_fitness < fitness_values[i]:
                population[i] = new_solution
                fitness_values[i] = new_fitness

        # Onlooker bees phase (select best solutions)
        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]
        
        # Record the best fitness value for plotting
        fitness_over_time.append(best_fitness)

    return best_solution, best_fitness, fitness_over_time
