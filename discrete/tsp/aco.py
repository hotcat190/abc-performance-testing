import numpy as np
import random

def ant_colony_optimization_tsp(distance_matrix, num_ants=100, iterations=100, alpha=1, beta=2, evaporation_rate=0.5, q0=0.9, depot=0):
    # Number of cities
    num_cities = len(distance_matrix)
    
    # Initialize pheromone matrix
    pheromone = np.ones((num_cities, num_cities))  # Pheromone initialization
    best_solution = None
    best_fitness = float('inf')
    
    # List to track best fitness at each iteration for plotting
    fitness_over_time = []

    # Ant construction
    for _ in range(iterations):
        all_solutions = []
        all_fitness = []

        # Each ant constructs a solution
        for ant in range(num_ants):
            solution = construct_solution(distance_matrix, pheromone, alpha, beta, q0, depot)
            fitness = calculate_fitness(solution, distance_matrix)

            # Track the best solution found
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = solution

            all_solutions.append(solution)
            all_fitness.append(fitness)

        # Update pheromone
        pheromone_delta = np.zeros((num_cities, num_cities))
        for i in range(num_ants):
            solution = all_solutions[i]
            fitness = all_fitness[i]
            for i in range(len(solution) - 1):
                pheromone_delta[solution[i], solution[i + 1]] += 1 / fitness

        # Apply pheromone evaporation
        pheromone = pheromone * (1 - evaporation_rate) + pheromone_delta
        
        # Track best fitness for this iteration
        fitness_over_time.append(best_fitness)

    return best_solution, best_fitness, fitness_over_time

# Construct a solution (tour) for an ant
def construct_solution(distance_matrix, pheromone, alpha, beta, q0, depot):
    num_cities = len(distance_matrix)
    visited = [False] * num_cities
    solution = []

    # Start at the depot (city 0)
    current_city = depot
    solution.append(current_city)
    visited[current_city] = True

    for _ in range(num_cities - 1):
        next_city = select_next_city(distance_matrix, pheromone, current_city, visited, alpha, beta, q0)
        solution.append(next_city)
        visited[next_city] = True
        current_city = next_city

    solution.append(depot)  # Return to the depot (city 0)
    return solution

# Select the next city based on pheromone and distance
def select_next_city(distance_matrix, pheromone, current_city, visited, alpha, beta, q0):
    num_cities = len(distance_matrix)
    probabilities = np.zeros(num_cities)
    
    for city in range(num_cities):
        if not visited[city]:
            pheromone_strength = pheromone[current_city, city] ** alpha
            distance_strength = (1 / (distance_matrix[current_city, city] + 1e-10)) ** beta
            probabilities[city] = pheromone_strength * distance_strength
    
    total_probability = np.sum(probabilities)
    if random.random() < q0:  # Exploitation (choose the best option)
        next_city = np.argmax(probabilities)
    else:  # Exploration (probabilistic selection)
        probabilities /= total_probability  # Normalize the probabilities
        next_city = np.random.choice(range(num_cities), p=probabilities)

    return next_city

# Calculate the total distance (fitness) of the solution (tour)
def calculate_fitness(solution, distance_matrix):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += distance_matrix[solution[i], solution[i + 1]]
    return total_distance
