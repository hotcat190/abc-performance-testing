import numpy as np
import random
import time

def ant_colony_optimization_knapsack(values, weights, capacity, population_size=10, iterations=100, alpha=1, beta=2, evaporation_rate=0.5, q0=0.9, callback=None):
    num_items = len(values)
    
    # Initialize pheromone matrix (pheromone values for each item)
    pheromone = np.ones(num_items)

    def select_item(pheromone, values, weights, available_items, alpha=1, beta=2):
        # Vectorized pheromone values and value-to-weight ratio calculations
        pheromone_values = pheromone[available_items] ** alpha

        value_weight_ratios = []
        for item in available_items:
            value_weight_ratios.append((values[item] / (weights[item] + 1e-10)) ** beta)
        value_weight_ratios = np.array(value_weight_ratios)
        
        # Calculate the probabilities in a vectorized manner
        probabilities = pheromone_values * value_weight_ratios
        
        # Normalize the probabilities
        probabilities /= probabilities.sum()

        # Select an item using the computed probabilities
        selected_item = np.random.choice(available_items, p=probabilities)
        return selected_item
        
    # Function to construct a solution for an ant
    def construct_solution(pheromone, values, weights, capacity):
        available_items = np.ones(len(values), dtype=bool)  # All items initially available
        current_available_items = np.where(available_items)[0]
        solution = []
        total_weight = 0
        
        while len(current_available_items) > 0:
            item = select_item(pheromone, values, weights, current_available_items)
            if total_weight + weights[item] <= capacity:
                solution.append(item)
                total_weight += weights[item]

                # Remove the selected item from available_items using boolean indexing
                available_items[item] = False                
                current_available_items = np.where(available_items)[0]
            else:
                break  # Stop if adding this item exceeds the capacity
        
        return solution, total_weight
    
    # Function to calculate fitness of a solution (total value of selected items)
    def calculate_fitness(solution):
        return sum(values[item] for item in solution)
    
    # Ant Colony Optimization loop
    best_solution = None
    best_fitness = 0
    best_solution_found = None
    
    # Track the time taken for the algorithm
    start_time = time.time()

    for iteration in range(iterations):
        # Ants construct solutions
        solutions = []
        fitness_values = []
        
        for _ in range(population_size):
            solution, total_weight = construct_solution(pheromone, values, weights, capacity)
            fitness = calculate_fitness(solution)
            solutions.append((solution, total_weight, fitness))
            fitness_values.append(fitness)
        
        # Find the best solution from all ants in the current iteration
        iteration_best_fitness = max(fitness_values)
        iteration_best_solution = solutions[np.argmax(fitness_values)][0]
        
        if iteration_best_fitness > best_fitness:
            best_fitness = iteration_best_fitness
            best_solution = iteration_best_solution

        if callback:
            callback(best_fitness, iteration)
        
        # Update pheromones
        pheromone_delta = np.zeros(num_items)
        for solution, total_weight, fitness in solutions:
            if fitness > 0:  # Add pheromone only if the solution is valid
                for item in solution:
                    pheromone_delta[item] += fitness
        
        # Apply pheromone evaporation
        pheromone *= (1 - evaporation_rate)
        
        # Update pheromones with new values from solutions
        pheromone += pheromone_delta
        
        # Optionally, you can apply a strong pheromone deposit (exploitation) for the best solution found so far
        if random.random() < q0:
            for item in best_solution:
                pheromone[item] += best_fitness        
    
    elapsed_time = time.time() - start_time
    
    return best_solution, best_fitness, elapsed_time
