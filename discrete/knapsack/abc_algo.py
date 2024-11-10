import numpy as np
import random
import time  # Import the time module to measure elapsed time

def abc_algorithm_knapsack(values, weights, capacity, population_size=100, iterations=1000, callback=None):
    # Number of items
    n = len(values)   

    # Initialize solutions (bees)
    population = np.random.randint(2, size=(population_size, n))  # Binary solutions    

    # Calculate initial fitness (total value of selected items)
    def fitness(solution):
        total_weight = np.dot(solution, weights)
        total_value = np.dot(solution, values)        

        # If total weight exceeds capacity, return a very low fitness value
        if total_weight > capacity:
            return 0
        return total_value
    
    fitness_values = np.array([fitness(sol) for sol in population])    

    # Store the best solution found
    best_solution = population[np.argmax(fitness_values)]
    best_fitness = np.max(fitness_values)
    
    # Set up limit for each bee (limit = number of times a bee can not improve its solution)
    limit = 100
    cycle = 0  

    # Start measuring time
    start_time = time.time()

    # Main loop of the algorithm
    for iteration in range(iterations):
        improved = False  # Track if any improvement happened

        # Employed bees phase
        for i in range(population_size):
            # Generate a neighbor solution (choose a random neighbor to modify)
            bee = population[i].copy()
            k = random.randint(0, n-1)
            bee[k] = 1 - bee[k]  # Flip the binary value of a random item            

            # Evaluate the new solution
            new_fitness = fitness(bee)
            if new_fitness > fitness_values[i]:
                population[i] = bee
                fitness_values[i] = new_fitness
                improved = True  # Mark that an improvement was made

        # Check if sum of fitness values is zero
        fitness_sum = np.sum(fitness_values)
        if fitness_sum == 0:
            probabilities = np.ones(population_size) / population_size  # Assign uniform probability if all fitness values are zero
        else:
            probabilities = fitness_values / fitness_sum  # Fitness-based selection probabilities

        for i in range(population_size):
            if random.random() < probabilities[i]:  # Select a bee to explore
                bee = population[i].copy()
                k = random.randint(0, n-1)
                bee[k] = 1 - bee[k]  # Flip the binary value of a random item                 

                # Evaluate the new solution
                new_fitness = fitness(bee)
                if new_fitness > fitness_values[i]:
                    population[i] = bee
                    fitness_values[i] = new_fitness
                    improved = True  # Mark that an improvement was made
        
        # Scout bees phase (if no improvement for some time, restart a bee's solution)
        for i in range(population_size):
            if fitness_values[i] == best_fitness:
                # Randomly reinitialize the bee's solution if it has not improved
                population[i] = np.random.randint(2, size=n)
                fitness_values[i] = fitness(population[i])
        
        # Update the best solution found
        current_best_fitness = np.max(fitness_values)
        current_best_solution = population[np.argmax(fitness_values)]
        
        # Check if current best fitness is better than the historical best
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
            improved = True  # Mark that an improvement was made
        
        # # Log each iteration's best fitness
        # print(f"Iteration {iteration + 1}: Current Best Fitness = {current_best_fitness}, Historical Best Fitness = {best_fitness}")
        
        # # Break early if no improvements were made
        # if not improved:
        #     print(f"No improvement in iteration {iteration + 1}, algorithm might be stagnating.")
        #     break

        if callback:
            callback(best_fitness, iteration)

    # End measuring time
    elapsed_time = time.time() - start_time

    return best_solution, best_fitness, elapsed_time
