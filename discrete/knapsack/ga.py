import numpy as np
import random
import time

def genetic_algorithm_knapsack(values, weights, capacity, population_size=100, iterations=100, mutation_rate=0.01, crossover_rate=0.7, callback=None):
    # Initialize population with random binary chromosomes (0 or 1)
    population = np.random.randint(2, size=(population_size, len(values)))

    def fitness(chromosome):
        total_weight = np.sum(chromosome * weights)
        total_value = np.sum(chromosome * values)
        
        if total_weight <= capacity:
            return total_value  # Valid solution, return total value
        else:
            return 0  # Invalid solution, return 0

    def select_parents(population, fitness_values):
        # Tournament selection
        idx = np.random.choice(np.arange(len(population)), size=2, replace=False)
        parent1, parent2 = population[idx]
        fit1, fit2 = fitness_values[idx]
        return parent1 if fit1 > fit2 else parent2

    def crossover(parent1, parent2):
        # Single-point crossover
        if random.random() < crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(child):
        # Bit-flip mutation
        if random.random() < mutation_rate:
            idx = random.randint(0, len(child) - 1)
            child[idx] = 1 - child[idx]  # Flip the bit
        return child

    best_solution = None
    best_fitness = float('-inf')

    start_time = time.time()

    # Start Genetic Algorithm iterations
    for generation in range(iterations):
        # Evaluate fitness for each individual in the population
        fitness_values = np.array([fitness(individual) for individual in population])

        # Track the best solution in the population
        current_best_fitness = np.max(fitness_values)
        current_best_solution = population[np.argmax(fitness_values)]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
        
        # Call the callback if provided
        if callback:
            callback(best_fitness, generation)
        
        # Create the next generation using selection, crossover, and mutation
        new_population = []

        for _ in range(population_size // 2):
            parent1 = select_parents(population, fitness_values)
            parent2 = select_parents(population, fitness_values)

            child1, child2 = crossover(parent1, parent2)

            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = np.array(new_population)

    elapsed_time = time.time() - start_time

    return best_solution, best_fitness, elapsed_time
