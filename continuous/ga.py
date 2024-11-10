import numpy as np
import random
import time

# Define Genetic Algorithm
def genetic_algorithm(fitness_function, dims, bound, population_size=100, generations=400, mutation_rate=0.01, crossover_rate=0.7):
    # Initialize Population (randomly generated solutions)
    population = np.random.uniform(-bound, bound, (population_size, dims))  # You can adjust the range here based on the problem

    def fitness(x):
        return 1 / (1 + fitness_function(x))  # Inverse of fitness_function to maximize it (for minimization problem)

    def select_parents(population, fitness_values):
        # Tournament Selection
        idx = np.random.choice(np.arange(len(population)), size=2, replace=False)
        parent1, parent2 = population[idx]
        fit1, fit2 = fitness_values[idx]
        return parent1 if fit1 > fit2 else parent2

    def crossover(parent1, parent2):
        # Single-point Crossover
        if random.random() < crossover_rate:
            point = random.randint(1, dims - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(child):
        # Mutation: Randomly change a value in the child
        if random.random() < mutation_rate:
            idx = random.randint(0, dims - 1)
            child[idx] = random.uniform(-5, 5)  # Mutation in the range -5 to 5
        return child

    # Track the best solution
    best_solution = None
    best_fitness = float('-inf')

    # Start the Genetic Algorithm
    start_time = time.time()
    for generation in range(generations):
        fitness_values = np.array([fitness(individual) for individual in population])

        # Track the best solution in the current population
        current_best_fitness = np.max(fitness_values)
        current_best_solution = population[np.argmax(fitness_values)]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution

        # Generate new population using selection, crossover, and mutation
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
