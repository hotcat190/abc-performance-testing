import numpy as np
import matplotlib.pyplot as plt

from abc_algo import abc_algorithm_knapsack
from ga import genetic_algorithm_knapsack
from aco import ant_colony_optimization_knapsack

from knapsack import values, weights, capacity

# Parameters
population_size = 100
iterations = 100
runs = 30

def run_multiple_time(algorithm_func, values, weights, capacity, population_size, iterations, num_runs=30):
    all_fitness_per_run = []  # To store fitness per iteration across runs
    total_time = 0
    
    for _ in range(num_runs):
        best_fitness_per_iteration = []
        
        def iteration_callback(fitness, iteration):
            # Track fitness at each iteration
            if iteration >= len(best_fitness_per_iteration):
                best_fitness_per_iteration.append(fitness)
            else:
                best_fitness_per_iteration[iteration] = max(best_fitness_per_iteration[iteration], fitness)
        
        # Run the algorithm with the callback to capture fitness per iteration
        best_solution, best_fitness, elapsed_time = algorithm_func(values, weights, capacity, population_size, iterations, callback=iteration_callback)
        
        all_fitness_per_run.append(best_fitness_per_iteration)
        total_time += elapsed_time

    avg_time = total_time / runs
    print(f"Average time for {algorithm_func.__name__}: {avg_time:.4f} seconds")

    # Calculate the average best fitness at each iteration across runs
    max_iterations = max(len(fitness_per_run) for fitness_per_run in all_fitness_per_run)
    avg_fitness_per_iteration = np.zeros(max_iterations)
    
    for i in range(max_iterations):
        iteration_fitness = [fitness_per_run[i] for fitness_per_run in all_fitness_per_run if i < len(fitness_per_run)]
        avg_fitness_per_iteration[i] = np.mean(iteration_fitness)
    
    return avg_fitness_per_iteration

# Run algorithms and collect average fitness per iteration
print("Running ABC...")
abc_avg_fitness = run_multiple_time(abc_algorithm_knapsack, values, weights, capacity, population_size, iterations=iterations)
print("Running GA...")
ga_avg_fitness = run_multiple_time(genetic_algorithm_knapsack, values, weights, capacity, population_size, iterations=iterations)
print("Running ACO...")
aco_avg_fitness = run_multiple_time(ant_colony_optimization_knapsack, values, weights, capacity, population_size, iterations=iterations)

# Plot the average best fitness per iteration for each algorithm
plt.plot(abc_avg_fitness, label='ABC Algorithm', color='blue', linestyle='-', linewidth=2)
plt.plot(ga_avg_fitness, label='Genetic Algorithm', color='green', linestyle='-', linewidth=2)
plt.plot(aco_avg_fitness, label='ACO Algorithm', color='red', linestyle='-', linewidth=2)

plt.xlabel("Iteration")
plt.ylabel("Average Best Fitness")
plt.title("Average Best Fitness per Iteration Across 30 Runs")
plt.legend()
plt.show()
