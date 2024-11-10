from functions import schaffer, sphere, rastrigin, rosenbrock

from abc_algo import abc_algorithm
from ga import genetic_algorithm
from aco import ant_colony_optimization

import matplotlib.pyplot as plt
import numpy as np

# Number of runs for averaging
num_runs = 30

# Lists to store results for plotting
algorithms = ["ABC", "GA", "ACO"]
best_fitness_values = {"Schaffer": [], "Sphere": [], "Rastrigin": [], "Rosenbrock": []}
execution_times = {"Schaffer": [], "Sphere": [], "Rastrigin": [], "Rosenbrock": []}

# Running the algorithms on continuous functions
for func, name in [(schaffer, "Schaffer"), (sphere, "Sphere"), (rastrigin, "Rastrigin"), (rosenbrock, "Rosenbrock")]:
    print(f"{name} Function:")

    bound = 0
    # Use match-case to assign range_value
    match name:
        case "Schaffer" | "Sphere":
            bound = 10
        case "Rastrigin":
            bound = 5.12
        case "Rosenbrock":
            bound = 2

    # Initialize lists to store fitness and time averages for the current function
    avg_fitness_per_algo = []
    avg_time_per_algo = []

    # Run each algorithm `num_runs` times and compute the average best fitness and time
    for algorithm in [abc_algorithm, genetic_algorithm, ant_colony_optimization]:
        fitness_sum = 0
        time_sum = 0

        for _ in range(num_runs):
            best_solution, best_fitness, elapsed_time = algorithm(func, dims=2, bound=bound)
            fitness_sum += best_fitness
            time_sum += elapsed_time

        # Calculate averages
        avg_best_fitness = fitness_sum / num_runs
        avg_elapsed_time = time_sum / num_runs

        print(f"{algorithm.__name__}: Avg Best Fitness = {avg_best_fitness}, Avg Time = {avg_elapsed_time:.2f}s")
        avg_fitness_per_algo.append(avg_best_fitness)
        avg_time_per_algo.append(avg_elapsed_time)

    # Append the average fitness and time data for this function to the respective dictionary
    best_fitness_values[name] = avg_fitness_per_algo
    execution_times[name] = avg_time_per_algo

    print()

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot for each function in a separate subplot
for ax, (name, fitness) in zip(axs.ravel(), best_fitness_values.items()):
    bars = ax.bar(algorithms, fitness, color=['blue', 'green', 'red'])
    ax.set_title(f"{name} Function")
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Avg Best Fitness')
    ax.set_ylim(0, max(fitness) + 0.25)  # Limit y-axis to start from 0 and go up to max fitness value + padding
    ax.grid(True)

    # Set the y-axis ticks with a minimum divisor of 0.5
    ax.yaxis.set_ticks(np.arange(0, max(fitness) + 0.25, 0.5))

    # Add the average best fitness values above the bars
    for i, bar in enumerate(bars):
        fitness_value = fitness[i]  # Directly get the average fitness value for each algorithm
        ax.text(bar.get_x() + bar.get_width() / 2, fitness_value, f'{fitness_value:.4f}', 
                ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
