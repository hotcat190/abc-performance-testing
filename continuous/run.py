from functions import schaffer, sphere, rastrigin, rosenbrock

from abc_algo import abc_algorithm
from ga import genetic_algorithm
from aco import ant_colony_optimization

import matplotlib.pyplot as plt
import numpy as np

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

    # Initialize lists to store fitness and time for the current function
    fitness_per_algo = []
    time_per_algo = []

    # ABC
    best_solution, best_fitness, elapsed_time = abc_algorithm(func, dims=2, bound=bound)
    print(f"ABC: Best Fitness = {best_fitness}, Time = {elapsed_time:.2f}s")
    fitness_per_algo.append(best_fitness)
    time_per_algo.append(elapsed_time)

    # GA
    best_solution, best_fitness, elapsed_time = genetic_algorithm(func, dims=2, bound=bound)
    print(f"GA: Best Fitness = {best_fitness}, Time = {elapsed_time:.2f}s")
    fitness_per_algo.append(best_fitness)
    time_per_algo.append(elapsed_time)

    # ACO
    best_solution, best_fitness, elapsed_time = ant_colony_optimization(func, dims=2, bound=bound)
    print(f"ACO: Best Fitness = {best_fitness}, Time = {elapsed_time:.2f}s")
    fitness_per_algo.append(best_fitness)
    time_per_algo.append(elapsed_time)

    # Append the fitness and time data for this function to the respective dictionary
    best_fitness_values[name] = fitness_per_algo
    execution_times[name] = time_per_algo

    print()

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot for each function in separate subplot
for ax, (name, fitness) in zip(axs.ravel(), best_fitness_values.items()):
    bars = ax.bar(algorithms, fitness, color=['blue', 'green', 'red'])
    ax.set_title(f"{name} Function")
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Best Fitness')
    ax.set_ylim(0, max(fitness) + 0.25)  # Limit y-axis to start from 0 and go up to max fitness value + padding
    ax.grid(True)

    # Set the y-axis ticks with a minimum divisor of 0.5
    ax.yaxis.set_ticks(np.arange(0, max(fitness) + 0.25, 0.5))

    # Add the best fitness values above the bars
    for i, bar in enumerate(bars):
        fitness_value = fitness[i]  # Directly get the fitness value for each algorithm
        ax.text(bar.get_x() + bar.get_width() / 2, fitness_value, f'{fitness_value:.4f}', 
                ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
