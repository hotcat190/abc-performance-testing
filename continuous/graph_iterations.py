import numpy as np
import matplotlib.pyplot as plt
from functions import schaffer, sphere, rastrigin, rosenbrock
from abc_algo import abc_algorithm
from ga import genetic_algorithm
from aco import ant_colony_optimization

# Define functions and their bounds
functions = [
    (schaffer, "Schaffer", 10),
    (sphere, "Sphere", 10),
    (rastrigin, "Rastrigin", 5.12),
    (rosenbrock, "Rosenbrock", 2)
]

# Parameters
dims = 2
population_size = 100  # Population size for GA and ACO
runs = 30  # Number of runs to average

# Collecting data for each algorithm and function
results = {name: {"ABC": [], "GA": [], "ACO": []} for _, name, _ in functions}

# Run each algorithm on each function 30 times to collect data
for func, name, bound in functions:
    print(f"Running algorithms on {name} function...")

    # Initialize empty lists for collecting fitness values for each run
    abc_fitness_runs = []
    ga_fitness_runs = []
    aco_fitness_runs = []

    # Define iterations based on the function
    if name == "Schaffer":
        iterations = 450
    elif name == "Sphere":
        iterations = 25
    elif name == "Rastrigin":
        iterations = 100
    elif name == "Rosenbrock":
        iterations = 50

    # Run ABC algorithm multiple times and collect fitness values
    for _ in range(runs):
        abc_fitness = []

        def abc_callback(best_fitness, iteration):
            abc_fitness.append(best_fitness)
            return iteration < iterations

        abc_algorithm(func, dims=dims, bound=bound, iterations=iterations, callback=abc_callback)
        abc_fitness_runs.append(abc_fitness)

    # Run GA algorithm multiple times and collect fitness values
    for _ in range(runs):
        ga_fitness = []

        def ga_callback(best_fitness, iteration):
            ga_fitness.append(best_fitness)
            return iteration < iterations

        genetic_algorithm(func, dims=dims, bound=bound, iterations=iterations, callback=ga_callback)
        ga_fitness_runs.append(ga_fitness)

    # Run ACO algorithm multiple times and collect fitness values
    for _ in range(runs):
        aco_fitness = []

        def aco_callback(best_fitness, iteration):
            aco_fitness.append(best_fitness)
            return iteration < iterations

        ant_colony_optimization(func, dims=dims, bound=bound, iterations=iterations, callback=aco_callback)
        aco_fitness_runs.append(aco_fitness)

    # Average the fitness values for each iteration across 30 runs
    results[name]["ABC"] = np.mean(np.array(abc_fitness_runs), axis=0)
    results[name]["GA"] = np.mean(np.array(ga_fitness_runs), axis=0)
    results[name]["ACO"] = np.mean(np.array(aco_fitness_runs), axis=0)

# Plotting results
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Best Fitness per Iteration for Each Algorithm")

# Plot each function's results
for ax, (name, algorithms) in zip(axs.ravel(), results.items()):
    ax.plot(algorithms["ABC"], label="ABC", color="blue")  # ABC in blue
    ax.plot(algorithms["GA"], label="GA", color="green")   # GA in green
    ax.plot(algorithms["ACO"], label="ACO", color="red")   # ACO in red

    ax.set_title(f"{name} Function")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Best Fitness")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
