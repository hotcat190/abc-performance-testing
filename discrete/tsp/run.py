import numpy as np
import matplotlib.pyplot as plt
from aco import ant_colony_optimization_tsp
from abc_algo import abc_algorithm_tsp
from ga import ga_algorithm_tsp
from tsp import data

def run_tsp():
    distance_matrix = np.array(data["distance_matrix"])
    depot = data["depot"]

    population_size = 50  # Population size for ABC and GA
    iterations = 100
    alpha = 1
    beta = 2
    evaporation_rate = 0.5
    q0 = 0.9
    limit = 100  # Max cycles without improvement

    num_runs = 30
    aco_fitness_over_time = []
    abc_fitness_over_time = []
    ga_fitness_over_time = []

    print("Running ABC...")
    for _ in range(num_runs):
        # Run ABC
        _, _, abc_fitness = abc_algorithm_tsp(distance_matrix)
        abc_fitness_over_time.append(abc_fitness)

    print("Running GA...")
    for _ in range(num_runs):
        # Run GA
        _, _, ga_fitness = ga_algorithm_tsp(distance_matrix)
        ga_fitness_over_time.append(ga_fitness)

    print("Running ACO...")
    for _ in range(num_runs):
        # Run ACO
        _, _, aco_fitness = ant_colony_optimization_tsp(distance_matrix)
        aco_fitness_over_time.append(aco_fitness)

    # Average best fitness for each iteration
    aco_avg_fitness = np.mean(aco_fitness_over_time, axis=0)
    abc_avg_fitness = np.mean(abc_fitness_over_time, axis=0)
    ga_avg_fitness = np.mean(ga_fitness_over_time, axis=0)

    # Plot the average best fitness per iteration
    plt.plot(abc_avg_fitness, label="ABC Algorithm", color="blue")
    plt.plot(ga_avg_fitness, label="GA Algorithm", color="green")
    plt.plot(aco_avg_fitness, label="ACO Algorithm", color="red")
    plt.xlabel("Iteration")
    plt.ylabel("Average Best Fitness")
    plt.title("Average Best Fitness Over Time for ACO, ABC, and GA")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_tsp()
