import numpy as np
import random

# Function to calculate the total distance of a given path
def calcDistance(path, distance_matrix, depot=0):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i], path[i + 1]]
    total_distance += distance_matrix[path[-1], depot]  # Returning to depot
    return total_distance

# the genetic algorithm
def ga_algorithm_tsp(distance_matrix, TARGET=0, population_size=100, generations=100, TOURNAMENT_SELECTION_SIZE=5, 
                     MUTATION_RATE=0.1, CROSSOVER_RATE=0.9, depot=0):
    
    lenCities = len(distance_matrix)
    # Initialize population with random paths, including depot at start/end
    population = []
    for _ in range(population_size):
        path = list(range(lenCities))
        path.remove(depot)
        random.shuffle(path)
        path = [depot] + path + [depot]  # Add depot at start and end
        population.append([calcDistance(path, distance_matrix, depot), path])  # Store the path and its fitness
    
    fitness_over_time = []
    gen_number = 0
    while gen_number < generations:
        new_population = []

        # Selecting two of the best options (elitism)
        new_population.append(sorted(population)[0])  # Keep the best solution
        new_population.append(sorted(population)[1])  # Keep the second-best solution

        # Generate new solutions through crossover
        for _ in range(int((population_size - 2) / 2)):
            # Crossover
            if random.random() < CROSSOVER_RATE:
                parent1 = sorted(random.choices(population, k=TOURNAMENT_SELECTION_SIZE))[0][1]
                parent2 = sorted(random.choices(population, k=TOURNAMENT_SELECTION_SIZE))[0][1]
                
                point = random.randint(1, lenCities - 2)  # Exclude depot positions
                child1 = parent1[:point] + [city for city in parent2 if city not in parent1[:point]]
                child2 = parent2[:point] + [city for city in parent1 if city not in parent2[:point]]
            else:
                # No crossover, just select parents randomly
                child1, child2 = random.choices(population)[0][1], random.choices(population)[0][1]
            
            # Mutation
            if random.random() < MUTATION_RATE:
                point1, point2 = random.sample(range(1, lenCities - 1), 2)
                child1[point1], child1[point2] = child1[point2], child1[point1]
                point1, point2 = random.sample(range(1, lenCities - 1), 2)
                child2[point1], child2[point2] = child2[point2], child2[point1]

            # Evaluate new solutions and add to new population
            new_population.append([calcDistance(child1, distance_matrix, depot), child1])
            new_population.append([calcDistance(child2, distance_matrix, depot), child2])

        population = new_population  # Set new population for the next generation
        gen_number += 1

        # Track the best fitness value at this generation
        best_fitness = sorted(population)[0][0]
        fitness_over_time.append(best_fitness)

        # # Print progress for every 10 generations
        # if gen_number % 10 == 0:
        #     print(f"Generation {gen_number}: Best Fitness = {best_fitness}")

        # If the target fitness is reached, break the loop early
        if best_fitness < TARGET:
            break

    # Return the best solution and its fitness, as well as fitness over time
    best_solution = sorted(population)[0]
    return best_solution, gen_number, fitness_over_time
