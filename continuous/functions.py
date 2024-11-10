import numpy as np

# Schaffer function
def schaffer(x):
    return 0.5 + (np.sin(np.sqrt(x[0]**2 + x[1]**2))**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2

# Sphere function
def sphere(x):
    return np.sum(x**2)

# Rastrigin function
def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
