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

# Rosenbrock function
def rosenbrock(x, a=1, b=100):
    # If 'x' has more than 2 dimensions, the Rosenbrock function can be generalized
    # assuming that x[0] is for 'a' and x[1:] is for 'b'
    f = 0
    for i in range(len(x) - 1):
        f += (a - x[i])**2 + b * (x[i+1] - x[i]**2)**2
    return f