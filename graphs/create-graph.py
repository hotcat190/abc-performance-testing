import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the functions
def schaffer(x):
    return 0.5 + (np.sin(np.sqrt(x**2))**2 - 0.5) / (1 + 0.001 * (x**2))**2

def sphere(x):
    return x**2

def rastrigin(x):
    A = 10
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

# Define the plotting function
def plot_function(func, x_range, y_range, title):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)

    # Apply the function element-wise to the meshgrid
    if func == rosenbrock:
        Z = func(X, Y)  # Rosenbrock takes both X and Y
    else:
        Z = np.fromiter((func(np.sqrt(x**2 + y**2)) for x, y in zip(np.ravel(X), np.ravel(Y))), dtype=float)

    # Reshape Z to match the shape of X and Y
    Z = Z.reshape(X.shape)

    # Plot the 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

# Plot the Schaffer function
plot_function(schaffer, (-10, 10), (-10, 10), "Schaffer Function")

# Plot the Sphere function
plot_function(sphere, (-10, 10), (-10, 10), "Sphere Function")

# Plot the Rastrigin function
plot_function(rastrigin, (-5.12, 5.12), (-5.12, 5.12), "Rastrigin Function")

# Plot the Rosenbrock function
plot_function(rosenbrock, (-2, 2), (-1, 3), "Rosenbrock Function")
