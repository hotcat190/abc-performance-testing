Performance tests on the Artificial Bee Colony algorithm for class INT3103 07 - Optimization.

# Testing method
ABC, GA, ACO is ran 30 times on several continuous functions and discrete problems.<br/>
The runner will calculate the average best_fitness during each iteration of each algorithm, and display them as a line graph.<br/>
For continuous functions, a bar chart is also provided for the final best_fitness value of the three algorithms.<br/>

# Continuous functions
<h2>Sphere function</h2>
A simple convex shape, used to test convergence speed.<br/>
Parameters: -10 < x<sub>i</sub> < 10, n = 2<br/>
<br/>
<img src="graphs/sphere.png">
  
<h2>Rastrigin function</h2>
A parabolic function with cosine waves on all its surface, tests an algorithm’s exploration capability and resilience to local minima traps.<br/>
Parameters: -5.12 < x<sub>i</sub> < 5.12, A = 10<br/>
<br/>
<img src="graphs/rastrigin.png">

<h2>Rosenbrock function</h2>
A narrow, curved valley leading to the global minimum, test an algorithm's ability to handle non-linear search paths and narrow, curved valleys.<br/>
Parameters: 2 < x<sub>1</sub> < 2, -1 < x<sub>2</sub> < 3, n = 3<br/>
<br/>
<img src="graphs/rosenbrock.png">

<h2>Schaffer function</h2>
The Schaffer function is non-convex and multimodal, with a pattern of ridges and valleys forming concentric rings around the origin, used to test algorithms on high-frequency oscillations and local minima challenges.<br/>
Parameters: -10 < x<sub>1</sub>, x<sub>2</sub> < 10<br/>
<br/>
<img src="graphs/schaffer.png">
