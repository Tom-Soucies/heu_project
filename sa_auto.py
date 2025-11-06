"""
Simulated Annealing (SA)

The SA algorithm with automatic settings based on the problem dimension. For the start temperature, the probability of accepting a solution that is 1 worse than the current solution is 0.1, and this value is 1/√n for the end temperature.
"""

import numpy as np
import ioh

class sa_auto:
  
  def __init__(self, budget: int):
    self.budget = budget

  def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
    n = problem.meta_data.n_variables
    # Automatic temperature settings
    start_temp = -1 / np.log(0.1)
    end_temp = -1 / np.log(1 / np.sqrt(n))
    alpha = (end_temp / start_temp) ** (1 / (self.budget / n))

    # Initialize with a random binary solution
    current_solution = np.random.randint(0, 2, size=n)
    current_fitness = problem(current_solution)
    temperature = start_temp

    while problem.state.evaluations < self.budget and not problem.state.optimum_found:
      for _ in range(n):
        # Generate neighbor by flipping one random bit
        neighbor = current_solution.copy()
        flip_index = np.random.randint(0, n)
        neighbor[flip_index] = 1 - neighbor[flip_index]  # Flip bit

        neighbor_fitness = problem(neighbor)
        delta_fitness = neighbor_fitness - current_fitness

        # Decide whether to accept the neighbor
        if delta_fitness > 0 or np.random.rand() < np.exp(delta_fitness / temperature):
          current_solution = neighbor
          current_fitness = neighbor_fitness

        # Update temperature
        temperature *= alpha
        
        # Update the problem state with the current solution
        problem(current_solution)
  
  def printer(self):
      print(f"Simulated Annealing with budget {self.budget} evaluations and automatic temperature settings")
        
        