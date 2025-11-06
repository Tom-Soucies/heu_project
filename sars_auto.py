"""
Simulated Annealing with Iterative Reheating (sars-auto)

The algorithm applies iterative restarts for the SA. We assign different function evaluation budgets to each restarting round.
"""

import numpy as np
import ioh

class sars_auto:
  
  def __init__(self, budget: int, rounds: int = 5):
    self.budget = budget
    self.rounds = rounds

  def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
    n = problem.meta_data.n_variables
    round_budget = self.budget // self.rounds
    start_temp = -1 / np.log(0.1)
    end_temp = -1 / np.log(1 / np.sqrt(n))

    for _ in range(self.rounds):
      alpha = (end_temp / start_temp) ** (1 / (round_budget / n))
      current_solution = np.random.randint(0, 2, size=n)
      current_fitness = problem(current_solution)
      temperature = start_temp

      evaluations = 0
      while evaluations < round_budget and not problem.state.optimum_found:
        for _ in range(n):
          if evaluations >= round_budget:
            break

          neighbor = current_solution.copy()
          flip_index = np.random.randint(0, n)
          neighbor[flip_index] = 1 - neighbor[flip_index]

          neighbor_fitness = problem(neighbor)
          evaluations += 1
          delta_fitness = neighbor_fitness - current_fitness

          if delta_fitness > 0 or np.random.rand() < np.exp(delta_fitness / temperature):
            current_solution = neighbor
            current_fitness = neighbor_fitness

          temperature *= alpha
          
          problem(current_solution)
          
  def parameter(self):
      return f'  {self.rounds}'
  
  def printer(self):
      print(f"Simulated Annealing with Iterative Reheating with budget {self.budget} evaluations over {self.rounds} rounds")