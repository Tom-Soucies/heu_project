"""
Fast Genetic Algorithm

The (1+1) fast GA differs from the (1+1) EA by sampling l from a power-law distribution with β = 1.5. The power-law distribution is a heavy-tailed distribution, and its probability of sampling large l > 1 is higher, compared to the standard bit mutation with p = 1/n.
"""

import numpy as np
import ioh

class fast_ga:
  
  def __init__(self, budget: int):
    self.budget = budget

  def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
    # Initialize with a random binary solution
    x = np.random.randint(0, 2, size=problem.meta_data.n_variables)
    problem(x)

    while problem.state.evaluations < self.budget and not problem.state.optimum_found:
      # Mutate: flip l bits where l is sampled from a power-law distribution
      candidate = problem.state.current_best.x.copy()

      # Sample l from a power-law distribution with β = 1.5
      beta = 1.5
      n = len(candidate)
      l = int(np.floor((np.random.pareto(beta) + 1)))  # Shifted Pareto to get values >= 1
      l = min(l, n)  # Ensure l does not exceed n

      # Randomly choose l distinct indices to flip
      flip_indices = np.random.choice(n, size=l, replace=False)
      for index in flip_indices:
        candidate[index] = 1 - candidate[index]  # Flip bit (0->1, 1->0)

      problem(candidate)
  
  def printer(self):
    print(f"(1+1) Fast GA with budget {self.budget} evaluations")