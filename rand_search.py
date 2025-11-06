"""
Random Search

The random search samples new solutions uniformly at random iteratively.
"""

import numpy as np
import ioh

class rand_search:
  'Simple random search algorithm for binary optimization'
  def __init__(self, budget: int):
    self.budget: int = budget

  def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
    'Evaluate the problem n times with a randomly generated binary solution'

    for _ in range(self.budget):
      # Generate random binary vector (0s and 1s)
      x = np.random.randint(0, 2, size=problem.meta_data.n_variables)
      problem(x)
  
  def printer(self):
    print(f"Random Search with budget {self.budget} evaluations")