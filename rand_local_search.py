"""
Randomized Local Search

The RLS differs from the (1+1) EA_{>0} by flipping exactly l = 1 bit in each iteration.
"""

import numpy as np
import ioh

class rand_local_search:
  'Simple randomized local search algorithm for binary optimization'
  def __init__(self, budget: int):
    self.budget: int = budget

  def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
    'Evaluate the problem by iteratively flipping a random bit in the current solution'

    # Initialize with a random binary solution
    x = np.random.randint(0, 2, size=problem.meta_data.n_variables)
    problem(x)

    while problem.state.evaluations < self.budget and not problem.state.optimum_found:
      # Create a candidate solution by flipping a random bit
      candidate = problem.state.current_best.x.copy()
      flip_index = np.random.randint(0, len(candidate))
      candidate[flip_index] = 1 - candidate[flip_index]  # Flip the bit

      problem(candidate)
  
  def printer(self):
    print(f"Randomized Local Search with budget {self.budget} evaluations")