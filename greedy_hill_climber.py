"""
Greedy Hill Climber

The (1 + 1) gHC flips one bit, going through the bit-string from left to right, in each iteration. It updates the parent when the offspring obtains fitness at least as good as its parent.
"""

import numpy as np
import ioh

class greedy_hill_climber:
  'Simple greedy hill climber algorithm for binary optimization'
  def __init__(self, budget: int):
    self.budget: int = budget

  def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
    'Evaluate the problem by iteratively flipping bits in the current solution from left to right'

    # Initialize with a random binary solution
    x = np.random.randint(0, 2, size=problem.meta_data.n_variables)
    problem(x)

    n = problem.meta_data.n_variables
    current_index = 0

    while problem.state.evaluations < self.budget and not problem.state.optimum_found:
      # Create a candidate solution by flipping the current bit
      candidate = problem.state.current_best.x.copy()
      candidate[current_index] = 1 - candidate[current_index]  # Flip the bit

      problem(candidate)

      # Move to the next bit (wrap around if at the end)
      current_index = (current_index + 1) % n
  
  def printer(self):
    print(f"Greedy Hill Climber with budget {self.budget} evaluations")