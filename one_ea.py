"""
Simple (1+1) Evolutionary Algorithm

The (1+1) EA_{>0} uses the standard bit mutation with a static mutation rate p = 1/n. The standard bit mutation samples l, the number of distinct bits to be flipped, from a conditional binomial distribution Bin_{>0}(n, p).
"""

import numpy as np
import ioh

class one_ea:

  def __init__(self, budget: int, mutation_rate: float):
    self.budget = budget
    self.mutation_rate = mutation_rate

  def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
    # Initialize with a random binary solution
    x = np.random.randint(0, 2, size=problem.meta_data.n_variables)
    problem(x)

    while problem.state.evaluations < self.budget and not problem.state.optimum_found:
      # Mutate: flip bits with given probability
      candidate = problem.state.current_best.x.copy()

      # Bit-flip mutation: flip each bit with probability mutation_rate
      for i in range(len(candidate)):
        if np.random.random() < self.mutation_rate:
          candidate[i] = 1 - candidate[i] # Flip bit (0->1, 1->0)

      problem(candidate)

  def parameter(self):
      return f'{self.mutation_rate}'
  
  def printer(self):
    print(f"(1+1) EA with budget {self.budget} evaluations and mutation rate {self.mutation_rate}")