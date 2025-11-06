"""
Two-Rate Evolutionary Algorithm (2-Rate EA)

The (1 + 10) EA_{>0} using standard bit mutation with self-adaptive mutation rates.
"""

import numpy as np
import ioh

class two_rate_ea:
  
  def __init__(self, budget: int, mutation_rate: float):
    self.budget = budget
    self.mutation_rate = mutation_rate # Initial mutation rate

  def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
    # Initialize with a random binary solution
    x = np.random.randint(0, 2, size=problem.meta_data.n_variables)
    problem(x)

    while problem.state.evaluations < self.budget and not problem.state.optimum_found:
      # Generate 10 offspring with different mutation rates
      offspring = []
      n = len(problem.state.current_best.x)
      for _ in range(10):
        candidate = problem.state.current_best.x.copy()

        # Decide mutation rate: half use r/2, half use 2r
        if np.random.rand() < 0.5:
          mutation_rate = max(1/n, self.mutation_rate / 2)
        else:
          mutation_rate = min(0.5, self.mutation_rate * 2)

        # Bit-flip mutation: flip each bit with probability mutation_rate
        for i in range(n):
          if np.random.random() < mutation_rate:
            candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)

        offspring.append((candidate, mutation_rate))

      # Evaluate all offspring and select the best one
      best_offspring = None
      best_fitness = -np.inf
      best_mutation_rate = self.mutation_rate

      for candidate, mutation_rate in offspring:
        fitness = problem(candidate)
        if fitness > best_fitness:
          best_fitness = fitness
          best_offspring = candidate
          best_mutation_rate = mutation_rate

      # Update current best solution
      problem(best_offspring)

      # Update mutation rate to that of the best offspring
      self.mutation_rate = best_mutation_rate
  
  def parameter(self):
      return f'{self.mutation_rate}'

  def printer(self):
    print(f"(1+10) 2-Rate EA with budget {self.budget} evaluations and an initial mutation rate {self.mutation_rate}")