"""
(1+10) Norm EA_{>0}

The (1 + 10) EA>0 using the normalized bit mutation with self-adaptive r, which samples l from a normal distribution N(r, r(1 - r/n)).
"""

import numpy as np
import ioh

class norm_ea:
  
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

        # Sample l from a normal distribution N(r, r(1 - r/n))
        r = mutation_rate * n
        stddev = np.sqrt(r * (1 - mutation_rate))
        l = int(np.round(np.random.normal(r, stddev)))
        l = max(1, min(l, n))  # Ensure l is in [1, n]

        # Randomly choose l distinct indices to flip
        flip_indices = np.random.choice(n, size=l, replace=False)
        for index in flip_indices:
          candidate[index] = 1 - candidate[index]  # Flip bit (0->1, 1->0)

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
    print(f"(1+10) norm EA with budget {self.budget} evaluations and an initial mutation rate {self.mutation_rate}")