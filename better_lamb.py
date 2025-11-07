"""
(1+(λ,λ)) EA_{>0} for Binary Optimization Problems

The (1+λ) EA>0 with self-adaptive λ. The algorithm applies the standard bit mutation and a biased or parameterized uniform crossover, where the mutation rate and the crossover probability depend on the value of λ. We implement in this paper the (1+(10,10)) EA_{>0} with an initial λ = 10.
"""

import numpy as np
import ioh

class better_lamb:
  
  def __init__(self, budget: int, lam: int = 10, rate: int = 2):
    self.budget = budget
    self.lam = lam  # Initial λ value
    self.rate = rate  # Adaptation rate for λ

  def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
    n = problem.meta_data.n_variables
    # Initialize with a random binary solution
    x = np.random.randint(0, 2, size=n)
    problem(x)

    lam = self.lam  # Initial λ value

    while problem.state.evaluations < self.budget and not problem.state.optimum_found:
      current = problem.state.current_best.x.copy()

      # Mutation phase
      mutation_rate = lam / n
      offspring = []
      for _ in range(lam):
        candidate = problem.state.current_best.x.copy()
        # Bit-flip mutation: flip each bit with probability mutation_rate
        for i in range(n):
          if np.random.random() < mutation_rate:
            candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)
        offspring.append(candidate)
      
      # Evaluate all offspring and select the best one
      fitness_values = [problem(ind) for ind in offspring]
      best_offspring_idx = np.argmax(fitness_values)
      best_offspring = offspring[best_offspring_idx]

      # Crossover phase
      crossover_prob = 1 / lam
      final_offspring = problem.state.current_best.x.copy()
      for i in range(n):
        if np.random.random() < crossover_prob:
          final_offspring[i] = best_offspring[i]

      problem(final_offspring)

      # Self-adaptation of λ
      if problem(final_offspring) < problem(current):
        lam = max(1, lam // self.rate)  # Decrease λ on success
      else:
        lam = min(n, lam * self.rate)   # Increase λ on failure
  
  def printer(self):
    print(f"(1+(λ,λ)) EA with budget {self.budget} evaluations")