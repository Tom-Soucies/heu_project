import numpy as np
import ioh

class RadolaSoucies:
  
  def __init__(self, budget: int = 50000, lam: int = 5):
    self.budget = budget
    self.lam = lam  # λ value

  def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
    n = problem.meta_data.n_variables
    # Initialize with a random binary solution
    x = np.random.randint(0, 2, size=n)
    problem(x)

    lam = self.lam  # Initial λ value

    while problem.state.evaluations < self.budget and not problem.state.optimum_found:

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