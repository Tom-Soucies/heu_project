"""
Univariate Marginal Distribution Algorithm (UMDA)

UMDA maintains a population of s solutions and uses the best s/2 solutions to estimate marginal distributions for each variable. It samples new populations based on the marginal distributions and updates the distributions iteratively. We set s = 50 in our experiments.
"""

import numpy as np
import ioh

class umda:
  
  def __init__(self, budget: int, population_size: int = 50, top_k: int = 10):
    self.budget = budget
    self.population_size = population_size
    self.top_k = top_k

  def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
    n = problem.meta_data.n_variables
    # self.top_k = self.population_size // 2

    # Initialize population with random binary solutions
    population = np.random.randint(0, 2, size=(self.population_size, n))
    fitness_values = np.array([problem(ind) for ind in population])

    while problem.state.evaluations < self.budget and not problem.state.optimum_found:
      # Select the best half of the population
      selected_indices = np.argsort(fitness_values)[-self.top_k:]
      selected_population = population[selected_indices]

      # Estimate marginal probabilities
      marginal_probs = np.mean(selected_population, axis=0)

      # Sample new population based on marginal probabilities
      new_population = np.random.rand(self.population_size, n) < marginal_probs
      population = new_population.astype(int)

      # Evaluate new population
      fitness_values = np.array([problem(ind) for ind in population])

  def parameter(self):
      return f'  {self.population_size}'

  def printer(self):
      print(f"Univariate Marginal Distribution Algorithm with budget {self.budget} evaluations, population size {self.population_size}, top-k {self.top_k}")