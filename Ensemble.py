"""
Class joining(1+1) EA, (1+(λ,λ)) EA, and UMDA

This ensemble generates candidate solutions from all three algorithms
and performs majority voting at each position to create the final solution.
"""

import numpy as np
import ioh


class Ensemble:
    
    def __init__(self, budget: int=50_000, lam: int = 10, rate: int = 2, population_size: int = 50, top_k: int = 25, mutation_rate_ea: float = 0.05):
        self.budget = budget
        self.lam = lam  # lamb
        self.rate = rate # lamb
        self.population_size = population_size  # UMDA
        self.top_k = top_k  # UMDA
        self.mutation_rate_ea = mutation_rate_ea
        
    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
        n = problem.meta_data.n_variables
        
        x = np.random.randint(0, 2, size=n)
        problem(x)
        
        # Initialize UMDA population and marginal probabilities
        umda_population = np.random.randint(0, 2, size=(self.population_size, n))
        umda_fitness = np.array([problem(ind) for ind in umda_population])
        marginal_probs = np.ones(n) * 0.5  # Start with uniform distribution
        
        # Initialize lambda for better_lamb
        lam = self.lam
        
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            current_best = problem.state.current_best.x.copy()
            current_fitness = problem.state.current_best.y
            
            # one_ea candidate
            candidate_ea = current_best.copy()
            for i in range(n):
                if np.random.random() < self.mutation_rate_ea:
                    candidate_ea[i] = 1 - candidate_ea[i]
            
            # better_lamb candidate
            mutation_rate_lamb = lam / n
            offspring = []
            for _ in range(lam):
                candidate = current_best.copy()
                for i in range(n):
                    if np.random.random() < mutation_rate_lamb:
                        candidate[i] = 1 - candidate[i]
                offspring.append(candidate)
            
            # Evaluate offspring and get best
            offspring_fitness = [problem(ind) for ind in offspring]
            best_offspring_idx = np.argmax(offspring_fitness)
            best_offspring = offspring[best_offspring_idx]
            
            # Crossover phase
            crossover_prob = 1 / lam
            candidate_lamb = current_best.copy()
            for i in range(n):
                if np.random.random() < crossover_prob:
                    candidate_lamb[i] = best_offspring[i]
            
            # Adapt lambda based on comparison
            if offspring_fitness[best_offspring_idx] >= current_fitness:
                lam = max(1, lam // self.rate)
            else:
                lam = min(n, lam * self.rate)
            
            # umda candidate
            # Update marginal probabilities from best solutions
            selected_indices = np.argsort(umda_fitness)[-self.top_k:]
            selected_population = umda_population[selected_indices]
            marginal_probs = np.mean(selected_population, axis=0)
            
            # Sample a single candidate from marginal distribution
            candidate_umda = (np.random.rand(n) < marginal_probs).astype(int)
            
            # Update UMDA population (sample new population)
            umda_population = (np.random.rand(self.population_size, n) < marginal_probs).astype(int)
            umda_fitness = np.array([problem(ind) for ind in umda_population])
            


            # VOTING
            voted_solution = np.zeros(n, dtype=int)
            for i in range(n):
                votes = candidate_ea[i] + candidate_lamb[i] + candidate_umda[i]
                # if 2 or more algorithms vote for 1 at position i, set to 1
                voted_solution[i] = 1 if votes >= 2 else 0
            
            problem(voted_solution)
    
    def printer(self):
        print(f"Ensemble Algorithm with budget {self.budget} evaluations")