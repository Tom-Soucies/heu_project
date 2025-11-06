print("""
Running Black-Box Optimization Algorithm for Pseudo-Boolean Problems

This script is structured as follows:
1. Installs the required dependencies and setting the params.
2. Accesses pseudo-boolean benchmark problem (can be specified when launching the script).
3. Chooses an algorithm to test between available ones that are either generic or the expected optimized solution called JT_algo (eventually default).
4. Investigates the performance of the chosen algorithm on the selected benchmark problem.

Later, when the JT_algo is implemented, this script can be used to compare its performance against other generic algorithms on various benchmark problems.

Different modules are imported to have a more convenient architecture and to facilitate future extensions of the code.
""")


### Step 1 ###
# Installing the required dependencies: ioh, iohinspector, tqdm, networkx, scipy
print("Step 1:")
print("Importing required dependencies...")
# Ignore the following lines if the packages are already installed
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ioh'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'iohinspector'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm', 'networkx', 'scipy']) # for some additional functionality

# Parsing command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run optimization algorithm on PBO benchmark problems.")
parser.add_argument('--problem', nargs=1, type=str, default='cut', # default problem is Max Cut
                    help='List of problem IDs to run the experiments on (default: Max Cut (cut), Max Coverage (cov), Max Influence (inf), Pack While Traveling (pwt))')
help_algo_text = '''
                    Algorithm ID to use for optimization (default: 0 for rand_search)
                      0: rand_search,
                      1: rand_local_search,
                      2: greedy_hill_climber,
                      3: one_ea (optionally with mutation rate as second argument),
                      4: fast_ga,
                      5: lamb_ea,
                      6: two_rate_ea (optionally with mutation rate as second argument),
                      7: norm_ea (optionally with mutation rate as second argument),
                      8: var_ea (optionally with mutation rate as second argument),
                      9: sa_auto,
                      10: sars_auto (optionally with number of rounds as second argument),
                      11: umda (optionally with population size as second argument)
                    '''
parser.add_argument('--algorithm', nargs='+', type=int, default=[0],
                    help=help_algo_text)
help_bench_text = '''
                    Decide which benchmark to run:
                      0: single experiment on specified problem (default),
                      1: benchmark of an algorithm over multiple problems
                      2: benchmark of a problem class over multiple algorithms
                    '''
parser.add_argument('--benchmark', '-b', nargs=1, type=int, default=[0],
                    help=help_bench_text)
parser.add_argument('--test', '-t', nargs=1, type=bool, default=[True],
                    help='If True, runs the current tested setup (default: True)')
args = parser.parse_args()

## Imports ##

# Classic libraries
import os
import matplotlib.pyplot as plt
import polars as pl
import random as rd

# IoH library for benchmark problems
import ioh
import iohinspector as inspector
from tqdm import tqdm

print("-" * 60)
print("Dependencies imported successfully.")
print("")
print("-" * 60)


## Some fixed params useful for step 2 and 3 ##
# Budget for the algorithms
BUDGET = 50000
# Dimension of the problems
DIMENSION = 500
# Number of runs for each experiment
N_RUNS = 5
# Benchmark type
BENCHMARK = args.benchmark[0]
# Test mode
TEST_MODE = args.test[0]

print("")
print("=" * 60)
print("")


### Step 2 ###
print("Step 2:")
print("Accessing pseudo-boolean benchmark problems...")

## Benchmark problems ##
# We will use problems from the PBO problem class (provided by the ioh library)
# See https://ioh.dev/docs/ioh/problems/pbo/ for more information on these problems.
# Especially, we will use the following problems: OneMax [1], LeadingOnes [2], LABS [3], IsingRing [19]
def name_to_problem(problem_name):
  name_dict = {
      "cut": 2000,
      "cov": 2100,
      "inf": 2200,
      "pwt": 2300,
  }
  return ioh.get_problem(name_dict.get(problem_name, 2000), instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH)

# Accessing pseudo-boolean benchmark problems (can be specified when launching the script)
# Asks for an argument when launching the script
# If no argument is provided, it will use OneMax as default problem
PROBLEM = name_to_problem(args.problem[0])

print(f"Problem '{PROBLEM.meta_data.name} (ID {PROBLEM.meta_data.problem_id})' accessed successfully.")
print("")
print("=" * 60)
print("")


### Step 3 ###
print("Step 3:")
print("Choosing an algorithm to test...")

## Algorithm available ##
# The objective is to compare a lot of algorithms on the benchmark problems. You can see the list below (the available ones are marked):
# [x] 0.  rand_search:         Random Search
# [x] 1.  rand_local_search:   Randomized Local Search
# [x] 2.  greedy_hill_climber: Greedy Hill Climber
# [x] 3.  one_ea:              (1+1) EA_{>0}
# [x] 4.  fast_ga:             (1+1) fast genetic algorithm (fast GA)
# [x] 5.  lamb_ea:             (1+(λ,λ)) EA_{>0}
# [x] 6.  two_rate_ea:         (1+10) 2rate-EA_{>0}
# [x] 7.  norm_ea:             (1+10) normEA_{>0}
# [x] 8.  var_ea:              (1+10) varEA_{>0}
# [x] 9.  sa_auto:             simulated annealing
# [x] 10. sars_auto:           simulated annealing with iterative restarts
# [x] 11. umda:                univariate marginal distribution algorithm (UMDA)
# [ ] 12. jt_algo:             the expected optimized solution

# Imports of the algorithms #
from rand_search import rand_search
from rand_local_search import rand_local_search
from greedy_hill_climber import greedy_hill_climber
from one_ea import one_ea
from fast_ga import fast_ga
from lamb_ea import lamb_ea
from two_rate_ea import two_rate_ea
from norm_ea import norm_ea
from var_ea import var_ea
from sa_auto import sa_auto
from sars_auto import sars_auto
from umda import umda

# ID to algorithm mapping #
algorithm_map = {
      0:  rand_search(budget=BUDGET),
      1:  rand_local_search(budget=BUDGET),
      2:  greedy_hill_climber(budget=BUDGET),
      3:  (lambda mutation_rate: one_ea(budget=BUDGET, mutation_rate=mutation_rate))(args.algorithm[1] if len(args.algorithm) > 1 else 0.05),
      4:  fast_ga(budget=BUDGET),
      5:  lamb_ea(budget=BUDGET),
      6:  (lambda mutation_rate: two_rate_ea(budget=BUDGET, mutation_rate=mutation_rate))(args.algorithm[1] if len(args.algorithm) > 1 else 0.05),
      7:  (lambda mutation_rate: norm_ea(budget=BUDGET, mutation_rate=mutation_rate))(args.algorithm[1] if len(args.algorithm) > 1 else 0.05),
      8:  (lambda mutation_rate: var_ea(budget=BUDGET, mutation_rate=mutation_rate))(args.algorithm[1] if len(args.algorithm) > 1 else 0.05),
      9:  sa_auto(budget=BUDGET),
      10: (lambda rounds: sars_auto(budget=BUDGET, rounds=rounds))(args.algorithm[1] if len(args.algorithm) > 1 else 5),
      11: (lambda population_size: umda(budget=BUDGET, population_size=population_size))(args.algorithm[1] if len(args.algorithm) > 1 else 50),
  }

# Choosing an algorithm to test between available ones that are either generic or the expected optimized solution called JT_algo.
# For now the default is rand_search.
ALGORITHM = algorithm_map.get(args.algorithm[0], rand_search(budget=BUDGET))

ALGORITHM.printer()
print(f"Algorithm '{ALGORITHM.__class__.__name__}' selected successfully.")
print("")
print("=" * 60)
print("")


### Step 4 ###
print("Step 4:")
print("Investigating the performance of the chosen algorithm on the selected benchmark problem...")

## Adding logger to the experiments ##
def open_logger(algorithm, problem, str=""):
  return ioh.logger.Analyzer(
    root=os.getcwd(),
    folder_name="ioh_logs" + "/" + str + f"/{problem.meta_data.name}_{algorithm}_v",
    algorithm_name=algorithm,
  )

## Benchmarking functions ##
# Benchmarking an algorithm on a single problem class #
# If we want to perform multiple runs with the same objective function, after every run, the problem has to be reset, 
# such that the internal state reflects the current run.

# Running a single experiment #
def run_experiment(problem, algorithm):
  print(f"Running experiment on problem '{problem.meta_data.name}' using algorithm '{algorithm.__class__.__name__}' for {N_RUNS} runs...")

  # Logging the problem
  logger = open_logger(algorithm.__class__.__name__, problem)
  problem.attach_logger(logger)

  for _ in tqdm(range(N_RUNS)):

    # Run the algorithm on the problem
    algorithm(problem)

    # Reset the problem
    problem.reset()

  logger.close()

  # Loading the benchmark data #
  manager = inspector.DataManager()
  manager.add_folder(logger.output_directory)
  results = manager.load(monotonic=True, include_meta_data=True)
  print(manager.overview)

  return results

# Benchmarking an algorithm over every problem class instances #
def benchmark_algo_class(problem, algorithm):
  
  # Logging the problem
  logger = open_logger(algorithm.__class__.__name__, problem)

  # Get all problem IDs in the specified problem class
  problems = problem.meta_data.name[:-4] # Remove the instance number from the name

  print(f"Benchmarking over all instances of problem class: {problems}")

  fids = [k if problems in v else None for k, v in ioh.ProblemClass.GRAPH.problems.items()]
  fids = [fid for fid in fids if fid is not None]
  for fid in fids:
    print(f"Running benchmark on problem ID {fid}...")
    problem = ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH)
    problem.attach_logger(logger)
    for _ in tqdm(range(N_RUNS)):
      algo = algorithm(budget=BUDGET)
      algo(problem)
      problem.reset()

  logger.close()

  # Loading the benchmark data #
  manager = inspector.DataManager()
  manager.add_folder(logger.output_directory)
  results = manager.load(monotonic=True, include_meta_data=True)
  print(manager.overview)

  return results

# Benchmarking a problem class over multiple algorithms #
def benchmark_problem_class(logger, problem):

  print(f"Benchmarking over all algorithms on problem class: {problem.meta_data.name[:-4]}")

  for algorithm_id in algorithm_map.keys():
    print(f"Running benchmark on algorithm ID {algorithm_id}...")
    algorithm = algorithm_map.get(algorithm_id, rand_search(budget=BUDGET))
    logger = open_logger(algorithm, problem)
    problem.attach_logger(logger)
    for _ in tqdm(range(N_RUNS)):
      algorithm(problem)
      problem.reset()
    logger.close()

  # Loading the benchmark data #
  manager = inspector.DataManager()
  manager.add_folder(logger.output_directory)
  results = manager.load(monotonic=True, include_meta_data=True)
  print(manager.overview)

  return results

# Params mode #
# Find the best parameters for each algorithm on multiple problems
def params():

  # Problem instances
  cut = rd.randint(2000, 2004)
  cov = rd.randint(2100, 2139)
  inf = rd.randint(2200, 2223)
  pwt = rd.randint(2300, 2308)
  problem_cut = ioh.get_problem(cut, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH)
  problem_cov = ioh.get_problem(cov, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH)
  problem_inf = ioh.get_problem(inf, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH)
  problem_pwt = ioh.get_problem(pwt, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH)
  problems = [problem_cut, problem_cov, problem_inf]

  # Running experiments
  for problem in problems:
  
    # Algorithm to test
    one_005 = one_ea(budget=BUDGET, mutation_rate=0.05)
    one_01 = one_ea(budget=BUDGET, mutation_rate=0.1)
    one_05 = one_ea(budget=BUDGET, mutation_rate=0.5)
    one_ = [one_005, one_01, one_05]
    
    two_005 = two_rate_ea(budget=BUDGET, mutation_rate=0.05)
    two_01 = two_rate_ea(budget=BUDGET, mutation_rate=0.1)
    two_05 = two_rate_ea(budget=BUDGET, mutation_rate=0.5)
    two_ = [two_005, two_01, two_05]
    
    norm_005 = norm_ea(budget=BUDGET, mutation_rate=0.05)
    norm_01 = norm_ea(budget=BUDGET, mutation_rate=0.1)
    norm_05 = norm_ea(budget=BUDGET, mutation_rate=0.5)
    norm_ = [norm_005, norm_01, norm_05]
    
    var_005 = var_ea(budget=BUDGET, mutation_rate=0.05)
    var_01 = var_ea(budget=BUDGET, mutation_rate=0.1)
    var_05 = var_ea(budget=BUDGET, mutation_rate=0.5)
    var_ = [var_005, var_01, var_05]
    
    sars_5 = sars_auto(budget=BUDGET, rounds=5)
    sars_10 = sars_auto(budget=BUDGET, rounds=10)
    sars_25 = sars_auto(budget=BUDGET, rounds=25)
    sars_ = [sars_5, sars_10, sars_25]

    umda_20 = umda(budget=BUDGET, population_size=20)
    umda_50 = umda(budget=BUDGET, population_size=50)
    umda_100 = umda(budget=BUDGET, population_size=100)
    umda_ = [umda_20, umda_50, umda_100]

    families = [one_, two_, norm_, var_, sars_, umda_]

    for algorithms in families:
      manager = inspector.DataManager()
      for algorithm in algorithms:
        print(f"Testing algorithm '{algorithm.__class__.__name__}' with parameter {algorithm.parameter()} on problem '{problem.meta_data.name}'...")
        param = algorithm.parameter()[2:]
        logger = open_logger(algorithm.__class__.__name__ + f"_{param}", problem, str="test_mode")
        problem.attach_logger(logger)
        for _ in tqdm(range(N_RUNS)):
          algorithm(problem)
          problem.reset()
        manager.add_folder(logger.output_directory)

      print("Test mode experiments completed.")

      # Plotting results
      results = manager.load(monotonic=True, include_meta_data=True)
      # print(manager.overview)
      # print(manager.algorithms)
      # print(manager.functions)

      _, ax = plt.subplots(figsize=(16, 9))
      inspector.plot.single_function_fixedbudget(results.filter(pl.col("function_id").eq(problem.meta_data.problem_id)), ax=ax, maximization=True, measures=['mean'])
      ax.set_xlim(1, 100000)
      ax.set_yscale('linear')
      ax.set_title(f"{algorithm.__class__.__name__} by parameters on {problem.meta_data.name}, budget={BUDGET}, dim={DIMENSION}")
      ax.grid()
      plt.savefig(f"plots/{algorithm.__class__.__name__}_{problem.meta_data.name}.png")

# Test mode #
# Running a tournament to find the best algorithm on multiple problems
def test_mode():
  print("Running in test mode...")
  
  # Problem instances
  problems = []
  # problems.append(ioh.get_problem(2000, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2001, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2002, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2003, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2004, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH))
  problems.append(ioh.get_problem(2100, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2101, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2102, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2103, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2104, instance=1, dimension=DIMENSION, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2200, instance=1, dimension=200, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2201, instance=1, dimension=200, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2202, instance=1, dimension=200, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2203, instance=1, dimension=200, problem_class=ioh.ProblemClass.GRAPH))
  # problems.append(ioh.get_problem(2204, instance=1, dimension=200, problem_class=ioh.ProblemClass.GRAPH))

  # Algorithms in tournament
  algorithms = [rand_search(budget=BUDGET),
                rand_local_search(budget=BUDGET),
                greedy_hill_climber(budget=BUDGET),
                one_ea(budget=BUDGET, mutation_rate=0.05),
                fast_ga(budget=BUDGET),
                lamb_ea(budget=BUDGET),
                two_rate_ea(budget=BUDGET, mutation_rate=0.1),
                norm_ea(budget=BUDGET, mutation_rate=0.1),
                var_ea(budget=BUDGET, mutation_rate=0.05),
                sa_auto(budget=BUDGET),
                sars_auto(budget=BUDGET, rounds=5),
                umda(budget=BUDGET, population_size=100)
               ]

  manager = inspector.DataManager()
  for problem in problems:
    for algorithm in algorithms:
      print(f"Testing algorithm '{algorithm.__class__.__name__}' on problem '{problem.meta_data.name}'...")
      logger = open_logger(algorithm.__class__.__name__, problem, str="test_mode")
      problem.attach_logger(logger)
      for _ in tqdm(range(25)):
        algorithm(problem)
        problem.reset()
      manager.add_folder(logger.output_directory)
  
  print("Test mode experiments completed.")

  results = manager.load(monotonic=True, include_meta_data=True)
  
  _, ax = plt.subplots(figsize=(16, 9))
  inspector.plot.plot_tournament_ranking(results, ax=ax)
  ax.set_title(f"Tournament Ranking, budget={BUDGET}, dim={DIMENSION}")
  ax.grid()
  plt.savefig("plots/tournament_ranking_max_coverage.png")

print("Running the experiments ...")

## Benchmarking ##
if TEST_MODE:
  test_mode()
  results = None
elif BENCHMARK == 0:
  # Single experiment on specified problem
  results = run_experiment(PROBLEM, ALGORITHM)
elif BENCHMARK == 1:
  # Benchmark of an algorithm over every problem's instances
  results = benchmark_algo_class(PROBLEM, ALGORITHM.__class__)
elif BENCHMARK == 2:
  # Benchmark of a problem class over multiple algorithms
  results = benchmark_problem_class(PROBLEM)
else:
  raise ValueError(f"Invalid benchmark option: {BENCHMARK}. Please choose 0, 1, or 2.")

print("")
print("Experiments completed successfully.")
print("")
print("=" * 60)
print("")
print("Results of the experiments:")

if TEST_MODE:
  raise Exception("IGNORE the following plotting code")

## Plotting the results ##
fig, ax = plt.subplots(figsize=(16, 9))
dt_plot = inspector.plot.single_function_fixedbudget(results.filter(pl.col("function_id") == PROBLEM.meta_data.problem_id), ax=ax, maximization=True, measures=['mean'])
ax.set_xlim(1, 100000)
ax.set_yscale('linear')
ax.set_title(f"Fixed Budget Performance on {PROBLEM.meta_data.name[:-4]} Problem using {ALGORITHM.__class__.__name__}")
ax.grid()
print("Plotting the results...")
# plt.savefig("plots/performance_plot.png")
plt.show()