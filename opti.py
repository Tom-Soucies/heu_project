"""Summary of the opti algorithm
This algorithm is designed to solve binary optimization problems using a simple iterative approach.
It will be tested on various benchmark problems provided by the ioh library (based on PBO).
"""

### Imports ###
import iohinspector
import ioh
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm 
import matplotlib.pyplot as plt
import os


### Algorithm Class ###
class opti:
    """
    Template for implementing your own binary optimization algorithm.
    
    Modify this class to implement your algorithm idea.
    """
    def __init__(self, budget: int = 50_000, **kwargs):
        self.budget = budget
        # Add your algorithm parameters here
        
    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
        """
        Your algorithm implementation goes here.
        
        Args:
            problem: The binary optimization problem to solve
        """
        # Example: Initialize with random solution
        x = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        problem(x)
        
        # Main optimization loop
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            x = np.random.randint(0, 2, size=problem.meta_data.n_variables)            
            problem(x)