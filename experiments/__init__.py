"""
Experiments: code that will be run for the experiments
"""

__all__ = [
    # .util
    'sample_bernoulli',
    # .environment
    'Env',
    'SPSAMFEnv',
    'StochasticSPSAMFEnv',
    # .algorithms
    'epsilon_greedy',
    'epsilon_greedy_decay',
    'ucb1',
    'explore_then_commit',
    'thompson_sampling_bernoulli',
    'sp_ucb1',
    # .experiment
    'run_experiment',
    'run_spsamf_experiment',
    'run_strategy_experiment'
]

from .util import *
from .environment import *
from .algorithms import *
from .experiment import *
