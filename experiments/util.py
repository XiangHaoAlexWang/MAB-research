from typing import Literal
import numpy as np

__all__ = [
    'sample_bernoulli'
]

def sample_bernoulli(mean: float, rng: np.random.Generator) -> Literal[0] | Literal[1]:
    """
    Samples from a bernoulli distribution

    Parameters
    ----------
    mean: float
        The mean of the distribution to sample from
    rng: np.random.Generator
        The random number generator to use

    Returns
    -------
    value: Literal[0] | Literal[1]
        The result of the distribution
    """
    return 0 if rng.random() >= mean else 1