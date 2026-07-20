from typing import Literal, overload
import math
import numpy as np
from .util import sample_bernoulli

__all__ = [
    'Env',
    'SPSAMFEnv',
    'StochasticSPSAMFEnv'
]


class Env:
    """
    An environment to run the algorithms

    Parameters
    ----------
    means: list[float]
        The base means
    rng: np.random.Generator
        The random number generator to use.

    Attributes
    ----------
    means: list[float]
        The base means
    algorithm: Literal['top']
        The algorithm to use for the arms.
    pull_history: list[int]
        The history of the arms pulled.
    regret_history: list[float] 
        The history of the regret.
    rng: np.random.Generator
        The random number generator that is used.
    """

    def __init__(self, means: list[float], rng: np.random.Generator):
        self.means = means
        self.pull_history = []
        self.regret_history = []
        self.regret = 0
        self.rng = rng
        self.top = max(self.means)

    def pull_arm(self, arm: int):
        """
        Pull the selected arm, using a bernoulli distribution with the mean of the arm as the probability of success.

        Parameters
        ----------
        arm: int
            The arm to pull.

        Returns
        -------
        float
            The reward from the arm.
        """
        self.pull_history.append(arm)
        self.regret += self.top - self.means[arm]
        self.regret_history.append(self.regret)
        return sample_bernoulli(self.means[arm], self.rng)


class SPSAMFEnv(Env):
    """
    An SP+SAMF environment. This environment runs the original SP+SAMF algorithm in a noisy environment.

    Parameters
    ----------
    means: list[float]
        The means for each arm
    top_means: list[float]
        The top means for each arm
    rng: np.random.Generator
        The random number generator to use for the environment
    strategy: 'optimal' | 'underbid' | 'overbid'
        The strategy for the arms to follow
    amount: float
        The amount to overbid or underbid by

    Attributes
    ----------
    means: list[float]
        The means for each arm
    top_means: list[float]
        The top means for each arm
    strategy: 'optimal' | 'underbid' | 'overbid'
        The strategy for the arms to follow
    amount: float
        The amount to overbid or underbid by
    pull_history: list[int]
        The history of the arms pulled.
    arm_utility_history: list[list[float]]
        The history of the utilities for each arm.
    arm_utility: list[float]
        The arms' utilities
    mprime: float
        The second highest mean for the arms.
    regret: float
        The regret of the platform.
    regret_history: list[float]
        The history of the regret of the platform.

    Raises
    ------
    AssertionError when you pass both 'optimal' and a value, or when the strategy is invalid.
    """
    @overload
    def __init__(self, means: list[float], top_means: list[float], rng: np.random.Generator, strategy: Literal['underbid'] | Literal['overbid'], amount: float):
        ...

    @overload
    def __init__(self, means: list[float], top_means: list[float], rng: np.random.Generator, strategy: Literal['optimal']):
        ...

    def __init__(self, means: list[float], top_means: list[float], rng: np.random.Generator, strategy: Literal['optimal'] | Literal['underbid'] | Literal['overbid'], amount: float | None = None):
        # must be one of the three choices
        assert (strategy in ('optimal', 'underbid', 'overbid'))
        # cannot be both optimal strategy and give amount.
        assert not ((strategy == 'optimal') and amount)
        super().__init__(means, rng)
        self.top_means = top_means
        self.strategy = strategy
        self.arm_utility_history: list[list[float]] = [[0] * len(self.means)]
        self.arm_utility = [0.] * len(self.means)
        self.amount = amount
        self.mprime = sorted(self.top_means)[-2]
        self.use_means = []
        self.top = max(top_means)
        for i in self.top_means:
            if i <= self.mprime:
                self.use_means.append(i)
            else:
                if self.strategy == 'optimal':
                    self.use_means.append(
                        min(i, self.mprime + math.log(len(self.means))))
                elif self.strategy == 'underbid':
                    assert self.amount
                    if i == self.mprime:
                        self.use_means.append(i - self.amount)
                    else:
                        self.use_means.append(
                            min(i, self.mprime + math.log(len(self.means))))
                elif self.strategy == 'overbid':
                    assert self.amount
                    if i == self.mprime:
                        self.use_means.append(i + self.amount)
                    else:
                        self.use_means.append(
                            min(i, self.mprime + math.log(len(self.means))))

    def get_bids(self) -> list[float]:
        """
        Get the bids for all of the arms.

        Returns
        -------
        A list containing the floats for the bids
        """
        l = []
        for arm in range(len(self.means)):
            l.append(self.pull_arm(arm))
        return l

    def pull_arm(self, arm: int) -> Literal[0] | Literal[1]:
        """
        Pull the selected arm

        Parameters
        ----------
        arm: (int)
            Which arm to pull

        Returns
        -------
        The result of the pull
        """
        self.arm_utility[arm] += 1
        self.arm_utility[arm] -= self.use_means[arm] - self.means[arm]
        self.arm_utility_history.append(self.arm_utility)
        self.pull_history.append(arm)
        self.regret += self.top - self.use_means[arm]
        self.regret_history.append(self.regret)
        return sample_bernoulli(self.use_means[arm], self.rng)


class StochasticSPSAMFEnv(Env):
    """
    A Stochastic SP+SAMF environment

    Parameters
    ----------
    means: list[float]
        The means for each arm
    top_means: list[float]
        The top means for each arm
    eps: float
        The error interval to use
    rng: np.random.Generator
        The random number generator to use for the environment
    strategy: 'optimal' | 'underbid' | 'overbid'
        The strategy for the arms to follow
    amount: float
        The amount to overbid or underbid by

    Attributes
    ----------
    means: list[float]
        The means for each arm
    top_means: list[float]
        The top means for each arm
    strategy: 'optimal' | 'underbid' | 'overbid'
        The strategy for the arms to follow
    amount: float
        The amount to overbid or underbid by
    pull_history: list[int]
        The history of the arms pulled.
    arm_utility_history: list[list[float]]
        The history of the utilities for each arm.
    arm_utility: list[float]
        The arms' utilities
    mprime: float
        The second highest mean for the arms.
    regret: float
        The regret of the platform.
    regret_history: list[float]
        The history of the regret of the platform.
    eps: float
        The confidence interval in use.

    Raises
    ------
    AssertationError when you pass both 'optimal' and a value, or when the strategy is invalid.
    """
    @overload
    def __init__(self, means: list[float], top_means: list[float], eps, rng: np.random.Generator, strategy: Literal['underbid'] | Literal['overbid'], amount: float):
        ...

    @overload
    def __init__(self, means: list[float], top_means: list[float], eps, rng: np.random.Generator, strategy: Literal['optimal']):
        ...

    def __init__(self, means: list[float], top_means: list[float], eps, rng: np.random.Generator, strategy: Literal['optimal'] | Literal['underbid'] | Literal['overbid'], amount: float | None = None):
        # must be one of the three choices
        assert (strategy in ('optimal', 'underbid', 'overbid'))
        # cannot be both optimal strategy and give amount.
        assert not ((strategy == 'optimal') and amount)
        super().__init__(means, rng)
        self.top_means = top_means
        self.strategy = strategy
        self.arm_utility_history: list[list[float]] = [[0] * len(self.means)]
        self.arm_utility = [0.] * len(self.means)
        self.amount = amount
        self.mprime = sorted(self.top_means)[-2]
        self.use_means = []
        self.eps = eps
        self.top = max(self.top_means)
        for i in self.top_means:
            if i <= self.mprime:
                self.use_means.append(i)
            else:
                if self.strategy == 'optimal':
                    self.use_means.append(
                        min(i, self.mprime + math.log(len(self.means))))
                elif self.strategy == 'underbid':
                    assert self.amount
                    if i == self.mprime:
                        self.use_means.append(i - self.amount)
                    else:
                        self.use_means.append(
                            min(i, self.mprime + math.log(len(self.means))))
                elif self.strategy == 'overbid':
                    assert self.amount
                    if i == self.mprime:
                        self.use_means.append(i + self.amount)
                    else:
                        self.use_means.append(
                            min(i, self.mprime + math.log(len(self.means))))

    def get_bids(self) -> list[float]:
        """
        Get the bids for all of the arms.

        Returns
        -------
        A list containing the floats for the bids
        """
        l = []
        pulls_per_arm = max(1, math.ceil(math.sqrt(2 / (self.eps ** 2))))

        for arm_idx, _ in enumerate(self.use_means):
            sum = 0
            for _ in range(pulls_per_arm):
                sum += self.pull_arm(arm_idx)
            l.append(sum / pulls_per_arm)

        return l

    def pull_arm(self, arm: int) -> Literal[0] | Literal[1]:
        """
        Pull the selected arm

        Parameters
        ----------
        arm: (int)
            Which arm to pull

        Returns
        -------
        The result of the pull
        """
        self.arm_utility[arm] += 1
        self.arm_utility[arm] -= self.use_means[arm] - self.means[arm]
        self.arm_utility_history.append(self.arm_utility)
        self.pull_history.append(arm)
        self.regret += self.top - self.use_means[arm]
        self.regret_history.append(self.regret)
        return sample_bernoulli(self.use_means[arm], self.rng)
