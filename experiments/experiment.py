from typing import Literal
import numpy as np
from .environment import Env, SPSAMFEnv, StochasticSPSAMFEnv
from .algorithms import epsilon_greedy, epsilon_greedy_decay, ucb1, thompson_sampling_bernoulli, explore_then_commit, sp_ucb1

__all__ = [
    'run_experiment',
    'run_spsamf_experiment',
    'run_strategy_experiment'
]


def run_experiment(num_runs: int, means: list[float], rng: np.random.Generator, total_steps: int, EPSILON: float, EPSILON_START: float, EPSILON_MIN: float, DECAY_RATE: float, ETC_M: int):
    """
    Run a basic experiment on the base ENV to compare algorithms.

    Parameters
    ----------
    num_runs: int
        number of runs to average over
    means: list[float]
        the means to use for the arms
    rng: np.random.Generate
        the random number generator to use
    total_steps: int
        the number of steps to run
    EPSILON: float
        for epsilon greedy
    EPSILON_START: float
        for the decaying epsilon greedy
    EPSILON_MIN: float
        for the decaying epsilon greedy
    DECAY_RATE: float
        for the decaying epsilon greedy
    ETC_M: int
        number of rounds to run the exploration phase of Explore then commit for each arm
    """
    # Dictionary to hold the raw regret histories for all 40 runs per algorithm
    # Format: { algorithm_name: list of lists (shape: 40 runs x total_steps) }
    raw_regrets = {
        "Epsilon-Greedy": [],
        "Epsilon-Greedy Decay": [],
        "UCB1": [],
        "Thompson Sampling": [],
        "Explore-Then-Commit": [],
        # "SAMBA": []
    }
    K = len(means)
    print(f"Running {num_runs} simulations for each algorithm...")

    for run in range(num_runs):
        # 1. Epsilon-Greedy
        env = Env(means, rng)
        epsilon_greedy(K, total_steps, EPSILON, env)
        raw_regrets["Epsilon-Greedy"].append(env.regret_history)

        # 2. Epsilon-Greedy Decay
        env = Env(means, rng)
        epsilon_greedy_decay(K, total_steps, EPSILON_START,
                             EPSILON_MIN, DECAY_RATE, env)
        raw_regrets["Epsilon-Greedy Decay"].append(env.regret_history)

        # 3. UCB1
        env = Env(means, rng)
        ucb1(K, total_steps, env)
        raw_regrets["UCB1"].append(env.regret_history)

        # 4. Thompson Sampling
        env = Env(means, rng)
        thompson_sampling_bernoulli(K, total_steps, env)
        raw_regrets["Thompson Sampling"].append(env.regret_history)

        # 5. Explore-Then-Commit
        env = Env(means, rng)
        explore_then_commit(K, total_steps, ETC_M, env)
        raw_regrets["Explore-Then-Commit"].append(env.regret_history)

        # # 6. SAMBA
        # env = Env(means, arm_algorithm='top')
        # samba(K, total_steps, SAMBA_ALPHA, SAMBA_BETA, env)
        # raw_regrets["SAMBA"].append(np.cumsum(env.regret_history))

    # --- 3. AVERAGE THE RESULTS ---
    averaged_regrets = {}
    for algo_name, regrets_list in raw_regrets.items():
        # Convert lists to a 2D numpy array and average across rows (axis=0)
        averaged_regrets[algo_name] = np.mean(regrets_list, axis=0)

    return averaged_regrets


def run_spsamf_experiment(means: list[float], top_means: list[float], rng: np.random.Generator, noisy: bool, num_runs: int, total_steps: int):
    """
    Run a SP+SAMF experiment to compare the stochastic and original versions
    """

    regret_histories = []
    assert num_runs
    for _ in range(num_runs):
        if noisy:
            env = StochasticSPSAMFEnv(
                means,
                top_means,
                0.01,
                rng,
                "optimal",
            )
        else:
            env = SPSAMFEnv(
                means,
                top_means,
                rng,
                "optimal",
            )

        bids = env.get_bids()
        sp_ucb1(len(means), total_steps, env, bids)
        regret_histories.append(env.regret_history)

    avg_regret = np.mean(np.vstack(regret_histories), axis=0)
    return avg_regret, regret_histories


def run_strategy_experiment(
    means: list[float],
    top_means: list[float],
    rng: np.random.Generator,
    num_runs: int,
    total_steps: int,
    strategy: Literal['optimal'] | Literal['underbid'] | Literal['overbid'],
    amount: float | None = None,
    noisy: bool = False,
    eps: float = 0.01,
):
    """
    Run a SP+SAMF experiment for a specific arm-bidding strategy and return
    averaged platform reward and arm utilities.

    Parameters
    ----------
    means: list[float]
        The true means for each arm.
    top_means: list[float]
        The reported top means for each arm.
    rng: np.random.Generator
        Random number generator to use for the experiment.
    num_runs: int
        Number of independent runs to average over.
    total_steps: int
        Number of rounds to simulate per run.
    strategy: {'optimal', 'underbid', 'overbid'}
        The strategy used by the arms.
    amount: float | None
        Amount to underbid/overbid by when applicable.
    noisy: bool
        Whether to use the stochastic SP+SAMF environment.
    eps: float
        Error tolerance used by the stochastic environment.

    Returns
    -------
    list
        A list of the form [avg_platform_reward, avg_arm_utilities].
    """
    assert num_runs > 0
    if strategy not in {"optimal", "underbid", "overbid"}:
        raise ValueError("strategy must be one of {'optimal', 'underbid', 'overbid'}")
    if strategy != "optimal" and amount is None:
        raise ValueError("amount must be provided for underbid/overbid strategies")

    platform_rewards = []
    utility_histories = []

    for _ in range(num_runs):
        if noisy:
            env = StochasticSPSAMFEnv(
                means,
                top_means,
                eps,
                rng,
                strategy, # pyright: ignore[reportArgumentType]
                amount, # pyright: ignore[reportArgumentType]
            )
        else:
            env = SPSAMFEnv(
                means,
                top_means,
                rng,
                strategy, # pyright: ignore[reportArgumentType]
                amount, # pyright: ignore[reportArgumentType]
            )

        bids = env.get_bids()
        _, _, total_reward = sp_ucb1(len(means), total_steps, env, bids)
        platform_rewards.append(float(total_reward))
        utility_histories.append(np.array(env.arm_utility, dtype=float))

    avg_platform_reward = float(np.mean(platform_rewards))
    avg_arm_utilities = np.mean(np.vstack(utility_histories), axis=0).tolist()
    return [avg_platform_reward, avg_arm_utilities]
