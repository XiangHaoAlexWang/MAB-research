import math
import numpy as np
from .environment import Env, StochasticSPSAMFEnv, SPSAMFEnv

__all__ = [
    'epsilon_greedy',
    'epsilon_greedy_decay',
    'ucb1',
    'thompson_sampling_bernoulli',
    'explore_then_commit',
    'sp_ucb1'
]


def epsilon_greedy(K: int, total_steps: int, epsilon: float, env: Env):
    """
    The epsilon-greedy algorithm

    Parameters
    ----------
    K: int
        Number of arms
    total_steps: int
        Number of rounds to play
    epsilon: float
        Probability of exploration (0 <= epsilon <= 1)
    env: Env
        The environment to run the algorithm in.

    Returns
    -------
    values: list[float]
        The estimated values of each arm after the algorithm has run.
    counts: list[int]
        The number of times each arm was pulled.
    total_reward: float
        The total reward accumulated over all rounds.
    """
    # Initialize counts and estimated values for each arm
    counts = [0] * K
    values = [0.0] * K
    total_reward = 0

    for _ in range(total_steps):
        if env.rng.random() < epsilon:
            # Explore: choose a random arm
            chosen_arm = int(env.rng.integers(0, K - 1))
        else:
            # Exploit: choose the best known arm (break ties randomly)
            max_val = max(values)
            best_arms = [i for i, val in enumerate(values) if val == max_val]
            chosen_arm = env.rng.choice(best_arms)

        # Pull the chosen arm and observe reward
        reward = env.pull_arm(chosen_arm)
        total_reward += reward

        # Update empirical average formula: Q_t(a) = Q_{t-1}(a) + (R_t - Q_{t-1}(a)) / N_t(a)
        counts[chosen_arm] += 1
        n = counts[chosen_arm]
        values[chosen_arm] += (reward - values[chosen_arm]) / n

    return values, counts, total_reward


def epsilon_greedy_decay(K: int, total_steps: int, epsilon_start: float,
                         epsilon_min: float, decay_rate: float, env: Env):
    """
    Run the epsilon-greedy algorithm with decaying epsilon.

    Parameters
    ----------
    K: int
        Number of arms
    total_steps: int
        Number of rounds to play
    epsilon_start: float
        Initial exploration probability (typically 1.0)
    epsilon_min: float
        Minimum exploration probability (typically 0.01 to 0.1)
    decay_rate: float
        Multiplicative factor to decrease epsilon each step (typically
        0.99 to 0.999)
    env: Env
        The environment to run in

    Returns
    -------
    values: list[float]
        The estimated values of each arm after the algorithm has run.
    counts: list[int]
        The number of times each arm was pulled.
    total_reward: float
        The total reward accumulated over all rounds.
    epsilon_history: list[float]
        The history of epsilon values over time.
    """
    # Initialize counts and estimated values for each arm
    counts = [0] * K
    values = [0.0] * K
    total_reward = 0
    epsilon = epsilon_start

    # Track epsilon values over time for debugging/plotting
    epsilon_history = []

    for _ in range(total_steps):
        epsilon_history.append(epsilon)
        if env.rng.random() < epsilon:
            # Explore: choose a random arm
            chosen_arm = int(env.rng.integers(0, K - 1))
        else:
            # Exploit: choose the best known arm (break ties randomly)
            max_val = max(values)
            best_arms = [i for i, val in enumerate(values) if val == max_val]
            chosen_arm = env.rng.choice(best_arms)

        # Pull the chosen arm and observe reward
        reward = env.pull_arm(chosen_arm)
        total_reward += reward

        # Update empirical average
        counts[chosen_arm] += 1
        n = counts[chosen_arm]
        values[chosen_arm] += (reward - values[chosen_arm]) / n

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * decay_rate)

    return values, counts, total_reward, epsilon_history


def ucb1(K: int, total_steps: int, env: Env):
    """
    Run the UCB1 algorithm.

    Parameters
    ----------
    K: int
        Number of arms
    total_steps: int
        Number of rounds to play
    env: Env
        Environment object with pull_arm method

    Returns
    -------
    values: list[float]
        The estimated values of each arm after the algorithm has run.
    counts: list[int]
        The number of times each arm was pulled.
    total_reward: float
        The total reward accumulated over all rounds.
    """
    counts = [0] * K
    values = [0.0] * K
    total_reward = 0

    # Try each arm once to initialize
    for arm in range(K):
        reward = env.pull_arm(arm)
        total_reward += reward
        counts[arm] = 1
        values[arm] = reward

    for step in range(K, total_steps):
        # Calculate UCB values for each arm
        ucb_values = []
        for arm in range(K):
            # step represents the total number of plays 't'
            bonus = math.sqrt((math.log(step)) / counts[arm])
            ucb_values.append(values[arm] + bonus)

        # Select the arm with the highest UCB value
        max_ucb = max(ucb_values)
        best_arms = [i for i, val in enumerate(ucb_values) if val == max_ucb]
        chosen_arm = env.rng.choice(best_arms)

        # Pull arm and update
        reward = env.pull_arm(chosen_arm)
        total_reward += reward
        counts[chosen_arm] += 1
        n = counts[chosen_arm]
        values[chosen_arm] += (reward - values[chosen_arm]) / n
    return values, counts, total_reward


def thompson_sampling_bernoulli(K: int, total_steps: int, env: Env):
    """
    Run the Thompson Sampling algorithm for Bernoulli rewards.

    Assumes rewards are binary (0 or 1).

    Parameters
    ----------
    K: int
        Number of arms
    total_steps: int
        Number of rounds to play
    env: Env
        Environment object with pull_arm method

    Returns
    -------
    values: list[float]
        The estimated probabilities of success for each arm after the 
        algorithm has run.
    counts: list[int]
        The number of times each arm was pulled.
    total_reward: float
        The total reward accumulated over all rounds.
    """
    # Initialize Beta prior parameters (alpha = 1, beta = 1 represents
    # uniform prior)
    alphas = [1.0] * K
    betas = [1.0] * K
    total_reward = 0
    counts = [0] * K

    for _ in range(total_steps):
        samples = []
        for arm in range(K):
            # Draw a sample from the Beta distribution for each arm
            samples.append(env.rng.beta(alphas[arm], betas[arm]))

        # Select arm with the highest sample
        chosen_arm = np.argmax(samples)

        # Pull arm (expects 0 or 1)
        reward = env.pull_arm(int(chosen_arm))
        total_reward += reward
        counts[chosen_arm] += 1

        # Update Beta distribution parameters
        if reward == 1:
            alphas[chosen_arm] += 1
        else:
            betas[chosen_arm] += 1

    # Estimated probabilities
    values = [alphas[i] / (alphas[i] + betas[i]) for i in range(K)]
    return values, counts, total_reward


def explore_then_commit(K: int, total_steps: int, m: int, env: Env):
    """
    Run the Explore-Then-Commit algorithm.

    Parameters
    ----------
    K: int
        Number of arms
    total_steps: int
        Number of rounds to play
    m: int
        Number of times to explore each arm initially (total exploration
        phase = K * m)
    env: Env
        Environment object with pull_arm method

    Returns
    -------
    values: list[float]
        The estimated values of each arm after the algorithm has run.
    counts: list[int]
        The number of times each arm was pulled.
    total_reward: float
        The total reward accumulated over all rounds.
    """
    counts = [0] * K
    values = [0.0] * K
    total_reward = 0

    # 1. Exploration Phase: Pull each arm 'm' times
    for arm in range(K):
        for _ in range(m):
            reward = env.pull_arm(arm)
            total_reward += reward
            counts[arm] += 1
            n = counts[arm]
            values[arm] += (reward - values[arm]) / n

    # Find the best arm based on exploration phase
    best_arm = values.index(max(values))

    # 2. Exploitation (Commit) Phase: Pull only the best arm for the
    # remaining steps
    remaining_steps = total_steps - (K * m)
    for _ in range(max(0, remaining_steps)):
        reward = env.pull_arm(best_arm)
        total_reward += reward
        counts[best_arm] += 1
        n = counts[best_arm]
        values[best_arm] += (reward - values[best_arm]) / n

    return values, counts, total_reward


def sp_ucb1(K: int, total_steps: int, env: StochasticSPSAMFEnv | SPSAMFEnv,
            bids: list[float]):
    """
    Run the UCB1 algorithm for SP+SAMF

    Parameters
    ----------
    K: (int)
        Number of arms
    total_steps: (int)
        Number of rounds to play
    env: (StochasticSPSAMFEnv | SPSAMFEnv)
        Environment object with pull_arm method
    bids: list[float]
        List of bid values for each arm

    Returns
    -------
    values: list[float]
        The estimated values of each arm after the algorithm has run.
    counts: list[int]
        The number of times each arm was pulled.
    total_reward: float
        The total reward accumulated over all rounds.
    """
    counts = [0] * K
    values = [0.0] * K
    total_reward = 0
    # Compute m' locally from the environment's top means
    mprime = sorted(getattr(env, "top_means", env.means))[-2]
    # Try each arm once to initialize
    for arm in range(K):
        reward = env.pull_arm(arm)
        total_reward += reward
        counts[arm] = 1
        values[arm] = reward

    for step in range(K, total_steps):
        # Calculate UCB values for each arm
        ucb_values = []
        for arm in range(K):
            # step represents the total number of plays 't'
            bonus = math.sqrt((math.log(step)) / counts[arm])
            ucb_values.append(values[arm] + bonus)

        # Select the arm with the highest UCB value
        max_ucb = max(ucb_values)
        best_arms = [i for i, val in enumerate(ucb_values) if val == max_ucb]
        chosen_arm = env.rng.choice(best_arms)

        # Pull arm and update
        reward = env.pull_arm(chosen_arm)
        total_reward += reward
        counts[chosen_arm] += 1
        n = counts[chosen_arm]
        values[chosen_arm] += (reward - values[chosen_arm]) / n

        # Eliminate arms that underbid during the bidding stage:
        # Define $\widehat \mu_i=\frac{1}{N_i}\sum_{s:I_s=i}Y_{i,s}$,
        # and $b_i=\sqrt{\frac{2}{N_i}}$.
        # If the arm reports $\widehat\mu_i<m'$, but $\widehat
        # \mu_i-b_i>m'+\epsilon$, then the algorithm will penalize or
        # eliminate it.
        if isinstance(env, StochasticSPSAMFEnv):
            for arm in range(K):
                if counts[arm] > 0:
                    estimated_mean = values[arm]
                    bonus = math.sqrt(2 / counts[arm])
                    if estimated_mean - bonus > bids[arm] + env.eps:
                        # Penalize or eliminate the arm
                        # Set its estimated value to -100 to discourage
                        # selection
                        values[arm] = -100
                        # Make count almost infinite minimize exploration term
                        counts[arm] = 1000000000
        else:
            for arm in range(K):
                if counts[arm] > 0:
                    estimated_mean = values[arm]
                    if bids[arm] < mprime and reward > mprime:
                        values[arm] = -100000000
                        counts[arm] = 10000000000
    return values, counts, total_reward
