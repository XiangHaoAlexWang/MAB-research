import numpy as np

from github.experiments.experiment import run_strategy_experiment


def test_run_strategy_experiment_returns_average_metrics():
    rng = np.random.default_rng(0)
    means = [0.3, 0.4, 0.5]
    top_means = [0.3, 0.4, 0.5]

    result = run_strategy_experiment(
        means=means,
        top_means=top_means,
        rng=rng,
        num_runs=2,
        total_steps=10,
        strategy="optimal",
    )

    assert isinstance(result, list)
    assert len(result) == 2

    platform_reward, arm_utilities = result
    assert isinstance(platform_reward, float)
    assert len(arm_utilities) == len(means)
    assert all(isinstance(value, float) for value in arm_utilities)
