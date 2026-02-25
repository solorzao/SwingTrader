import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    """Confidence interval from bootstrap resampling."""
    metric_name: str
    point_estimate: float
    ci_lower: float  # 2.5th percentile
    ci_upper: float  # 97.5th percentile
    samples: np.ndarray
    is_significant: bool  # CI doesn't cross zero

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower


def bootstrap_metric(
    data: np.ndarray | pd.Series,
    metric_fn: callable,
    n_samples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap confidence interval for any metric.

    Args:
        data: Input data to resample
        metric_fn: Function that computes the metric from data
        n_samples: Number of bootstrap resamples
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    point_estimate = float(metric_fn(data))

    boot_stats = np.empty(n_samples)
    for i in range(n_samples):
        resample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = metric_fn(resample)

    alpha = (1 - ci_level) / 2
    ci_lower = float(np.percentile(boot_stats, alpha * 100))
    ci_upper = float(np.percentile(boot_stats, (1 - alpha) * 100))

    # Significant if CI doesn't cross zero
    is_significant = (ci_lower > 0) or (ci_upper < 0)

    return BootstrapResult(
        metric_name="",
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        samples=boot_stats,
        is_significant=is_significant,
    )


def bootstrap_sharpe(daily_returns: pd.Series, n_samples: int = 1000) -> BootstrapResult:
    """Bootstrap CI specifically for Sharpe ratio."""
    def sharpe_fn(returns):
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return np.sqrt(252) * np.mean(returns) / np.std(returns)

    result = bootstrap_metric(daily_returns.values, sharpe_fn, n_samples)
    result.metric_name = "sharpe_ratio"
    return result


def bootstrap_returns(daily_returns: pd.Series, n_samples: int = 1000) -> BootstrapResult:
    """Bootstrap CI for total return."""
    def total_return_fn(returns):
        return float(np.prod(1 + returns) - 1)

    result = bootstrap_metric(daily_returns.values, total_return_fn, n_samples)
    result.metric_name = "total_return"
    return result


def bootstrap_win_rate(trade_pnls: np.ndarray, n_samples: int = 1000) -> BootstrapResult:
    """Bootstrap CI for win rate."""
    def win_rate_fn(pnls):
        if len(pnls) == 0:
            return 0.0
        return float((pnls > 0).sum() / len(pnls))

    result = bootstrap_metric(trade_pnls, win_rate_fn, n_samples)
    result.metric_name = "win_rate"
    return result
