from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeatureConfig:
    """Technical indicator parameters."""
    sma_periods: list[int] = field(default_factory=lambda: [10, 20, 50])
    ema_periods: list[int] = field(default_factory=lambda: [12, 26])
    rsi_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    stoch_period: int = 14
    # Feature toggles
    sma: bool = True
    ema: bool = True
    rsi: bool = True
    macd: bool = True
    bollinger: bool = True
    atr: bool = True
    obv: bool = True
    stochastic: bool = True

    def to_indicator_config(self) -> dict:
        """Convert to the dict format TechnicalIndicators expects."""
        return {
            "features": {
                "sma": self.sma, "ema": self.ema, "rsi": self.rsi,
                "macd": self.macd, "bollinger": self.bollinger, "atr": self.atr,
                "obv": self.obv, "stochastic": self.stochastic,
            },
            "params": {
                "rsi_period": self.rsi_period,
                "sma_periods": list(self.sma_periods),
                "ema_periods": list(self.ema_periods),
                "atr_period": self.atr_period,
                "bb_period": self.bb_period,
                "stoch_period": self.stoch_period,
            }
        }


@dataclass
class LabelConfig:
    """Signal labeling parameters."""
    forward_days: int = 5
    threshold: float = 0.02


@dataclass
class SplitConfig:
    """Temporal split ratios."""
    train: float = 0.6
    val: float = 0.1
    test: float = 0.1
    holdout: float = 0.2

    def __post_init__(self):
        total = self.train + self.val + self.test + self.holdout
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class BacktestConfig:
    """Backtesting parameters."""
    initial_capital: float = 10_000
    commission: float = 0.001
    slippage: float = 0.0005
    position_size: float = 1.0


@dataclass
class WalkForwardConfig:
    """Walk-forward backtesting parameters."""
    mode: str = "expanding"  # "expanding" or "rolling"
    train_window_days: int = 252  # 1 year
    step_days: int = 21  # 1 month
    retrain_every: int = 1  # retrain every N steps
    min_train_samples: int = 100  # minimum training samples required

    def __post_init__(self):
        if self.mode not in ("expanding", "rolling"):
            raise ValueError(f"mode must be 'expanding' or 'rolling', got '{self.mode}'")


@dataclass
class AppConfig:
    """Top-level application configuration."""
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
