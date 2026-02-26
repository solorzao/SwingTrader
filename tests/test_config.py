import pytest
from pathlib import Path
from swing_trader.config import (
    AppConfig, BacktestConfig, FeatureConfig,
    LabelConfig, SplitConfig, WalkForwardConfig,
)


class TestFeatureConfig:
    def test_default_values(self):
        cfg = FeatureConfig()
        assert cfg.sma_periods == [10, 20, 50]
        assert cfg.ema_periods == [12, 26]
        assert cfg.rsi_period == 14
        assert cfg.bb_period == 20
        assert cfg.sma is True
        assert cfg.obv is True

    def test_to_indicator_config_format(self):
        cfg = FeatureConfig()
        result = cfg.to_indicator_config()
        assert "features" in result
        assert "params" in result
        assert result["features"]["sma"] is True
        assert result["params"]["rsi_period"] == 14
        assert result["params"]["sma_periods"] == [10, 20, 50]

    def test_to_indicator_config_custom_values(self):
        cfg = FeatureConfig(sma_periods=[5, 15], rsi_period=21, sma=False)
        result = cfg.to_indicator_config()
        assert result["features"]["sma"] is False
        assert result["params"]["sma_periods"] == [5, 15]
        assert result["params"]["rsi_period"] == 21

    def test_to_indicator_config_returns_list_copies(self):
        cfg = FeatureConfig()
        result = cfg.to_indicator_config()
        result["params"]["sma_periods"].append(999)
        assert 999 not in cfg.sma_periods

    def test_from_indicator_config_roundtrip(self):
        original = FeatureConfig(sma_periods=[5, 15], rsi_period=21, sma=False, obv=False)
        config_dict = original.to_indicator_config()
        restored = FeatureConfig.from_indicator_config(config_dict)
        assert restored.sma_periods == [5, 15]
        assert restored.rsi_period == 21
        assert restored.sma is False
        assert restored.obv is False
        assert restored.ema is True  # default preserved

    def test_from_indicator_config_partial_dict(self):
        # UI may not include all params
        config_dict = {
            "features": {"sma": True, "rsi": False},
            "params": {"rsi_period": 10}
        }
        cfg = FeatureConfig.from_indicator_config(config_dict)
        assert cfg.sma is True
        assert cfg.rsi is False
        assert cfg.rsi_period == 10
        assert cfg.ema is True  # defaults for missing keys
        assert cfg.sma_periods == [10, 20, 50]  # default


class TestSplitConfig:
    def test_default_ratios_sum_to_one(self):
        cfg = SplitConfig()
        assert abs(cfg.train + cfg.val + cfg.test + cfg.holdout - 1.0) < 1e-6

    def test_invalid_ratios_raises(self):
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            SplitConfig(train=0.5, val=0.1, test=0.1, holdout=0.1)


class TestWalkForwardConfig:
    def test_default_values(self):
        cfg = WalkForwardConfig()
        assert cfg.mode == "expanding"
        assert cfg.train_window_days == 252

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            WalkForwardConfig(mode="invalid")


class TestAppConfig:
    def test_creates_with_all_defaults(self):
        cfg = AppConfig()
        assert isinstance(cfg.features, FeatureConfig)
        assert isinstance(cfg.labels, LabelConfig)
        assert isinstance(cfg.splits, SplitConfig)
        assert isinstance(cfg.backtest, BacktestConfig)
        assert isinstance(cfg.walk_forward, WalkForwardConfig)
        assert cfg.cache_dir == Path("data/cache")
        assert cfg.models_dir == Path("models")

    def test_custom_sub_configs(self):
        cfg = AppConfig(features=FeatureConfig(rsi_period=21))
        assert cfg.features.rsi_period == 21
        assert cfg.splits.train == 0.6

    def test_independent_instances(self):
        cfg1 = AppConfig()
        cfg2 = AppConfig()
        cfg1.features.rsi_period = 99
        assert cfg2.features.rsi_period == 14
