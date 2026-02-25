import pytest
import pandas as pd
import numpy as np
from swing_trader.data.splits import TemporalSplitManager, TemporalSplit


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)}, index=dates)
    y = pd.Series(np.random.choice([-1, 0, 1], n), index=dates)
    return X, y


def test_split_produces_correct_sizes(sample_data):
    X, y = sample_data
    mgr = TemporalSplitManager()
    split = mgr.split(X, y)
    assert split.train_size == 120  # 60% of 200
    assert split.val_size == 20  # 10% of 200
    assert split.test_size == 20  # 10% of 200
    assert split.holdout_size == 40  # remainder (20%)
    assert split.train_size + split.val_size + split.test_size + split.holdout_size == 200


def test_split_preserves_chronological_order(sample_data):
    X, y = sample_data
    mgr = TemporalSplitManager()
    split = mgr.split(X, y)
    assert split.X_train.index[-1] < split.X_val.index[0]
    assert split.X_val.index[-1] < split.X_test.index[0]
    assert split.X_test.index[-1] < split.X_holdout.index[0]


def test_split_stores_correct_boundary_dates(sample_data):
    X, y = sample_data
    mgr = TemporalSplitManager()
    split = mgr.split(X, y)
    assert split.train_end_date == split.X_train.index[-1]
    assert split.val_end_date == split.X_val.index[-1]
    assert split.test_end_date == split.X_test.index[-1]
    assert split.holdout_start_date == split.X_holdout.index[0]


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        TemporalSplitManager(train_ratio=0.5, val_ratio=0.1, test_ratio=0.1, holdout_ratio=0.1)


def test_minimum_samples_required():
    X = pd.DataFrame({"f1": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
    y = pd.Series([0, 1, 0], index=X.index)
    mgr = TemporalSplitManager()
    with pytest.raises(ValueError, match="at least 20 samples"):
        mgr.split(X, y)


def test_split_train_test_backward_compat(sample_data):
    X, y = sample_data
    mgr = TemporalSplitManager()
    X_train, y_train, X_test, y_test, train_end = mgr.split_train_test(X, y)
    assert len(X_train) == 160  # 80% of 200
    assert len(X_test) == 40  # 20% of 200
    assert train_end == X_train.index[-1]


def test_no_data_overlap(sample_data):
    X, y = sample_data
    mgr = TemporalSplitManager()
    split = mgr.split(X, y)
    all_indices = (
        split.X_train.index.tolist() + split.X_val.index.tolist() +
        split.X_test.index.tolist() + split.X_holdout.index.tolist()
    )
    assert len(all_indices) == len(set(all_indices))


def test_integer_index_doesnt_crash():
    X = pd.DataFrame({"f1": np.random.randn(50)})
    y = pd.Series(np.random.choice([-1, 0, 1], 50))
    mgr = TemporalSplitManager()
    split = mgr.split(X, y)
    assert split.train_size > 0
