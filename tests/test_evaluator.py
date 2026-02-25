import pytest
import pandas as pd
import numpy as np
from swing_trader.evaluation.evaluator import ModelEvaluator, EvaluationReport, Metrics
from swing_trader.data.splits import TemporalSplitManager
from swing_trader.models.random_forest import RandomForestModel


@pytest.fixture
def trained_model_and_split():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
    }, index=dates)
    y = pd.Series(np.random.choice([-1, 0, 1], n), index=dates)

    splitter = TemporalSplitManager()
    split = splitter.split(X, y)

    model = RandomForestModel(n_estimators=10, random_state=42)
    model.fit(split.X_train, split.y_train)

    return model, split


def test_evaluate_returns_report(trained_model_and_split):
    model, split = trained_model_and_split
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, split)

    assert isinstance(report, EvaluationReport)
    assert isinstance(report.train_metrics, Metrics)
    assert isinstance(report.test_metrics, Metrics)
    assert 0 <= report.train_metrics.accuracy <= 1
    assert 0 <= report.test_metrics.accuracy <= 1


def test_overfit_detection(trained_model_and_split):
    model, split = trained_model_and_split
    evaluator = ModelEvaluator(overfit_threshold=0.10)
    report = evaluator.evaluate(model, split)

    # Overfit score = train_f1 - test_f1
    expected_score = report.train_metrics.f1_macro - report.test_metrics.f1_macro
    assert abs(report.overfit_score - expected_score) < 1e-6
    assert report.is_overfit == (report.overfit_score > 0.10)


def test_class_distribution(trained_model_and_split):
    model, split = trained_model_and_split
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, split)

    assert isinstance(report.class_distribution_train, dict)
    assert isinstance(report.class_distribution_test, dict)
    assert sum(report.class_distribution_train.values()) == split.train_size


def test_summary_output(trained_model_and_split):
    model, split = trained_model_and_split
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, split)
    summary = report.summary()
    assert "Accuracy" in summary
    assert "F1 (macro)" in summary
    assert "Overfit score" in summary
