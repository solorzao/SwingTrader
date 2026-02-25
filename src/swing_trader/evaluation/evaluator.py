import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from swing_trader.data.splits import TemporalSplit
from swing_trader.models.base import BaseModel
from swing_trader.backtest.engine import BacktestResult
from swing_trader.evaluation.bootstrap import (
    bootstrap_sharpe, bootstrap_returns, bootstrap_win_rate, BootstrapResult,
)


@dataclass
class Metrics:
    """Classification metrics for a single dataset split."""
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float


@dataclass
class EvaluationReport:
    """Comprehensive model evaluation report."""
    train_metrics: Metrics
    test_metrics: Metrics
    overfit_score: float  # train_f1 - test_f1
    is_overfit: bool  # overfit_score > threshold
    class_distribution_train: dict[int, int]
    class_distribution_test: dict[int, int]
    confidence_intervals: dict[str, BootstrapResult] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Model Evaluation Report",
            "=" * 40,
            f"{'Metric':<20} {'Train':>10} {'Test':>10}",
            "-" * 40,
            f"{'Accuracy':<20} {self.train_metrics.accuracy:>10.2%} {self.test_metrics.accuracy:>10.2%}",
            f"{'F1 (macro)':<20} {self.train_metrics.f1_macro:>10.2%} {self.test_metrics.f1_macro:>10.2%}",
            f"{'Precision (macro)':<20} {self.train_metrics.precision_macro:>10.2%} {self.test_metrics.precision_macro:>10.2%}",
            f"{'Recall (macro)':<20} {self.train_metrics.recall_macro:>10.2%} {self.test_metrics.recall_macro:>10.2%}",
            "-" * 40,
            f"Overfit score: {self.overfit_score:.2%} {'⚠ OVERFIT' if self.is_overfit else '✓ OK'}",
        ]
        return "\n".join(lines)


class ModelEvaluator:
    """Evaluate models with train/test comparison and overfitting detection."""

    def __init__(self, overfit_threshold: float = 0.10):
        """
        Args:
            overfit_threshold: Flag overfitting if train_f1 - test_f1 exceeds this
        """
        self.overfit_threshold = overfit_threshold

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
        """Compute classification metrics."""
        return Metrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        )

    def evaluate(
        self,
        model: BaseModel,
        split: TemporalSplit,
    ) -> EvaluationReport:
        """Evaluate a model on train and test splits.

        Args:
            model: Fitted model
            split: TemporalSplit with train/test data
        """
        # Train predictions
        train_preds = model.predict(split.X_train)
        y_train = split.y_train.values[-len(train_preds):]
        train_metrics = self._compute_metrics(y_train, train_preds)

        # Test predictions
        test_preds = model.predict(split.X_test)
        y_test = split.y_test.values[-len(test_preds):]
        test_metrics = self._compute_metrics(y_test, test_preds)

        overfit_score = train_metrics.f1_macro - test_metrics.f1_macro

        # Class distributions
        def class_dist(y):
            values, counts = np.unique(y, return_counts=True)
            return {int(v): int(c) for v, c in zip(values, counts)}

        return EvaluationReport(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            overfit_score=overfit_score,
            is_overfit=overfit_score > self.overfit_threshold,
            class_distribution_train=class_dist(split.y_train.values),
            class_distribution_test=class_dist(split.y_test.values),
        )

    def add_backtest_confidence_intervals(
        self,
        report: EvaluationReport,
        backtest_result: BacktestResult,
        n_samples: int = 1000,
    ) -> EvaluationReport:
        """Add bootstrap confidence intervals from backtest results."""
        cis = {}

        if len(backtest_result.daily_returns) > 10:
            cis["sharpe_ratio"] = bootstrap_sharpe(
                backtest_result.daily_returns, n_samples
            )
            cis["total_return"] = bootstrap_returns(
                backtest_result.daily_returns, n_samples
            )

        if len(backtest_result.trades) > 5:
            cis["win_rate"] = bootstrap_win_rate(
                backtest_result.trades["pnl"].values, n_samples
            )

        report.confidence_intervals = cis
        return report
