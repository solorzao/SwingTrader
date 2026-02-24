"""Training metadata management for tracking data ranges and preventing overlap."""
import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional


@dataclass
class TrainingInfo:
    """Metadata about a training run's data configuration."""
    train_start: str  # YYYY-MM-DD
    train_end: str  # YYYY-MM-DD
    tickers: list[str]
    model_type: str
    model_filename: str
    forward_days: int = 5  # Label look-ahead period
    trained_at: str = ""  # ISO timestamp of training run

    @property
    def train_start_date(self) -> date:
        return date.fromisoformat(self.train_start)

    @property
    def train_end_date(self) -> date:
        return date.fromisoformat(self.train_end)

    @property
    def safe_backtest_start(self) -> date:
        """Earliest safe backtest start date (train_end + forward_days gap)."""
        return self.train_end_date + timedelta(days=self.forward_days)


class TrainingMetadataStore:
    """Persist and query training metadata to prevent train/test overlap."""

    FILENAME = "training_metadata.json"

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self._path = self.models_dir / self.FILENAME

    def save(self, info: TrainingInfo) -> None:
        """Save training metadata, appending to existing records."""
        records = self._load_all()

        # Replace if same model filename exists, otherwise append
        records = [r for r in records if r["model_filename"] != info.model_filename]
        if not info.trained_at:
            info.trained_at = datetime.now().isoformat()
        records.append(asdict(info))

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(records, indent=2))

    def get_latest(self) -> Optional[TrainingInfo]:
        """Get the most recently trained model's metadata."""
        records = self._load_all()
        if not records:
            return None
        # Sort by trained_at descending
        records.sort(key=lambda r: r.get("trained_at", ""), reverse=True)
        return TrainingInfo(**records[0])

    def get_for_model(self, model_filename: str) -> Optional[TrainingInfo]:
        """Get metadata for a specific model file."""
        records = self._load_all()
        for r in records:
            if r["model_filename"] == model_filename:
                return TrainingInfo(**r)
        return None

    def get_all(self) -> list[TrainingInfo]:
        """Get all training metadata records."""
        return [TrainingInfo(**r) for r in self._load_all()]

    def check_overlap(
        self,
        backtest_start: str | date,
        backtest_end: str | date,
        model_filename: Optional[str] = None,
    ) -> dict:
        """
        Check if a backtest date range overlaps with training data.

        Returns a dict with:
            - overlaps: bool
            - overlap_days: int (number of overlapping days)
            - safe_start: str (earliest safe backtest start date)
            - message: str (human-readable description)
        """
        if isinstance(backtest_start, str):
            backtest_start = date.fromisoformat(backtest_start)
        if isinstance(backtest_end, str):
            backtest_end = date.fromisoformat(backtest_end)

        if model_filename:
            info = self.get_for_model(model_filename)
        else:
            info = self.get_latest()

        if info is None:
            return {
                "overlaps": False,
                "overlap_days": 0,
                "safe_start": backtest_start.isoformat(),
                "message": "No training metadata found — cannot verify data separation.",
            }

        train_start = info.train_start_date
        train_end = info.train_end_date
        safe_start = info.safe_backtest_start

        # Calculate overlap: backtest must start after train_end + forward_days
        overlap_start = max(backtest_start, train_start)
        overlap_end = min(backtest_end, train_end)
        overlap_days = max(0, (overlap_end - overlap_start).days + 1)

        # Also check the gap period (forward_days after train_end)
        gap_end = safe_start
        if backtest_start < gap_end and overlap_days == 0:
            # Data is in the gap zone (label leakage risk)
            gap_overlap_days = (gap_end - backtest_start).days
            return {
                "overlaps": True,
                "overlap_days": gap_overlap_days,
                "safe_start": safe_start.isoformat(),
                "message": (
                    f"Backtest starts {gap_overlap_days} day(s) inside the label gap zone. "
                    f"Training labels look {info.forward_days} days ahead from the training "
                    f"end date ({info.train_end}). "
                    f"Safe backtest start: {safe_start.isoformat()}"
                ),
            }

        if overlap_days > 0:
            return {
                "overlaps": True,
                "overlap_days": overlap_days,
                "safe_start": safe_start.isoformat(),
                "message": (
                    f"Backtest overlaps with training data by {overlap_days} day(s). "
                    f"Training period: {info.train_start} to {info.train_end}. "
                    f"Safe backtest start: {safe_start.isoformat()}"
                ),
            }

        return {
            "overlaps": False,
            "overlap_days": 0,
            "safe_start": safe_start.isoformat(),
            "message": "No overlap detected — backtest data is clean.",
        }

    def suggest_backtest_range(
        self, model_filename: Optional[str] = None
    ) -> Optional[dict]:
        """
        Suggest a non-overlapping backtest date range based on training metadata.

        Returns dict with 'start' and 'end' dates, or None if no metadata.
        """
        if model_filename:
            info = self.get_for_model(model_filename)
        else:
            info = self.get_latest()

        if info is None:
            return None

        start = info.safe_backtest_start
        end = date.today()

        if start >= end:
            return None

        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
        }

    def _load_all(self) -> list[dict]:
        """Load all records from the metadata file."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text())
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
