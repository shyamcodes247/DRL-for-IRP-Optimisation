import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ResultsLogger:
    """
        Minimal training-results logger. Creates one timestamped run
        directory under `results_root` and writes:

            <results_root>/<run_name>/
                config.json      -- hyperparameters, written once via `log_config`
                metrics.csv       -- one row per `log_epoch` call, flushed immediately
                checkpoints/      -- created on demand by `checkpoint_path`

        `metrics.csv`'s columns are fixed by the first `log_epoch` call's
        keys (via `csv.DictWriter`); every later call must pass the same
        keys.
    """

    def __init__(self, results_root: str, run_name: Optional[str] = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{run_name}_{timestamp}" if run_name else timestamp
        self.run_dir = Path(results_root) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file = None
        self._csv_writer = None

    def log_config(self, config: Dict[str, Any]) -> None:
        """Writes `config` (e.g. CLI args/hyperparameters) as pretty-printed JSON."""
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Appends one row to `metrics.csv`: `epoch` plus every key in `metrics`.
        The column set is fixed by whichever call opens the file first.
        """
        row = {"epoch": epoch, **metrics}
        if self._csv_writer is None:
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def checkpoint_path(self, name: str) -> str:
        """Returns a path under `<run_dir>/checkpoints/`, creating that directory if needed."""
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return str(checkpoint_dir / name)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
