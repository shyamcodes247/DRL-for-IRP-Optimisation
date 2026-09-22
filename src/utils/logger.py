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
                config.json       -- hyperparameters, written once via `log_config`
                <name>.csv         -- one row per `log_metrics(name, ...)` call, flushed
                                      immediately (e.g. "metrics.csv" for per-epoch
                                      training stats, "eval.csv" for periodic greedy
                                      evaluation runs — see `MTPPO.evaluate_episode`)
                checkpoints/       -- created on demand by `checkpoint_path`

        Each named stream's CSV columns are fixed by its first `log_metrics`
        call's keys (via `csv.DictWriter`); every later call to that stream
        must pass the same keys.
    """

    def __init__(self, results_root: str, run_name: Optional[str] = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{run_name}_{timestamp}" if run_name else timestamp
        self.run_dir = Path(results_root) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._csv_files: Dict[str, Any] = {}
        self._csv_writers: Dict[str, Any] = {}

    def log_config(self, config: Dict[str, Any]) -> None:
        """Writes `config` (e.g. CLI args/hyperparameters) as pretty-printed JSON."""
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_metrics(self, name: str, step: int, metrics: Dict[str, Any], step_key: str = "epoch") -> None:
        """
        Appends one row to `<name>.csv`: `step_key` plus every key in `metrics`.
        A given `name` opens (and fixes the columns of) its own CSV file on
        first use.
        """
        row = {step_key: step, **metrics}
        writer = self._csv_writers.get(name)
        if writer is None:
            f = open(self.run_dir / f"{name}.csv", "w", newline="")
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            self._csv_files[name] = f
            self._csv_writers[name] = writer
        writer.writerow(row)
        self._csv_files[name].flush()

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Shorthand for `log_metrics("metrics", epoch, metrics)` (per-epoch training stats)."""
        self.log_metrics("metrics", epoch, metrics)

    def checkpoint_path(self, name: str) -> str:
        """Returns a path under `<run_dir>/checkpoints/`, creating that directory if needed."""
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return str(checkpoint_dir / name)

    def close(self) -> None:
        for f in self._csv_files.values():
            f.close()
        self._csv_files = {}
        self._csv_writers = {}
