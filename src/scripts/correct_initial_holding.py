"""
Back-fills the period-0 opening-stock holding cost into evaluation CSVs that
were written before `IRPEnv` charged it (see `IRPEnv.initial_holding_cost`).

`val.csv` and `eval.csv` hold costs produced by `MTPPO.evaluate_episode`, so
every row written before that term existed understates `inv_cost`,
`total_cost` and `total_cost_best_route` by exactly

    h_0 * B_0  +  sum_i h_i * I_i^0

for its instance. Because every factor comes from the instance file, the
shortfall is a per-instance constant and the rows can be corrected
arithmetically -- there is no need to reload 66 checkpoints and re-run
evaluation. That equivalence is not assumed: it was checked against a fresh
re-evaluation of one full epoch and agreed to floating-point exactness.

`metrics.csv` is deliberately left alone. It records training rewards, and the
term is excluded from `r_inv` by design, so those numbers were never short.
`routes.csv` holds distances only.

Writing a `.initial_holding_applied` stamp into each run directory makes this
safe to re-run: a second pass would otherwise double-count the correction,
which is the one way this script can silently corrupt data.

Usage:
    python src/scripts/correct_initial_holding.py --run-dir src/results/seed0_...
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..")
for _p in (_SRC_DIR, os.path.join(_SRC_DIR, "agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from environment.irp_env import IRPEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAMP = ".initial_holding_applied"
COST_COLUMNS = ("inv_cost", "total_cost", "total_cost_best_route")
TARGET_FILES = ("val.csv", "eval.csv")


def constant_for(instance: str, config: dict, cache: Dict[str, float]) -> float:
    """
    The period-0 holding cost for one instance, as named in a CSV's `instance`
    column (e.g. `Instances_lowcost_H3/abs5n30`). Cached because a CSV names
    the same instance once per evaluated epoch.
    """
    if instance not in cache:
        env = IRPEnv(
            data_file_path=str(PROJECT_ROOT / "data" / f"{instance}.dat"),
            loc_dim=config["loc_dim"],
            lookback_window=config["lookback_window"],
            product_price=config["product_price"],
            penalty_factor=config["penalty_factor"],
            delivery_cost=config["delivery_cost"],
        )
        cache[instance] = float(env.initial_holding_cost)
    return cache[instance]


def correct_run(run_dir: Path, dry_run: bool) -> None:
    stamp = run_dir / STAMP
    if stamp.exists():
        print(f"{run_dir.name}: already corrected ({STAMP} present), skipping")
        return

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cache: Dict[str, float] = {}
    touched = []

    for name in TARGET_FILES:
        path = run_dir / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)

        missing = [c for c in COST_COLUMNS if c not in (fieldnames or [])]
        if missing:
            raise ValueError(f"{path} lacks expected column(s): {missing}")

        total = 0.0
        for row in rows:
            const = constant_for(row["instance"], config, cache)
            total += const
            for column in COST_COLUMNS:
                row[column] = repr(float(row[column]) + const)

        print(f"  {name}: {len(rows)} rows, +{total / max(len(rows), 1):.2f} mean")
        touched.append((path, fieldnames, rows))

    if dry_run:
        print(f"{run_dir.name}: dry run, nothing written")
        return

    for path, fieldnames, rows in touched:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    stamp.write_text(
        "val.csv and eval.csv have had the period-0 opening-stock holding cost "
        "(h_0*B_0 + sum_i h_i*I_i^0) added to inv_cost, total_cost and "
        "total_cost_best_route. Delete this file only if those columns are "
        "restored to their pre-correction values.\n",
        encoding="utf-8",
    )
    print(f"{run_dir.name}: corrected {len(touched)} file(s), stamped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for run in args.run_dir:
        correct_run(Path(run).resolve(), args.dry_run)


if __name__ == "__main__":
    main()
