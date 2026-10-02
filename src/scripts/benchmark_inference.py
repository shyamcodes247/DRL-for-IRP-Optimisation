"""
Measures how long a trained MTPPO policy takes to produce a solution, for
comparison against the branch-and-cut CPU times Archetti et al. (2007)
report in their Table I.

The two numbers answer different questions, and the comparison is only
honest if that is stated alongside it:

  - Archetti et al. report time to *prove optimality* (or hit a 2-hour
    cutoff). The answer is exact.
  - This script reports time for a *single greedy forward pass* of a trained
    policy. The answer is feasible but suboptimal, so the reported cost gap
    is part of the result, not a footnote to it.
  - Their hardware was a single-core 2.8 GHz Pentium IV (2007). We therefore
    pin torch to one thread, which removes the thread-count difference but
    not the ~two decades of single-core improvement. Raw speedup ratios
    overstate the algorithmic difference by whatever that factor is.

Diagnostics are switched off during timing (`collect_diagnostics=False`):
the 2-opt reference tours the evaluation path normally builds cost far more
than the forward passes they exist to measure, and they are not part of
producing a solution. Costs are measured in a separate, untimed pass with
diagnostics on, so the quality figures carry the full breakdown.

Usage:
    python src/scripts/benchmark_inference.py \
        --run-dir src/results/seed0_20261001_172956 \
        --run-dir src/results/seed1_20261001_172956 \
        --run-dir src/results/seed2_20261001_172955
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..")
_AGENT_DIR = os.path.join(_SRC_DIR, "agent")
for _p in (_SRC_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from environment.irp_env import IRPEnv
from training.mtppo import MTPPO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_TIMES = PROJECT_ROOT / "data" / "benchmarks" / "archetti2007_vmir_ou_times.csv"


def load_published_times(path: Path) -> Dict[tuple, float]:
    """Reads the Archetti et al. (2007) Table I times, keyed (cost_class, H, n)."""
    times: Dict[tuple, float] = {}
    with open(path, encoding="utf-8") as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        for row in rows:
            times[(row["cost_class"], int(row["H"]), int(row["n"]))] = float(row["seconds"])
    return times


def describe_instance(path: str) -> Dict[str, Any]:
    """
    Pulls the instance's cost class, horizon and retailer count out of its
    path. The folder carries the cost class and horizon
    (`Instances_lowcost_H6`), the filename the replicate and size
    (`abs5n30.dat`), so both are parsed rather than inferred from the data.
    """
    parent, stem = Path(path).parent.name, Path(path).stem
    folder = re.match(r"Instances_(lowcost|highcost)_H(\d+)", parent)
    name = re.match(r"abs(\d+)n(\d+)", stem)
    if not folder or not name:
        raise ValueError(f"cannot parse cost class / size from {path!r}")
    return {
        "instance": f"{parent}/{stem}",
        "cost_class": folder.group(1),
        "H": int(folder.group(2)),
        "replicate": int(name.group(1)),
        "n": int(name.group(2)),
    }


def build_agent(config: Dict[str, Any], checkpoint: Path, device: str) -> MTPPO:
    """
    Rebuilds the agent exactly as `train.py` built it, from the run's own
    logged config, then loads the checkpoint's weights. Deriving the feature
    widths here instead of reading them back from the checkpoint would let
    this script and training drift apart silently.
    """
    loc_dim, lookback = config["loc_dim"], config["lookback_window"]
    agent = MTPPO(
        node_feature_dims={
            "critic": loc_dim + 6,
            "inventory": loc_dim + 3,
            "routing": 2 * loc_dim + 2,
        },
        history_dim=1 + 2 * lookback,
        global_feature_dim=2,
        gin_dims=config["gin_dims"],
        mlp_dims=config["mlp_dims"],
        embed_dim=loc_dim + 2 * lookback,
        loc_dim=loc_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        clip_eps=config["clip_eps"],
        value_coef=config["value_coef"],
        entropy_coef=config["entropy_coef"],
        max_grad_norm=config["max_grad_norm"],
        skip_dead_zone=config["skip_dead_zone"],
        device=device,
    )
    agent.load(str(checkpoint))
    return agent


def read_manifest(manifest: Path, data_root: Path) -> List[str]:
    """Resolves a split manifest's relative lines against the data root."""
    paths = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.append(str((data_root / line).resolve()))
    return paths


def time_instance(agent: MTPPO, env: IRPEnv, repeats: int) -> Dict[str, float]:
    """
    Times `repeats` greedy solves of one instance, after a discarded warm-up
    pass. The warm-up matters: the first call through a fresh graph pays for
    lazy allocation and kernel selection, which is a one-off startup cost
    rather than the per-solution cost being reported.

    Reports the median as the headline (robust to a scheduler hiccup mid-run)
    alongside min and mean, so an unstable measurement is visible rather than
    averaged into something that looks precise.
    """
    agent.evaluate_episode(env, collect_diagnostics=False)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        agent.evaluate_episode(env, collect_diagnostics=False)
        samples.append(time.perf_counter() - start)

    return {
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "min_s": min(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def benchmark_run(run_dir: Path, repeats: int, device: str,
                  checkpoint_name: str) -> List[Dict[str, Any]]:
    """Times every instance in one run's held-out evaluation split."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")

    data_root = Path(config["data_root"])
    eval_paths = read_manifest(
        PROJECT_ROOT / config["eval_manifest"], data_root
    )

    load_start = time.perf_counter()
    agent = build_agent(config, checkpoint, device)
    load_s = time.perf_counter() - load_start

    rows = []
    for path in eval_paths:
        meta = describe_instance(path)

        env_start = time.perf_counter()
        env = IRPEnv(
            data_file_path=path,
            loc_dim=config["loc_dim"],
            lookback_window=config["lookback_window"],
            product_price=config["product_price"],
            penalty_factor=config["penalty_factor"],
            delivery_cost=config["delivery_cost"],
        )
        env_s = time.perf_counter() - env_start

        timing = time_instance(agent, env, repeats)
        # Quality comes from a separate pass with diagnostics on, so it is
        # never inside a timed region.
        quality = agent.evaluate_episode(env, collect_diagnostics=True)

        rows.append({
            "run": run_dir.name,
            "seed": config["seed"],
            **meta,
            **timing,
            "env_build_s": env_s,
            "model_load_s": load_s,
            "total_cost": quality["total_cost"],
            "inv_cost": quality["inv_cost"],
            "routing_cost": quality["routing_cost"],
            "total_cost_best_route": quality["total_cost_best_route"],
        })
        print(f"  {meta['instance']:<34} {timing['median_s'] * 1000:8.1f} ms"
              f"   cost {quality['total_cost']:10.1f}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", action="append", required=True,
                        help="A results directory containing config.json and checkpoints/. "
                             "Repeat for several seeds.")
    parser.add_argument("--checkpoint-name", default="mtppo_best.pt",
                        help="Checkpoint file inside <run-dir>/checkpoints (default: the "
                             "best-validation checkpoint each run saved itself).")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Timed solves per instance, after one discarded warm-up.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch thread count. Defaults to 1 to match the single-core "
                             "Pentium IV the published times were measured on.")
    parser.add_argument("--out", default=None,
                        help="Where to write the per-instance CSV "
                             "(default: results/inference_timing.csv).")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    published = load_published_times(BENCHMARK_TIMES)

    rows: List[Dict[str, Any]] = []
    for run in args.run_dir:
        run_dir = Path(run).resolve()
        print(f"\n{run_dir.name} ({args.checkpoint_name}, {args.repeats} timed repeats)")
        rows.extend(benchmark_run(run_dir, args.repeats, args.device, args.checkpoint_name))

    for row in rows:
        row["archetti_s"] = published.get((row["cost_class"], row["H"], row["n"]))
        row["speedup"] = (
            row["archetti_s"] / row["median_s"]
            if row["archetti_s"] else None
        )

    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / "inference_timing.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}")

    print_summary(rows)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    """Per-(cost class, H, n) table of our time against the published time."""
    print("\nMedian solve time vs Archetti et al. (2007) VMIR-OU branch-and-cut")
    print("(our time: one greedy forward pass, single thread, suboptimal solution;")
    print(" their time: proven optimality on a 2.8 GHz Pentium IV, mean over abs1-abs5)\n")
    header = f"{'class':<9} {'H':>2} {'n':>3} {'ours (ms)':>11} {'B&C (s)':>9} {'speedup':>10}"
    print(header)
    print("-" * len(header))

    keys = sorted({(r["cost_class"], r["H"], r["n"]) for r in rows},
                  key=lambda k: (k[1], k[0], k[2]))
    for key in keys:
        matching = [r for r in rows if (r["cost_class"], r["H"], r["n"]) == key]
        ours_ms = statistics.fmean(r["median_s"] for r in matching) * 1000
        theirs = matching[0]["archetti_s"]
        if theirs is None:
            speed = "n/a"
        elif theirs == 0.0:
            speed = "-"            # published as 0.0s; no meaningful ratio
        else:
            speed = f"{theirs / (ours_ms / 1000):,.0f}x"
        theirs_str = "n/a" if theirs is None else f"{theirs:,.1f}"
        print(f"{key[0]:<9} {key[1]:>2} {key[2]:>3} {ours_ms:>11.1f} {theirs_str:>9} {speed:>10}")


if __name__ == "__main__":
    main()
