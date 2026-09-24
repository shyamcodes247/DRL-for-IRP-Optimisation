import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..")
_AGENT_DIR = os.path.join(_SRC_DIR, "agent")
for _p in (_SRC_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from environment.irp_env import IRPEnv
from training.mtppo import MTPPO
from utils.logger import ResultsLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _looks_like_instance_file(path: str) -> bool:
    """
    A real instance file's first line is `<num_nodes> <episode_length>
    <vehicle_capacity>`. Manifest files living in the same data folders
    (e.g. `comp_format_instances.dat`, `test_instances.dat`) just list
    instance filenames one per line, and fail this check — `convert_instance`
    would otherwise crash trying to parse a filename as three numbers.
    """
    try:
        with open(path) as f:
            first_line = f.readline().split()
        if len(first_line) != 3:
            return False
        int(first_line[0])
        int(first_line[1])
        float(first_line[2])
        return True
    except (OSError, ValueError):
        return False


def _read_manifest(manifest_path: str, data_root: str) -> List[str]:
    """
    Reads a manifest file (a plain list of instance paths relative to
    `data_root`, one per line — e.g. `data/splits/train_by_replicate.txt`)
    and resolves each to a full path.
    """
    with open(manifest_path) as f:
        names = [line.strip() for line in f if line.strip()]
    return [str(Path(data_root) / name) for name in names]


def _filter_instance_files(candidates: List[str]) -> List[str]:
    """Drops manifest-style files (e.g. comp_format_instances.dat) from a candidate list."""
    paths = [p for p in candidates if _looks_like_instance_file(p)]
    skipped = [p for p in candidates if p not in paths]
    if skipped:
        print(f"Skipping {len(skipped)} non-instance file(s): {[Path(p).name for p in skipped]}")
    return paths


def resolve_train_paths(args: argparse.Namespace) -> List[str]:
    """
    Resolves the training instance pool (the m instances Algorithm 1 samples
    each epoch) from, in priority order: --instance (a single file),
    --train-manifest (an explicit list, e.g. for a train/eval split), or
    --data-glob (every matching file, the default).
    """
    if args.instance:
        return [args.instance]
    if args.train_manifest:
        candidates = _read_manifest(args.train_manifest, args.data_root)
    else:
        candidates = sorted(glob.glob(args.data_glob))
    if not candidates:
        raise FileNotFoundError(
            f"No instance files found (--train-manifest={args.train_manifest!r}, "
            f"--data-glob={args.data_glob!r})."
        )
    paths = _filter_instance_files(candidates)
    if not paths:
        raise FileNotFoundError("No valid instance files found among the resolved training candidates.")
    return paths


def resolve_eval_paths(args: argparse.Namespace, train_paths: List[str]) -> List[str]:
    """
    Resolves the held-out evaluation pool from --eval-manifest. Without one,
    falls back to evaluating on the training pool itself (the prior
    behavior) — returns `train_paths` unchanged (same object), so callers
    can skip rebuilding envs for it.
    """
    if not args.eval_manifest:
        return train_paths
    candidates = _read_manifest(args.eval_manifest, args.data_root)
    if not candidates:
        raise FileNotFoundError(f"No instance files found in eval manifest: {args.eval_manifest}")
    paths = _filter_instance_files(candidates)
    if not paths:
        raise FileNotFoundError(f"No valid instance files found in eval manifest: {args.eval_manifest}")
    return paths


def build_envs(paths: List[str], args: argparse.Namespace) -> list:
    """Builds one IRPEnv per path, all sharing the same environment hyperparameters."""
    return [
        IRPEnv(
            data_file_path=path,
            loc_dim=args.loc_dim,
            lookback_window=args.lookback_window,
            product_price=args.product_price,
            penalty_factor=args.penalty_factor,
            delivery_cost=args.delivery_cost,
        )
        for path in paths
    ]


def derive_run_name(paths: List[str]) -> str:
    """Names a run after its instance(s): the single stem, the shared parent folder
    name (+ count) when every instance comes from one subfolder, or `multi_<n>_<first_stem>`."""
    stems = [Path(p).stem for p in paths]
    if len(stems) == 1:
        return stems[0]
    parents = {Path(p).parent.name for p in paths}
    if len(parents) == 1:
        return f"{next(iter(parents))}_{len(stems)}instances"
    return f"multi_{len(stems)}_{stems[0]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MTPPO on the IRP-VMI benchmark instances.")

    default_data_dir = PROJECT_ROOT / "data" / "Instances_lowcost_H6"
    parser.add_argument("--instance", type=str, default=None, help="Path to a single benchmark instance file.")
    parser.add_argument(
        "--data-glob",
        type=str,
        default=str(default_data_dir / "*.dat"),
        help="Glob pattern for the m problem instances trained on: every matching file "
        "is rolled out once per epoch (Algorithm 1's \"sampling m instances\", realized "
        "here as the fixed pool of instance files touched every epoch). Manifest-style "
        "files that don't parse as an instance (e.g. comp_format_instances.dat) are "
        "skipped automatically. Ignored if --instance or --train-manifest is set.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Root directory that --train-manifest/--eval-manifest entries are resolved "
        "relative to.",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=None,
        help="Path to a manifest file (e.g. data/splits/train_by_replicate.txt) listing "
        "the m training instances, one path per line, relative to --data-root. Overrides "
        "--data-glob; ignored if --instance is set.",
    )
    parser.add_argument(
        "--eval-manifest",
        type=str,
        default=None,
        help="Path to a manifest file (e.g. data/splits/eval_by_replicate.txt) listing "
        "held-out instances, one path per line, relative to --data-root, evaluated "
        "greedily every --eval-every epochs but never trained on. Without this, "
        "evaluation runs on the training pool itself (no generalization signal).",
    )

    parser.add_argument("--loc-dim", type=int, default=2)
    parser.add_argument("--lookback-window", type=int, default=3)
    parser.add_argument("--product-price", type=float, default=20.0)
    parser.add_argument("--penalty-factor", type=float, default=0.3)
    parser.add_argument("--delivery-cost", type=float, default=1.0)

    parser.add_argument("--gin-dims", type=int, nargs="+", default=[64, 128, 128])
    parser.add_argument("--mlp-dims", type=int, nargs="+", default=[128, 128])

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "src" / "results"),
        help="Root directory under which each run gets its own timestamped subfolder "
        "(metrics.csv, config.json, checkpoints/).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Prefix for the run's results subfolder name (default: derived from the "
        "instance file(s)).",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=25,
        help="Run a greedy evaluation episode (paper-style Inv.cost/VRP.Dist/Fill-rate "
        "breakdown, logged to eval.csv) every this many epochs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    def _describe(paths: List[str]) -> str:
        names = [Path(p).name for p in paths]
        return str(names) if len(names) <= 10 else f"{len(names)} files, e.g. {names[:3]} ... {names[-1]}"

    train_paths = resolve_train_paths(args)
    train_envs = build_envs(train_paths, args)
    print(f"Loaded {len(train_envs)} training instance(s): {_describe(train_paths)}")

    eval_paths = resolve_eval_paths(args, train_paths)
    if eval_paths is train_paths:
        eval_envs = train_envs
        print("Evaluating on the training pool (pass --eval-manifest for a held-out generalization check).")
    else:
        eval_envs = build_envs(eval_paths, args)
        print(f"Loaded {len(eval_envs)} held-out evaluation instance(s): {_describe(eval_paths)}")

    run_name = args.run_name or derive_run_name(train_paths)

    logger = ResultsLogger(results_root=args.results_dir, run_name=run_name)
    logger.log_config(vars(args))
    print(f"Logging results to {logger.run_dir}")

    node_feature_dims = {
        "critic": args.loc_dim + 6,
        "inventory": args.loc_dim + 3,
        "routing": args.loc_dim + 1,
    }
    history_dim = 1 + 2 * args.lookback_window
    embed_dim = args.loc_dim + 2 * args.lookback_window

    mtppo = MTPPO(
        node_feature_dims=node_feature_dims,
        history_dim=history_dim,
        global_feature_dim=2,
        gin_dims=args.gin_dims,
        mlp_dims=args.mlp_dims,
        embed_dim=embed_dim,
        loc_dim=args.loc_dim,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )

    def run_evaluation(epoch):
        results = []
        for env, path in zip(eval_envs, eval_paths):
            metrics = mtppo.evaluate_episode(env)
            # `Path(path).parent.name` disambiguates instances that share a filename
            # across data subfolders (e.g. abs5n30.dat exists under both
            # Instances_highcost_H3 and Instances_lowcost_H3, with different cost
            # parameters) — logging the stem alone would make them indistinguishable
            # in eval.csv.
            instance_label = f"{Path(path).parent.name}/{Path(path).stem}"
            metrics = {"instance": instance_label, **metrics}
            logger.log_metrics("eval", epoch, metrics)
            results.append(metrics)

        if len(results) <= 6:
            for metrics in results:
                print(
                    f"  [eval @ epoch {epoch:5d}] {metrics['instance']:>32s}  "
                    f"inv_cost={metrics['inv_cost']:9.2f}  vrp_dist={metrics['vrp_distance']:8.2f}  "
                    f"fill_rate={metrics['fill_rate']:6.2f}%  total_cost={metrics['total_cost']:9.2f}"
                )
        else:
            # Too many instances to print one line each — full detail is still in
            # eval.csv; here just show the mean across all m instances.
            n = len(results)
            mean = lambda key: sum(m[key] for m in results) / n
            print(
                f"  [eval @ epoch {epoch:5d}] {n} instances  "
                f"mean inv_cost={mean('inv_cost'):9.2f}  mean vrp_dist={mean('vrp_distance'):8.2f}  "
                f"mean fill_rate={mean('fill_rate'):6.2f}%  mean total_cost={mean('total_cost'):9.2f}"
            )

    last_eval_epoch = None

    def on_epoch_end(epoch, episode_stats, losses):
        nonlocal last_eval_epoch
        mean_r_inv = sum(s["r_inv"] for s in episode_stats) / len(episode_stats)
        mean_r_vrp = sum(s["r_vrp"] for s in episode_stats) / len(episode_stats)
        mean_total = sum(s["total_reward"] for s in episode_stats) / len(episode_stats)
        logger.log_epoch(
            epoch,
            {
                "mean_r_inv": mean_r_inv,
                "mean_r_vrp": mean_r_vrp,
                "mean_total_reward": mean_total,
                **losses,
            },
        )
        if args.eval_every and epoch % args.eval_every == 0:
            run_evaluation(epoch)
            last_eval_epoch = epoch
        if epoch % args.checkpoint_every == 0:
            path = logger.checkpoint_path(f"mtppo_epoch{epoch}.pt")
            mtppo.save(path)
            print(f"  saved checkpoint -> {path}")

    try:
        mtppo.train(
            envs=train_envs,
            num_epochs=args.num_epochs,
            log_every=args.log_every,
            on_epoch_end=on_epoch_end,
        )
        # Skip if the periodic eval (above) already covered the final epoch —
        # e.g. num_epochs=300 with eval_every=25 triggers it there too, and
        # running it again would double-log every instance under the same
        # epoch in eval.csv.
        if last_eval_epoch != args.num_epochs:
            print("Final evaluation:")
            run_evaluation(args.num_epochs)
    finally:
        logger.close()

    final_path = logger.checkpoint_path("mtppo_final.pt")
    mtppo.save(final_path)
    print(f"Training complete. Final checkpoint -> {final_path}")
    print(f"Metrics -> {logger.run_dir / 'metrics.csv'}")
    print(f"Eval history -> {logger.run_dir / 'eval.csv'}")


if __name__ == "__main__":
    main()
