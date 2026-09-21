import argparse
import glob
import os
import sys
from pathlib import Path

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


def build_envs(args: argparse.Namespace) -> list:
    """Resolves --instance/--data-glob into a list of IRPEnv instances."""
    if args.instance:
        paths = [args.instance]
    else:
        paths = sorted(glob.glob(args.data_glob))
        if not paths:
            raise FileNotFoundError(f"No instance files matched: {args.data_glob}")

    envs = [
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
    print(f"Loaded {len(envs)} instance(s): {[Path(p).name for p in paths]}")
    return envs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MTPPO on the IRP-VMI benchmark instances.")

    default_instance = PROJECT_ROOT / "data" / "Instances_lowcost_H6" / "abs1n5.dat"
    parser.add_argument("--instance", type=str, default=None, help="Path to a single benchmark instance file.")
    parser.add_argument(
        "--data-glob",
        type=str,
        default=str(default_instance),
        help="Glob pattern for one or more instance files to sample from each epoch "
        "(ignored if --instance is set).",
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
    parser.add_argument("--episodes-per-epoch", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=50)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    envs = build_envs(args)

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

    checkpoint_dir = None
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(epoch, episode_stats, losses):
        if checkpoint_dir and epoch % args.checkpoint_every == 0:
            path = checkpoint_dir / f"mtppo_epoch{epoch}.pt"
            mtppo.save(str(path))
            print(f"  saved checkpoint -> {path}")

    mtppo.train(
        envs=envs,
        num_epochs=args.num_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        log_every=args.log_every,
        on_epoch_end=on_epoch_end,
    )

    if checkpoint_dir:
        final_path = checkpoint_dir / "mtppo_final.pt"
        mtppo.save(str(final_path))
        print(f"Training complete. Final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
