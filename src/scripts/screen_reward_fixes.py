"""
Screening analysis for the two inventory-reward fixes (joint reward
normalisation, with and without gamma = 1.0) against the unmodified seed-0 run.

Reports, per run and per 25-epoch checkpoint, the quantities that would show a
shift away from just-in-time replenishment: how many retailers are visited in
each period, how full each delivery is, and how the holding/routing split moves.

`MTPPO.evaluate_episode` returns episode-level means only, so per-period visit
rates and fill fractions are produced here by stepping the episode with the
agent's own methods. That duplication is a drift risk -- a copy of the greedy
loop can diverge from the real one and nobody notices -- so `--verify` checks
the replication's cost figures against `evaluate_episode` on the same
checkpoint and instance, and the run aborts if they disagree.

Nothing in the training or evaluation path is modified or re-run; this reads
saved checkpoints only.

Usage:
    python src/scripts/screen_reward_fixes.py \
        --out reports/experiments/<name>/screen.csv
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_THIS, ".."), os.path.join(_THIS, "..", "agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.features import build_inventory_features, build_inventory_history
from environment.irp_env import IRPEnv
from training.mtppo import MTPPO

ROOT = Path(__file__).resolve().parents[2]

# z_ML_est = z_OU * (1 + delta/100), where delta is Archetti et al. (2007)'s
# Table IX "Total cost" column: the percent variation of the VMIR-ML optimum
# relative to the VMIR-OU optimum, read from the paper per (cost class, H, n).
#
# The report's Section 4.5 used class averages. Per-size factors are used here
# instead because the spread within a class is wide -- low-cost H=3 runs from
# -9.89% at n=45 to -15.94% at n=10 -- so a class average discards real
# structure. Each figure is still an average over the five replicates abs1-abs5
# of its row, so applying it to abs4 or abs5 alone remains an approximation.
TABLE_IX = {
    ("lowcost", 3): {5: -13.23, 10: -15.94, 15: -13.68, 20: -14.73, 25: -14.60,
                     30: -10.97, 35: -10.89, 40: -11.21, 45: -9.89, 50: -11.04},
    ("highcost", 3): {5: -8.80, 10: -8.55, 15: -6.92, 20: -6.40, 25: -5.96,
                      30: -4.98, 35: -4.46, 40: -4.72, 45: -4.12, 50: -4.36},
    ("lowcost", 6): {5: -26.06, 10: -22.18, 15: -21.64, 20: -21.78, 25: -22.73, 30: -21.27},
    ("highcost", 6): {5: -15.69, 10: -12.53, 15: -10.76, 20: -10.66, 25: -10.61, 30: -9.03},
}


def ml_optimum(z_ou: float, cost_class: str, H: int, n: int) -> float:
    """The estimated VMIR-ML optimum for one instance, from its OU optimum."""
    return z_ou * (1.0 + TABLE_IX[(cost_class, H)][n] / 100.0)


def load_optima() -> Dict[str, float]:
    """Per-instance z*_OU from Archetti et al.'s own result workbooks."""
    import xlrd

    out: Dict[str, float] = {}
    for cls in ("lowcost", "highcost"):
        for H in (3, 6):
            sheet = xlrd.open_workbook(
                str(ROOT / "data" / "benchmarks" / f"Results_{cls}_H{H}.xls")
            ).sheet_by_index(0)
            for r in range(1, sheet.nrows):
                name = str(sheet.cell_value(r, 0)).strip()
                if not re.match(r"abs\d+n\d+\.dat$", name):
                    continue
                out[f"Instances_{cls}_H{H}/{name[:-4]}"] = float(sheet.cell_value(r, 2))
    return out


def build_agent(config: Dict[str, Any], checkpoint: Path) -> MTPPO:
    """Rebuilds the agent from a run's own logged config, as train.py built it."""
    loc_dim, lookback = config["loc_dim"], config["lookback_window"]
    uses_position = config.get("routing_uses_position", True)
    # A discrete-choice run's decoder has one output per choice, so the actor
    # must be rebuilt with the same action set or the state dict will not load.
    raw_choices = config.get("inventory_choices")
    if isinstance(raw_choices, str):
        raw_choices = [float(x) for x in raw_choices.split(",")]
    agent = MTPPO(
        node_feature_dims={
            "critic": loc_dim + 6,
            "inventory": loc_dim + 3,
            "routing": (2 * loc_dim + 2) if uses_position else (loc_dim + 1),
        },
        history_dim=1 + 2 * lookback,
        global_feature_dim=2,
        gin_dims=config["gin_dims"],
        mlp_dims=config["mlp_dims"],
        embed_dim=loc_dim + 2 * lookback,
        loc_dim=loc_dim,
        gamma=config["gamma"],
        skip_dead_zone=config["skip_dead_zone"],
        upper_dead_zone=config.get("upper_dead_zone"),
        inventory_choices=raw_choices,
        routing_uses_position=uses_position,
        device="cpu",
    )
    agent.load(str(checkpoint))
    return agent


def make_env(path: str, config: Dict[str, Any]) -> IRPEnv:
    return IRPEnv(
        data_file_path=path,
        loc_dim=config["loc_dim"],
        lookback_window=config["lookback_window"],
        product_price=config["product_price"],
        penalty_factor=config["penalty_factor"],
        delivery_cost=config["delivery_cost"],
        charge_initial_holding=config.get("charge_initial_holding", True),
    )


def rollout(agent: MTPPO, env: IRPEnv) -> Dict[str, Any]:
    """
    One greedy episode, recording per-period visit counts and fill fractions
    alongside the cost breakdown. Mirrors `MTPPO.evaluate_episode`'s decisions
    exactly -- same mean Beta action, same delivery transform, same argmax
    routing -- and `--verify` holds it to that.

    `fill` is q / (k - I) over the retailers actually served that period, i.e.
    how much of the available headroom each delivery took. A policy making
    larger, less frequent deliveries shows a higher fill and fewer visits.
    """
    agent.critic.eval(); agent.inv_actor.eval(); agent.routing_actor.eval()
    scale = agent._location_scale(env)
    retailer_scale, _ = agent._quantity_scales(env)
    _, critic_obs, _ = env.reset()

    inv_cost = float(env.initial_holding_cost)
    distance = 0.0
    visits: List[int] = []
    fills: List[float] = []
    terminated = False

    with torch.no_grad():
        while not terminated:
            inv_obs = agent._inventory_obs_from_critic(critic_obs)
            node_feats = agent._normalize_location(
                build_inventory_features(inv_obs, retailer_scale), scale
            ).to(agent.device)
            hist = build_inventory_history(inv_obs, retailer_scale).to(agent.device)
            fraction = agent.inv_actor.mean_action(node_feats, hist)
            headroom = np.maximum(env.retailer_max_capacity - env.retailers_current_inventory, 0.0)
            delivery = agent._delivery_from_fraction(
                fraction.cpu().numpy(),
                env.retailers_current_inventory,
                env.current_demand,
                env.retailer_max_capacity,
                agent.skip_dead_zone,
            )
            routing_obs, r_inv, _ = env.inventory_action_step(delivery)
            inv_cost += -r_inv

            served = env.replenishment_amount > 1e-6
            visits.append(int(served.sum()))
            usable = served & (headroom > 1e-9)
            fills.append(
                float(np.mean(env.replenishment_amount[usable] / headroom[usable]))
                if usable.any() else float("nan")
            )

            guard = 0
            while True:
                from agent.features import build_routing_features
                feats = agent._normalize_location(
                    build_routing_features(routing_obs, retailer_scale, agent.routing_uses_position),
                    scale, num_coord_cols=agent._routing_coord_cols,
                ).to(agent.device)
                mask = torch.from_numpy(routing_obs["visited_mask"]).to(agent.device)
                logits = agent.routing_actor(feats).masked_fill(mask == 1, float("-inf"))
                action = int(torch.argmax(logits).item())
                routing_obs, r_vrp, next_critic, terminated, _, _ = env.routing_action_step(action)
                distance += -r_vrp / max(env.delivery_cost, 1e-12)
                if next_critic is not None:
                    critic_obs = next_critic
                    break
                guard += 1
                assert guard < 10_000, "routing loop did not terminate"

    agent.critic.train(); agent.inv_actor.train(); agent.routing_actor.train()
    routing_cost = distance * env.delivery_cost
    return {
        "inv_cost": inv_cost, "routing_cost": routing_cost,
        "total_cost": inv_cost + routing_cost,
        "visits": visits, "fills": fills, "num_retailers": env.num_retailers,
    }


def describe(path: str) -> Dict[str, Any]:
    parent, stem = Path(path).parent.name, Path(path).stem
    folder = re.match(r"Instances_(lowcost|highcost)_H(\d+)", parent)
    return {"key": f"{parent}/{stem}", "cost_class": folder.group(1), "H": int(folder.group(2))}


def analyse(agent: MTPPO, paths: List[str], config: Dict[str, Any],
            optima: Dict[str, float]) -> Dict[str, Any]:
    """Aggregates one checkpoint over one instance pool."""
    per_period_visits: Dict[int, Dict[int, List[float]]] = {3: defaultdict(list), 6: defaultdict(list)}
    per_period_fill: Dict[int, Dict[int, List[float]]] = {3: defaultdict(list), 6: defaultdict(list)}
    inv, rout, tot = [], [], []
    gap_ml, gap_ou = [], []
    by_class: Dict[tuple, List[float]] = defaultdict(list)

    for path in paths:
        meta = describe(path)
        env = make_env(path, config)
        res = rollout(agent, env)
        inv.append(res["inv_cost"]); rout.append(res["routing_cost"]); tot.append(res["total_cost"])
        for p, (v, f) in enumerate(zip(res["visits"], res["fills"]), start=1):
            per_period_visits[meta["H"]][p].append(100.0 * v / res["num_retailers"])
            if not np.isnan(f):
                per_period_fill[meta["H"]][p].append(f)
        z_ou = optima.get(meta["key"])
        if z_ou:
            z_ml = ml_optimum(z_ou, meta["cost_class"], meta["H"],
                              int(meta["key"].split("n")[-1]))
            gap_ou.append(100.0 * (res["total_cost"] - z_ou) / z_ou)
            g = 100.0 * (res["total_cost"] - z_ml) / z_ml
            gap_ml.append(g)
            by_class[(meta["cost_class"], meta["H"])].append(g)

    out: Dict[str, Any] = {
        "mean_inv_cost": float(np.mean(inv)),
        "mean_routing_cost": float(np.mean(rout)),
        "mean_total_cost": float(np.mean(tot)),
        "mean_gap_ml_pct": float(np.mean(gap_ml)) if gap_ml else float("nan"),
        "mean_gap_ou_pct": float(np.mean(gap_ou)) if gap_ou else float("nan"),
    }
    for H in (3, 6):
        for p in sorted(per_period_visits[H]):
            out[f"visit_pct_H{H}_p{p}"] = float(np.mean(per_period_visits[H][p]))
            vals = per_period_fill[H][p]
            out[f"fill_H{H}_p{p}"] = float(np.mean(vals)) if vals else float("nan")
    for (cls, H), vals in by_class.items():
        out[f"gap_ml_{cls}_H{H}"] = float(np.mean(vals))
    return out


def training_metrics(run_dir: Path, epoch: int) -> Dict[str, float]:
    """Critic loss / entropy at this epoch, straight from the run's metrics.csv."""
    for row in csv.DictReader(open(run_dir / "metrics.csv")):
        if int(row["epoch"]) == epoch:
            return {
                "value_loss": float(row["value"]),
                "entropy_mean": float(row["entropy"]),
                "value_explained_var": float(row["value_explained_var"]),
                "advantage_bias": float(row["advantage_bias"]),
                "logit_spread": float(row["mean_logit_spread"]) if row.get("mean_logit_spread") not in (None, "", "nan") else float("nan"),
            }
    return {}


def read_manifest(manifest: Path, data_root: Path) -> List[str]:
    return [str((data_root / l.strip()).resolve())
            for l in manifest.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--epochs", default="25-300")
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--out",
                    default=str(ROOT / "reports" / "experiments" / "reward_normalisation" / "screen.csv"))
    ap.add_argument("--verify", action="store_true", default=True)
    args = ap.parse_args()

    torch.set_num_threads(1)
    lo, hi = (int(x) for x in args.epochs.split("-"))
    epochs = list(range(lo, hi + 1, args.step))
    optima = load_optima()

    rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}

    for run in args.run_dir:
        run_dir = Path(run).resolve()
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        data_root = Path(config["data_root"])
        val = read_manifest(ROOT / config["val_manifest"], data_root)
        test = read_manifest(ROOT / config["eval_manifest"], data_root)
        print(f"\n{run_dir.name}  (gamma={config['gamma']}, "
              f"joint_norm={config.get('joint_reward_normalisation', False)})")

        for epoch in epochs:
            ckpt = run_dir / "checkpoints" / f"mtppo_epoch{epoch}.pt"
            if not ckpt.exists():
                print(f"  epoch {epoch}: no checkpoint, skipped"); continue
            agent = build_agent(config, ckpt)

            if args.verify:
                env = make_env(val[0], config)
                mine = rollout(agent, env)
                env2 = make_env(val[0], config)
                theirs = agent.evaluate_episode(env2, collect_diagnostics=False)
                for k in ("inv_cost", "routing_cost", "total_cost"):
                    if abs(mine[k] - theirs[k]) > 1e-6:
                        sys.exit(f"replication drift at {run_dir.name} ep{epoch}: "
                                 f"{k} {mine[k]} vs evaluate_episode {theirs[k]}")

            res = analyse(agent, val, config, optima)
            res.update(training_metrics(run_dir, epoch))
            res.update({"run": run_dir.name, "epoch": epoch, "split": "val",
                        "gamma": config["gamma"],
                        "joint_norm": config.get("joint_reward_normalisation", False)})
            rows.append(res)
            print(f"  ep{epoch:>3} val: gap_ML {res['mean_gap_ml_pct']:6.1f}%  "
                  f"total {res['mean_total_cost']:8.0f}  "
                  f"inv {res['mean_inv_cost']:7.0f}  rout {res['mean_routing_cost']:7.0f}")

            if run_dir.name not in best or res["mean_gap_ml_pct"] < best[run_dir.name]["gap"]:
                best[run_dir.name] = {"gap": res["mean_gap_ml_pct"], "epoch": epoch,
                                      "config": config, "test": test, "run_dir": run_dir}

    print("\n=== best-validation-gap checkpoint, evaluated on the held-out test set ===")
    for name, info in best.items():
        agent = build_agent(info["config"], info["run_dir"] / "checkpoints" / f"mtppo_epoch{info['epoch']}.pt")
        res = analyse(agent, info["test"], info["config"], optima)
        res.update(training_metrics(info["run_dir"], info["epoch"]))
        res.update({"run": name, "epoch": info["epoch"], "split": "test",
                    "gamma": info["config"]["gamma"],
                    "joint_norm": info["config"].get("joint_reward_normalisation", False)})
        rows.append(res)
        print(f"  {name}  ep{info['epoch']}: test gap_ML {res['mean_gap_ml_pct']:.1f}%  "
              f"total {res['mean_total_cost']:.0f}")

    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    lead = ["run", "epoch", "split", "gamma", "joint_norm"]
    fields = lead + [f for f in fields if f not in lead]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
