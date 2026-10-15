"""
Fixed-fraction baseline: replace the inventory actor with a fixed rule and ask
whether any rule beats the trained agent.

The agent converges to near-daily just-in-time replenishment, which is the main
remaining gap to Archetti et al. (2007)'s optima. Two explanations fit that
observation and they call for opposite responses:

  - the max-level strategy IS expressible through the action mapping, and the
    agent simply fails to find it -> a learning problem (credit assignment,
    exploration);
  - the mapping cannot express it at all -> no amount of training helps, and
    the environment or the transform is what needs changing.

A fixed rule needs no training, so it separates the two. For each period and
retailer, the minimum delivery that avoids a stockout is

    m_i = min(max(0, d_i - I_i), k_i - I_i)

exactly as `MTPPO._delivery_from_fraction` computes it. A retailer with m_i = 0
gets action 0 and is not visited; one with m_i > 0 gets a fixed action f. The
actions then pass through the unmodified transform (dead zone 0.1) and the
environment's capacity scaling, so only the inventory strategy differs between
conditions.

Setting f for *every* retailer would be wrong: any action above the dead zone
delivers something, so that visits everyone every period and tests nothing.

Routing is held constant across all conditions -- every condition's stops are
sequenced by the same 2-opt routine, and the agent is scored on
`total_cost_best_route`, which is its inventory decisions with 2-opt routing.
Any cost difference is therefore attributable to inventory alone.

Nothing in the training code, the environment or any existing run folder is
touched; this reads saved checkpoints and steps the environment.

Usage:
    python src/diagnostics/fixed_fraction.py
    python src/diagnostics/fixed_fraction.py --trace Instances_lowcost_H6/abs4n10
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_THIS, ".."), os.path.join(_THIS, "..", "agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from environment.irp_env import IRPEnv
from training.mtppo import MTPPO
from training.routing_diagnostics import distance_matrix, tour_length, two_opt_tour

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "experiments" / "fixed_fraction"

# Archetti et al. (2007) Table IX, "Total cost" column: percent variation of the
# VMIR-ML optimum relative to VMIR-OU. Class averages over the per-size rows,
# which is the method the project report uses. (The per-size figures vary widely
# within a class -- low-cost H=3 runs -9.89% to -15.94% -- so the class average
# is a lossy summary; `--per-size-adjustment` uses the finer factors instead.)
TABLE_IX_PER_SIZE = {
    ("lowcost", 3): {5: -13.23, 10: -15.94, 15: -13.68, 20: -14.73, 25: -14.60,
                     30: -10.97, 35: -10.89, 40: -11.21, 45: -9.89, 50: -11.04},
    ("highcost", 3): {5: -8.80, 10: -8.55, 15: -6.92, 20: -6.40, 25: -5.96,
                      30: -4.98, 35: -4.46, 40: -4.72, 45: -4.12, 50: -4.36},
    ("lowcost", 6): {5: -26.06, 10: -22.18, 15: -21.64, 20: -21.78, 25: -22.73, 30: -21.27},
    ("highcost", 6): {5: -15.69, 10: -12.53, 15: -10.76, 20: -10.66, 25: -10.61, 30: -9.03},
}
TABLE_IX_CLASS = {k: float(np.mean(list(v.values()))) for k, v in TABLE_IX_PER_SIZE.items()}

FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
AGENT_CHECKPOINTS = [
    ("seed0", "src/results/seed0_20261007_130952", "mtppo_epoch325.pt"),
    ("seed1", "src/results/seed1_20261007_130952", "mtppo_epoch450.pt"),
    ("seed2", "src/results/seed2_20261007_130952", "mtppo_epoch450.pt"),
]


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
                if re.match(r"abs\d+n\d+\.dat$", name):
                    out[f"Instances_{cls}_H{H}/{name[:-4]}"] = float(sheet.cell_value(r, 2))
    return out


def describe(path: str) -> Dict[str, Any]:
    parent, stem = Path(path).parent.name, Path(path).stem
    folder = re.match(r"Instances_(lowcost|highcost)_H(\d+)", parent)
    size = re.match(r"abs\d+n(\d+)", stem)
    return {"key": f"{parent}/{stem}", "cost_class": folder.group(1),
            "H": int(folder.group(2)), "n": int(size.group(1))}


def ml_optimum(z_ou: float, meta: Dict[str, Any], per_size: bool) -> float:
    k = (meta["cost_class"], meta["H"])
    delta = TABLE_IX_PER_SIZE[k][meta["n"]] if per_size else TABLE_IX_CLASS[k]
    return z_ou * (1.0 + delta / 100.0)


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


def close_period(env: IRPEnv) -> Optional[Any]:
    """
    Drives the environment's routing loop to the end of the period so the clock
    advances, taking stops in arbitrary order and discarding the distance it
    charges. Safe because deliveries are capped at one vehicle load, so every
    selected stop is servable whatever the order, and because the order affects
    only distance -- quantities were already committed by
    `inventory_action_step`. Routing cost is computed separately by 2-opt over
    the same stops, which is what makes the conditions comparable.
    """
    guard = 0
    while True:
        eligible = np.flatnonzero(env.visited_mask == 0)
        _, _, critic_obs, terminated, _, _ = env.routing_action_step(int(eligible[0]))
        if critic_obs is not None:
            return critic_obs, terminated
        guard += 1
        assert guard < 10_000, "routing loop did not terminate"


def two_opt_period_cost(env: IRPEnv, served: Sequence[int], dist: np.ndarray) -> float:
    """Closed 2-opt tour over this period's stops; 0 when nobody is served."""
    if len(served) == 0:
        return 0.0
    return tour_length(dist, two_opt_tour(dist, list(served)))


def run_fixed(env: IRPEnv, fraction: float, dead_zone: float,
              trace: bool = False, upper_dead_zone: Optional[float] = None) -> Dict[str, Any]:
    """One episode under the fixed-fraction rule."""
    _, critic_obs, _ = env.reset()
    dist = distance_matrix(np.vstack([env.depot_location, env.location]).astype(float))

    inv_cost = float(env.initial_holding_cost)
    distance = 0.0
    visits: List[int] = []
    stockouts = 0
    terminated = False
    period = 0

    while not terminated:
        inventory = env.retailers_current_inventory.copy()
        demand = env.current_demand.copy()
        headroom = np.maximum(env.retailer_max_capacity - inventory, 0.0)
        minimum = np.clip(demand - inventory, 0.0, headroom)

        action = np.where(minimum > 1e-9, fraction, 0.0)
        delivery = MTPPO._delivery_from_fraction(
            action, inventory, demand, env.retailer_max_capacity, dead_zone, upper_dead_zone
        )
        _, r_inv, info = env.inventory_action_step(delivery)
        inv_cost += -r_inv
        stockouts += int(info["stockout_count"])

        q = env.replenishment_amount.copy()
        served = [i + 1 for i in np.flatnonzero(q > 1e-6)]
        visits.append(len(served))
        leg = two_opt_period_cost(env, served, dist)
        distance += leg

        if trace:
            order = two_opt_tour(dist, served) if served else []
            print(f"  period {period + 1}: {len(served)}/{env.num_retailers} visited, "
                  f"2-opt tour {leg:.0f}")
            print(f"    route: 0 -> {' -> '.join(str(x) for x in order) or '(none)'} -> 0")
            for i in range(env.num_retailers):
                mark = "*" if q[i] > 1e-6 else " "
                print(f"    {mark} r{i+1:<3} I={inventory[i]:8.2f}  d={demand[i]:7.2f}  "
                      f"k={env.retailer_max_capacity[i]:8.2f}  m={minimum[i]:7.2f}  "
                      f"a={action[i]:.2f}  q={q[i]:8.2f}")

        critic_obs, terminated = close_period(env)
        period += 1

    routing_cost = distance * env.delivery_cost
    return {"inv_cost": inv_cost, "routing_cost": routing_cost,
            "total_cost": inv_cost + routing_cost, "visits": visits,
            "stockout_count": stockouts, "num_retailers": env.num_retailers}


def run_agent(agent: MTPPO, env: IRPEnv) -> Dict[str, Any]:
    """
    One greedy episode, scored with 2-opt routing so it is comparable with the
    fixed rules. `total_cost_best_route` is the agent's inventory decisions with
    its tours re-ordered by 2-opt, which is exactly the same accounting.
    """
    route_log: List[Dict[str, Any]] = []
    m = agent.evaluate_episode(env, route_log=route_log)
    return {
        "inv_cost": m["inv_cost"],
        "routing_cost": m["total_cost_best_route"] - m["inv_cost"],
        "total_cost": m["total_cost_best_route"],
        "visits": [int(r["visits"]) for r in route_log],
        "stockout_count": int(m["stockout_count"]),
        "num_retailers": env.num_retailers,
    }


def build_agent(config: Dict[str, Any], checkpoint: Path) -> MTPPO:
    loc_dim, lookback = config["loc_dim"], config["lookback_window"]
    agent = MTPPO(
        node_feature_dims={"critic": loc_dim + 6, "inventory": loc_dim + 3,
                           "routing": 2 * loc_dim + 2},
        history_dim=1 + 2 * lookback, global_feature_dim=2,
        gin_dims=config["gin_dims"], mlp_dims=config["mlp_dims"],
        embed_dim=loc_dim + 2 * lookback, loc_dim=loc_dim,
        gamma=config["gamma"], skip_dead_zone=config["skip_dead_zone"], device="cpu",
    )
    agent.load(str(checkpoint))
    return agent


def read_manifest(manifest: Path, data_root: Path) -> List[str]:
    return [str((data_root / l.strip()).resolve())
            for l in manifest.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def aggregate(per_instance: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mean_total_cost": float(np.mean([r["total_cost"] for r in per_instance])),
        "mean_inv_cost": float(np.mean([r["inv_cost"] for r in per_instance])),
        "mean_routing_cost": float(np.mean([r["routing_cost"] for r in per_instance])),
        "mean_gap_ml_pct": float(np.mean([r["gap_ml_pct"] for r in per_instance])),
        "mean_gap_ou_pct": float(np.mean([r["gap_ou_pct"] for r in per_instance])),
        "total_stockouts": int(sum(r["stockout_count"] for r in per_instance)),
    }
    by_class = defaultdict(list)
    for r in per_instance:
        by_class[(r["cost_class"], r["H"])].append(r["gap_ml_pct"])
    for (cls, H), vals in by_class.items():
        out[f"gap_ml_{cls}_H{H}"] = float(np.mean(vals))
    per_period = {3: defaultdict(list), 6: defaultdict(list)}
    for r in per_instance:
        for p, v in enumerate(r["visits"], start=1):
            per_period[r["H"]][p].append(100.0 * v / r["n"])
    for H in (3, 6):
        for p in sorted(per_period[H]):
            out[f"visit_pct_H{H}_p{p}"] = float(np.mean(per_period[H][p]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default=None,
                    help="Instance key to print a per-period hand check for, "
                         "e.g. Instances_lowcost_H6/abs4n10")
    ap.add_argument("--per-size-adjustment", action="store_true",
                    help="Use Table IX per-size factors instead of class averages.")
    ap.add_argument("--fractions", default=None,
                    help="Comma-separated f values to test, overriding the default sweep. "
                         "Useful for probing near an endpoint, e.g. '0.75,0.999999,1.0'.")
    ap.add_argument("--skip-agents", action="store_true",
                    help="Fixed rules only; skips loading the trained checkpoints.")
    ap.add_argument("--upper-dead-zone", type=float, default=None,
                    help="Apply an upper dead zone to the delivery mapping.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    torch.set_num_threads(1)
    optima = load_optima()
    config = json.loads((ROOT / "src/results/seed0_20261007_130952/config.json").read_text())
    data_root = Path(config["data_root"])
    dead_zone = config["skip_dead_zone"]

    if args.trace:
        path = str((data_root / f"{args.trace}.dat").resolve())
        for f in (0.0, 1.0):
            print(f"\n{'='*70}\nHAND CHECK  {args.trace}  f = {f}\n{'='*70}")
            res = run_fixed(make_env(path, config), f, dead_zone, trace=True)
            print(f"  TOTAL: holding {res['inv_cost']:.1f}  routing {res['routing_cost']:.1f}  "
                  f"total {res['total_cost']:.1f}  stockouts {res['stockout_count']}")
        return

    fractions = ([float(x) for x in args.fractions.split(",")]
                 if args.fractions else FRACTIONS)

    agents = []
    for name, run, ckpt in ([] if args.skip_agents else AGENT_CHECKPOINTS):
        cfg = json.loads((ROOT / run / "config.json").read_text())
        agents.append((f"agent {name}", build_agent(cfg, ROOT / run / "checkpoints" / ckpt), cfg))

    rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    for split, manifest in (("val", config["val_manifest"]), ("test", config["eval_manifest"])):
        paths = read_manifest(ROOT / manifest, data_root)
        print(f"\n=== {split} ({len(paths)} instances) ===")
        conditions: List[Any] = [(f"f={f:g}", f, None) for f in fractions]
        conditions += [(name, None, agent) for name, agent, _ in agents]

        for label, f, agent in conditions:
            per_instance = []
            for path in paths:
                meta = describe(path)
                env = make_env(path, config)
                res = (run_fixed(env, f, dead_zone, upper_dead_zone=args.upper_dead_zone)
                       if agent is None else run_agent(agent, env))
                z_ou = optima[meta["key"]]
                z_ml = ml_optimum(z_ou, meta, args.per_size_adjustment)
                rec = {**meta, "condition": label, "split": split, **res,
                       "z_ou": z_ou, "z_ml_est": z_ml,
                       "gap_ou_pct": 100.0 * (res["total_cost"] - z_ou) / z_ou,
                       "gap_ml_pct": 100.0 * (res["total_cost"] - z_ml) / z_ml}
                rec["visits"] = " ".join(str(v) for v in res["visits"])
                per_instance.append({**rec, "visits": res["visits"]})
                rows.append(rec)
            agg = aggregate(per_instance)
            summary.append({"condition": label, "split": split, **agg})
            flag = "  !! STOCKOUTS" if agg["total_stockouts"] else ""
            print(f"  {label:<12} total {agg['mean_total_cost']:9.0f}  "
                  f"hold {agg['mean_inv_cost']:7.0f}  rout {agg['mean_routing_cost']:7.0f}  "
                  f"gap_ML {agg['mean_gap_ml_pct']:6.1f}%{flag}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, data in (("per_instance.csv", rows), ("summary.csv", summary)):
        fields: List[str] = []
        for r in data:
            for k in r:
                if k not in fields: fields.append(k)
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(data)
        print(f"wrote {out / name} ({len(data)} rows)")


if __name__ == "__main__":
    main()
