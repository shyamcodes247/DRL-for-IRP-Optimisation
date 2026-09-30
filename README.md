# DRL for IRP Optimisation

Deep reinforcement learning for the **Inventory Routing Problem under Vendor
Managed Inventory (IRP-VMI)** — a single supplier decides, at each discrete
time period, how much of a product to replenish at each retailer and what
route a single vehicle should take to deliver it, jointly minimising holding
cost and transportation cost over a planning horizon.

This project trains an **MTPPO** (Multi-Task Proximal Policy Optimization)
agent — a two-actor, one-critic architecture following **Lu et al. (2025)** —
on the classical IRP benchmark instances of **Archetti, Bertazzi, Laporte and
Speranza (2007)**, "A Branch-and-Cut Algorithm for a Vendor Managed Inventory
Routing Problem" (their exact branch-and-cut solutions provide a ground-truth
comparison point).

## Architecture

A **centralized-training, decentralized-execution (CTDE)** setup with three
networks, each built on a Graph Isomorphism Network (GIN) encoder over the
depot + retailer node set:

- **Inventory actor** (`src/agent/inventory_actor.py`) — decides a continuous
  replenishment quantity for every retailer in one shot, each period. Outputs
  a per-retailer Normal distribution (mean, std).
- **Routing actor** (`src/agent/routing_actor.py`) — a pointer-network-style
  policy that sequentially picks the next node for the vehicle to visit,
  masked to already-served nodes and to nodes the remaining vehicle load
  can't cover.
- **Critic** (`src/agent/critic.py`) — a single shared value function
  evaluating the joint (inventory + routing) state once per period, before
  either actor acts.

Both actors are trained with clipped PPO surrogate objectives against **one
shared advantage** (not two separate ones): each period's inventory and
routing rewards are z-scored separately and summed before computing a single
discounted advantage, so the inventory actor's gradient reflects the routing
cost consequences of its replenishment decisions too, not just its own
holding/stockout cost in isolation. See `src/training/mtppo.py` and
`src/training/rollout_buffer.py` for the full algorithm and its correspondence
to Lu et al.'s Algorithm 1.

## Environment

`src/environment/irp_env.py` implements the simulation (`IRPEnv`, a custom
Gymnasium-style environment — split into `inventory_action_step` and a
variable-length `routing_action_step` loop per period, since one period is
one inventory decision followed by a sequential tour, not a single `step()`).

It follows Archetti et al. (2007)'s VMI model, with one deliberate
relaxation and one configurable trade-off:

- **One vehicle, one route per period** — total deliveries in a period are
  capped at the vehicle's capacity; there's no mid-tour depot reload.
- **Continuous replenishment** rather than the paper's strict
  order-up-to-level policy — any amount up to a retailer's remaining
  headroom, matching Lu et al.'s architecture (this corresponds to the
  paper's more flexible **VMIR** relaxation rather than its primary
  **VMIR-OU** model).
- **Sales-loss costing**, controlled by `--sales-loss-cost {auto,soft,none}`:
  - `none` matches Archetti's model exactly — stockouts are a hard
    feasibility constraint (`IRPEnv` auto-tops-up delivery to avoid them,
    subject to real depot-stock/vehicle-capacity limits), never priced.
  - `soft` (the default once a price is given) prices lost sales instead,
    which is what lets the sales-loss term appear in the reward gradient at
    all — useful for training runs where a hard constraint would remove any
    learning signal around stockout risk.

## Project layout

```
src/
  agent/           GIN encoder, inventory actor, routing actor, critic, MLP head
  environment/     IRPEnv (the simulation) and the benchmark-instance parser
  training/        MTPPO algorithm, rollout buffer, train.py CLI entry point
  utils/           ResultsLogger (per-run metrics/config/checkpoints)
  tests/           pytest suite (environment invariants, network smoke tests, ...)
  results/         one timestamped subfolder per training run (gitignored data)
data/
  Instances_{low,high}cost_H{3,6}/   Archetti et al. (2007)'s benchmark instances
  splits/                            train/eval manifests used by --train-manifest
```

## Setup

Requires Python 3.10+ (developed and tested on 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas gymnasium torch pytest
```

## Usage

Train with defaults (every instance in `data/Instances_lowcost_H6/`, 200
epochs, CPU):

```bash
cd src
python training/train.py
```

Train on a different instance set, with more epochs:

```bash
python training/train.py --data-glob "../data/Instances_lowcost_H3/*.dat" --num-epochs 500
```

Train/evaluate on fixed manifests instead of a glob (for a proper held-out
generalisation split):

```bash
python training/train.py \
  --train-manifest ../data/splits/train_by_replicate.txt \
  --eval-manifest ../data/splits/eval_by_replicate.txt \
  --data-root ../data
```

Run `python training/train.py --help` for the full option list — network
sizes (`--gin-dims`, `--mlp-dims`), PPO hyperparameters (`--lr`, `--gamma`,
`--clip-eps`, `--entropy-coef`, `--ppo-epochs`), sales-loss costing
(`--sales-loss-cost`, `--product-price`, `--penalty-factor`), and
run/checkpoint/eval-logging options.

Each run writes to its own timestamped folder under `src/results/`:
`config.json` (the resolved CLI args), `metrics.csv` (per-epoch training
stats), `eval.csv` (periodic greedy-policy cost breakdown — inventory cost,
delivery distance, fill rate — comparable to the paper's reported metrics),
and `checkpoints/`.

## Testing

```bash
cd src
python -m pytest -q
```

Covers benchmark-instance parsing, `IRPEnv`'s feasibility invariants
(inventory bounds, vehicle capacity, routing deadlock-freedom, the
hard-feasibility no-stockout guarantee under `--sales-loss-cost none`), and
an MTPPO training smoke test (a few real epochs, checked for non-finite
weights).

## Benchmark data

Instance files (`data/Instances_*/*.dat`) follow Archetti et al. (2007)'s
format: a header line (`num_nodes`, `episode_length`, `vehicle_capacity`),
one supplier row (id, x, y, initial inventory, production rate, holding
cost), then one row per retailer (id, x, y, initial inventory, max capacity,
min capacity, per-period demand, holding cost). Parsed by
`src/environment/data_converter.py`.
