# Trained MTPPO checkpoints

Best-validation checkpoints from the three-seed run of 2026-10-07 (500 epochs
each). These are the weights behind every figure in
[`../reports/inference_timing.md`](../reports/inference_timing.md).

| file | seed | epoch | val cost | eval cost | eval cost (re-measured) |
|---|---|---|---|---|---|
| `seed0_epoch325.pt` | 0 | 325 | 9,722 | 9,498 | 9,498 |
| `seed1_epoch450.pt` | 1 | 450 | 9,952 | 9,929 | 9,929 |
| `seed2_epoch450.pt` | 2 | 450 | 9,812 | 9,757 | 9,757 |

Costs are means per instance over the 32 held-out evaluation instances
(replicate abs5, listed in `data/splits/eval_by_replicate.txt`). The last two
columns are computed by different code paths -- the training loop's own
evaluation and `src/scripts/benchmark_inference.py` -- and agree, which is the
check that these files are the ones the report describes.

The epoch chosen is the one with the lowest mean validation cost on replicate
abs4 (`data/splits/val_by_replicate.txt`), a set used for neither training nor
the reported evaluation. On this run the lowest-mean-validation-*gap* criterion
picks the same epoch for all three seeds.

## What is in a file

`torch.save` of a dict with six entries: `critic`, `inv_actor`,
`routing_actor` state dicts, and the matching `critic_optimizer`,
`inv_optimizer`, `routing_optimizer` states. `MTPPO.load` expects all six, so
the optimiser states are kept rather than stripped -- they make a file roughly
twice the size but allow training to be resumed rather than only replayed.

## Loading one

The architecture is not stored in the file, so it has to be constructed with
the same hyperparameters it was trained under. Those are recorded in
`reports/training_runs/seed<N>/config.json`; the ones that affect the shapes
are `loc_dim=2`, `lookback_window=3`,
`gin_dims=[64, 128, 128]`, `mlp_dims=[128, 128]`.

```python
import sys; sys.path[:0] = ["src", "src/agent"]
from environment.irp_env import IRPEnv
from training.mtppo import MTPPO

LOC_DIM, LOOKBACK = 2, 3
agent = MTPPO(
    node_feature_dims={"critic": LOC_DIM + 6, "inventory": LOC_DIM + 3,
                       "routing": 2 * LOC_DIM + 2},
    history_dim=1 + 2 * LOOKBACK,
    global_feature_dim=2,
    gin_dims=[64, 128, 128],
    mlp_dims=[128, 128],
    embed_dim=LOC_DIM + 2 * LOOKBACK,
    loc_dim=LOC_DIM,
    skip_dead_zone=0.1,
    device="cpu",
)
agent.load("models/seed0_epoch325.pt")

env = IRPEnv(
    data_file_path="data/Instances_lowcost_H3/abs5n30.dat",
    loc_dim=LOC_DIM, lookback_window=LOOKBACK,
    product_price=None, penalty_factor=None, delivery_cost=1.0,
)
print(agent.evaluate_episode(env)["total_cost"])
```

`skip_dead_zone` matters at inference, not just in training: it is part of the
transform from the inventory actor's Beta output to a delivery quantity
(`MTPPO._delivery_from_fraction`), so a different value evaluates a different
policy than the one these weights were trained as.

`product_price=None, penalty_factor=None` disables lost-sales pricing, which is
how these were trained -- stockouts are a hard feasibility constraint, matching
Archetti et al. (2007)'s model rather than being priced into the reward.

## Caveat on reproducibility

These weights were trained with inter-node distances floored (`c_ij =
floor(sqrt(dx^2 + dy^2))`, Archetti et al.'s convention) and with period-0
opening-stock holding cost charged at both echelons. Evaluating them against an
environment that rounds distances, or that omits the period-0 term, will not
reproduce the costs above.
