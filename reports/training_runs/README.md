# Training-run logs

Per-epoch logs from the three-seed run of 2026-10-07 (500 epochs each), copied
out of the gitignored `src/results/` so the results behind
[`../inference_timing.md`](../inference_timing.md) are reproducible from the
repository alone. The matching weights are in [`../../models/`](../../models).

One directory per seed, each containing:

| file | contents |
|---|---|
| `config.json` | every hyperparameter the run was launched with |
| `metrics.csv` | one row per epoch: rewards, losses, entropy, critic diagnostics, routing diagnostics |
| `val.csv` | one row per (eval epoch, validation instance) -- replicate abs4, used only to select the best checkpoint |
| `eval.csv` | one row per (eval epoch, held-out instance) -- replicate abs5, the reported evaluation set |
| `routes.csv` | one row per (eval epoch, instance, period): the visit sequence, distance travelled, 2-opt distance over the same stops, and mean nearest-neighbour rank |

## Reading the cost columns

`val.csv` and `eval.csv` carry `inv_cost`, `routing_cost`, `total_cost` and
`total_cost_best_route`. All include holding cost on the period-0 opening stock
at both echelons, which Archetti et al. (2007) charge because their
holding-cost sum runs from t=0.

`total_cost_best_route` is the same episode's cost with its tours re-ordered by
2-opt and the inventory decisions held fixed, so the gap between it and
`total_cost` is the price of the routing actor's sequencing alone.

`metrics.csv` records *rewards*, not costs, and the period-0 term is
deliberately excluded from the reward: it is a per-instance constant that no
action can change, so including it would shift the critic's targets and disturb
reward normalization while carrying no information. Summed `mean_r_inv` is
therefore lower than `eval.csv`'s `inv_cost` by exactly that constant. The
reward is a training signal; the cost is an accounting figure.

## Routing diagnostics in metrics.csv

`mean_nn_rank` is the most informative column: at each hop the eligible nodes
are ranked by distance from where the vehicle stands, normalized to [0, 1].
0.0 means the policy always took the nearest eligible stop, 0.5 is
indistinguishable from choosing uniformly at random, 1.0 always the farthest.
`mean_tour_ratio` is how many times longer the travelled tours were than a
nearest-neighbour tour over the same stops.

Both are worth reading alongside `mean_logit_spread`: on seed 0 the routing
policy degrades late in training (nn_rank 0.087 at epoch 350 to 0.327 by 450,
with logit spread collapsing from 4.84 to 0.65), which is why checkpoint
selection by validation cost matters here and why seed 0's selected epoch is
325 rather than something later.
