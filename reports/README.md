# Reports

Finished write-ups and the data behind them. Distinct from `src/results/`,
which holds raw per-run output and is gitignored: everything here is committed
and meant to be read.

## Headline results

| | |
|---|---|
| [`inference_timing.md`](inference_timing.md) | Solution speed against Archetti et al. (2007)'s branch-and-cut. The whole 32-instance test set solves in 0.21 s against 5.88 h of reported CPU time; the scaling comparison is the claim that survives the hardware gap. |
| [`training_runs/`](training_runs) | Per-epoch logs, configs and console output for the three seeds behind the published checkpoints in `models/`. |

## Experiments

Each directory holds one screening experiment's data. All used seed 0 unless
stated, so they establish direction rather than final performance.

| experiment | question | outcome |
|---|---|---|
| [`fixed_fraction/`](experiments/fixed_fraction) | Can a fixed rule beat the trained agent? | **Yes.** "Visit when needed, then fill completely" (f=1) reaches a 28.5% test gap against the best seed's 43.1%. Proves the strategy is expressible, so the gap is a learning problem. |
| [`reward_normalisation/`](experiments/reward_normalisation) | Does normalising the two reward streams jointly, or undiscounting, fix just-in-time delivery? | **No, both worse.** Separate z-scoring over-weights holding 1.51x, but removing it also compresses routing, weakening the incentive to skip. Has its own [README](experiments/reward_normalisation/README.md). |
| [`upper_dead_zone/`](experiments/upper_dead_zone) | Does widening "full fill" from a single point to an interval help? | **No, worse** (73.0% vs 49.6%). The greedy policy never crossed the threshold, and the band just below it is actively punished. |
| [`discrete_choices/`](experiments/discrete_choices) | Does a Categorical over {0, ½, 1} beat the continuous Beta? | **Yes, the only change that worked.** Test gap 49.6% -> 35.7%, better on every class. A full fill becomes an atom the policy can sample and select. |
| [`entropy_probe/`](experiments/entropy_probe) | Is the discrete policy's plateau caused by entropy collapse? | **No.** 20x the coefficient held entropy 37x higher and changed the result not at all (35.1% vs 35.7%). |

## Reading the CSVs

Screening files are one row per (run, epoch, split). Costs are means per
instance over the 32 instances of that split; `gap_ml_pct` is against the
estimated maximum-level optimum (Archetti et al.'s published order-up-to
optima, adjusted by their Table IX), `gap_ou_pct` against the published
optima directly and so needs no estimate.

Regenerate any of them with `src/scripts/screen_reward_fixes.py --out <path>`,
or `src/diagnostics/fixed_fraction.py` for the fixed-rule sweep.

## Where the gap stands

The agent reaches a 35.7% test gap; the f=1 fixed rule reaches 28.5%. The
remaining difference is credit assignment: the inventory actor receives one
shared advantage scalar per timestep covering all n retailers, so it can learn
"fill more often on average" but not "fill this retailer and skip that one".
The fixed rule needs no credit assignment at all. That is the untouched
structural problem, and the direct analogue of the self-critical baseline that
fixed routing.
