# Inventory reward fixes: a negative result

Screened 2026-10-10. Per-epoch data in [`reward_fix_screen.csv`](reward_fix_screen.csv)
(39 rows, run x epoch). Regenerate with `src/scripts/screen_reward_fixes.py`.

**Both candidate fixes made total cost worse. Neither was adopted.** The
switches that produced them were reverted after this screening; they survive in
commit history (see "Reproducing" below) so the result can be re-checked.

## What was tested, and why

The policy converges to near-daily just-in-time deliveries, which is the main
remaining gap to Archetti et al. (2007)'s optima. Two defects were identified
that both make the agent optimise something other than the stated objective
(`min C_h + C_r`, Appendix A Eq. A.3), and both appeared to suppress exactly the
max-level strategy of paying holding cost now to save routing cost later:

1. **Separate reward normalisation.** `RolloutBuffer.compute_advantage` computes
   `norm(r_inv) + norm(r_vrp)`. Z-scoring each stream separately equalises their
   variances, which reprices holding cost against distance. Measured on a
   trained checkpoint over 24 training instances: sd 841.8 for `r_inv` against
   1268.3 for `r_vrp`, so holding is weighted **1.51x** more than the objective
   warrants. Correlation of the distorted reward with the true per-period total
   is 0.9908.
2. **gamma = 0.9.** The objective is undiscounted over a finite horizon, but
   future routing savings are marked down by 19% at H=3 and **41% at H=6**.

Both predict the gap should widen as holding gets cheaper and horizons lengthen,
which is what the report's Table 6 shows (21.2% -> 41.6% -> 63.3% -> 103.1%).
The reasoning was motivated, and it was wrong.

Deliberately **not** tested: any reward term that pays the agent for larger
deliveries. That builds the conclusion into the objective; if larger deliveries
are optimal, a correct objective should produce them.

## Result

Seed 0, 300 epochs per condition, checkpoint selected by lowest mean validation
gap, then evaluated on the held-out test set (abs5):

| run | best epoch | holding | routing | total | gap vs est. ML | gap vs z*_OU |
|---|---|---|---|---|---|---|
| baseline | 275 | 4,115 | 5,371 | 9,486 | 49.6% | 30.1% |
| joint only | 225 | 4,097 | 5,868 | 9,965 | 57.5% | 36.4% |
| joint + gamma=1.0 | 300 | 4,104 | 5,924 | 10,028 | 59.4% | 38.0% |

Gap by class, against the estimated ML optimum:

| run | high-cost H3 | high-cost H6 | low-cost H3 | low-cost H6 |
|---|---|---|---|---|
| baseline | 21.5% | 35.3% | 63.4% | 87.5% |
| joint only | 21.2% | 47.2% | 63.7% | 117.8% |
| joint + gamma=1.0 | 22.1% | 49.1% | 65.2% | 122.0% |

## Why it failed

The 1.51x over-weighting of holding cost is real, but the inference drawn from
it was backwards. Separate z-scoring equalises the two variances, so it does not
only inflate holding -- it equally **compresses routing**, whose spread is the
larger of the two. Removing it therefore reduces the relative weight on routing
cost, and routing cost is what visits are made of. The agent's incentive to skip
a retailer got weaker, not stronger.

The behaviour shows it directly. Percentage of retailers visited per period at
H=6, on the test set:

| run | p1 | p2 | p3 | p4 | p5 | p6 |
|---|---|---|---|---|---|---|
| baseline | 0.0% | 46.5% | 100.0% | 46.5% | 100.0% | 46.5% |
| joint only | 0.0% | 46.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| joint + gamma=1.0 | 0.0% | 46.5% | 100.0% | 100.0% | 100.0% | 100.0% |

The baseline skips roughly half the retailers in periods 4 and 6; both fixes
visit everyone. Delivery fill fraction over visited retailers tells the same
story -- the baseline front-loads to 0.80 in period 2, the fixes do not:

| run | p1 | p2 | p3 | p4 | p5 | p6 |
|---|---|---|---|---|---|---|
| baseline | -- | 0.80 | 0.51 | 0.80 | 0.51 | 0.80 |
| joint only | -- | 0.50 | 0.51 | 0.43 | 0.47 | 0.47 |
| joint + gamma=1.0 | -- | 0.65 | 0.53 | 0.33 | 0.56 | 0.31 |

At H=3 all three conditions are identical (0% / 48.7% / 100%), so the entire
difference lies in the longer horizon -- and the damage concentrates in low-cost
H6, the class where the max-level strategy has most to gain.

The uncomfortable implication is that the distortion was *accidentally helping*,
by exaggerating routing's weight relative to the true objective. That is not a
reason to keep it: it optimises the wrong objective and merely lands closer.
But it does mean separate normalisation is not the cause of the just-in-time
behaviour.

## gamma = 1.0 was stable, just unhelpful

The concern when launching was that undiscounted returns would raise advantage
variance enough to destabilise the critic. They did not:

| run | value loss | explained variance | advantage bias |
|---|---|---|---|
| baseline | 0.679 | 0.929 | +0.008 |
| joint only | 0.197 | 0.960 | +0.114 |
| joint + gamma=1.0 | 0.372 | 0.938 | +0.094 |

Explained variance 0.938 against the baseline's 0.929. The condition failed on
cost, not on stability, so variance is not the reason to reject it.

## What this rules out

The report's Section 6.3 names reward normalisation and the discount factor as
the likely culprits behind just-in-time replenishment. This screening eliminates
both. That leaves the delivery transform itself -- Eq. (7)'s dead zone and the
Beta parameterisation, which govern how much of a retailer's headroom a single
delivery can claim -- as the next place to look.

## Caveats

- **One seed, 300 epochs.** The baseline comparison is its epoch-275 checkpoint,
  not the epoch-325 one the report quotes. These are directions, not performance
  numbers.
- The ~500-unit deficit is larger than the 431-unit spread across the three
  baseline seeds (9,498-9,929), so it is unlikely to be pure seed noise, but one
  seed per condition cannot settle that.
- ML optima are estimated from Archetti et al.'s Table IX per-size factors
  applied to their published `z*_OU`; the `z*_OU` gaps need no estimate.

## Reproducing

The `--joint-reward-normalisation` switch was removed after this screening.
To re-run either condition, check out the commit that introduced it (search
`git log -S joint_reward_normalisation`) and launch with
`--joint-reward-normalisation [--gamma 1.0]`. The analysis script needs no
change: it reads the flag out of each run's `config.json`.
