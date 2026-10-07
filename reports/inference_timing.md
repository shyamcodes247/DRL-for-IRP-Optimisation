# Solution speed vs Archetti et al. (2007)

Measured 2026-10-02. Per-instance figures in [`inference_timing.csv`](inference_timing.csv);
reproduce with:

```
python src/scripts/benchmark_inference.py \
    --run-dir "src/results/seed0_20261001_172956:mtppo_epoch400.pt" \
    --run-dir "src/results/seed1_20261001_172956:mtppo_epoch450.pt" \
    --run-dir "src/results/seed2_20261001_172955:mtppo_epoch375.pt" \
    --repeats 20
```

## What was measured

Three checkpoints, one per seed, each solving all 32 held-out evaluation
instances: **epoch 400** (seed 0), **epoch 450** (seed 1) and **epoch 375**
(seed 2). These are the checkpoints with the lowest mean validation *gap* —
percentage above the estimated optimum — which is the selection criterion used
here in preference to raw cost.

Note that this is a different set from the one `mtppo_best.pt` holds. That file
is written by the training loop on lowest mean validation *cost*, which gives
epochs 475, 400 and 375 instead. The two criteria agree on seed 2 and disagree
on seeds 0 and 1. The gap criterion turned out to generalise better on both of
the seeds where they differ (see "Selection criterion" below), so nothing is
lost by preferring it, but the distinction matters when reading the numbers:
the costs below do **not** match the figures the training runs printed for
their own best checkpoints.

The reported objective includes holding cost on the period-0 opening stock at
both echelons (`h_0 * B_0 + sum_i h_i * I_i^0`), which Archetti et al. (2007)
charge because their holding-cost sum runs from t=0. It was missing until
2026-10-07 and is worth a mean of 869 per instance here, 8.6% of total cost, so
any figure quoted from an earlier version of this file is understated by about
that much. See "Period-0 holding cost" below.

Timing is the median of 20 greedy rollouts per instance after one discarded
warm-up pass, single-threaded, excluding model load (156 ms, once per process)
and instance parsing (1.3 ms per instance).

Routing diagnostics are switched off while timing. The 2-opt reference tours
the evaluation path normally builds cost more than the forward passes they
exist to measure, and they are not part of producing a solution; a test pins
that the fast path returns a bit-identical solution
(`test_skipping_diagnostics_does_not_change_the_solution`).

The measurement is stable: median relative standard deviation across the 20
repeats is 1.0%. One instance reaches 18.8% (seed 0, highcost H6 n=30), but its
median of 15.04 ms sits close to its minimum of 14.00 ms, so that spread is a
few slow outliers rather than a shifted distribution — which is why the median,
not the mean, is the headline statistic throughout.

## Results

Our median solve time against the branch-and-cut CPU times of Archetti et al.
(2007), Table I, VMIR-OU variant. Our column is averaged over the three seeds.

| class | H | n | ours (ms) | B&C (s) | ratio |
|---|---|---|---|---|---|
| lowcost | 3 | 5 | 1.4 | 0.0 | – |
| lowcost | 3 | 10 | 2.4 | 0.0 | – |
| lowcost | 3 | 15 | 3.4 | 1.0 | 293× |
| lowcost | 3 | 20 | 4.3 | 7.4 | 1,715× |
| lowcost | 3 | 25 | 4.9 | 27.8 | 5,729× |
| lowcost | 3 | 30 | 5.9 | 79.6 | 13,494× |
| lowcost | 3 | 35 | 7.2 | 208.6 | 29,121× |
| lowcost | 3 | 40 | 8.2 | 686.4 | 83,504× |
| lowcost | 3 | 45 | 9.6 | 1,049.6 | 109,715× |
| lowcost | 3 | 50 | 10.5 | 3,572.4 | 338,639× |
| highcost | 3 | 5 | 1.4 | 0.0 | – |
| highcost | 3 | 10 | 2.4 | 0.0 | – |
| highcost | 3 | 15 | 3.5 | 1.8 | 517× |
| highcost | 3 | 20 | 4.3 | 7.6 | 1,750× |
| highcost | 3 | 25 | 4.9 | 24.4 | 4,984× |
| highcost | 3 | 30 | 6.0 | 101.6 | 16,951× |
| highcost | 3 | 35 | 7.2 | 237.0 | 32,713× |
| highcost | 3 | 40 | 8.3 | 561.6 | 67,833× |
| highcost | 3 | 45 | 9.6 | 1,189.6 | 123,591× |
| highcost | 3 | 50 | 10.6 | 2,842.8 | 267,320× |
| lowcost | 6 | 5 | 3.4 | 0.0 | – |
| lowcost | 6 | 10 | 5.8 | 5.0 | 855× |
| lowcost | 6 | 15 | 8.5 | 29.2 | 3,440× |
| lowcost | 6 | 20 | 10.9 | 558.0 | 50,971× |
| lowcost | 6 | 25 | 13.3 | 1,063.0 | 79,662× |
| lowcost | 6 | 30 | 15.5 | 3,216.0 | 206,947× |
| highcost | 6 | 5 | 3.4 | 0.0 | – |
| highcost | 6 | 10 | 5.9 | 6.0 | 1,014× |
| highcost | 6 | 15 | 8.5 | 31.6 | 3,709× |
| highcost | 6 | 20 | 11.0 | 530.2 | 48,237× |
| highcost | 6 | 25 | 13.3 | 1,018.2 | 76,580× |
| highcost | 6 | 30 | 15.6 | 4,099.8 | 262,873× |

Whole evaluation set, one seed: **0.227 s** against **5.88 h** of reported
branch-and-cut time.

## Scaling

The ratio above mixes an algorithmic difference with twenty years of hardware,
so the scaling behaviour is the more defensible claim — it is measured within
each method and is unaffected by the hardware gap.

| | n = 5 → 50 (H=3) | H = 3 → 6 (n=30) |
|---|---|---|
| ours | 1.4 → 10.6 ms (**7.7×** for 10× the retailers) | 5.9 → 15.6 ms (**2.6×** for 2× the periods) |
| B&C (lowcost) | 0.0 → 3,572 s (**>3,500×** from n=15 alone) | 79.6 → 3,216 s (**40×**) |

Our cost is close to linear in both the number of retailers and the horizon,
which is what the architecture predicts: one inventory decision per retailer
per period, and one routing forward pass per hop. Branch-and-cut grows
exponentially in both, and at n=50 H=3 it is already within an order of
magnitude of the paper's two-hour cutoff.

The slightly super-linear H factor (2.6× rather than 2×) is expected: a longer
horizon raises the number of stops served per period as well as the number of
periods, so hops grow a little faster than H.

## Selection criterion

Timing is indifferent to which checkpoint is loaded — the architecture is
identical, so only the weights change — but solution quality is not. For the
record, mean total cost per instance on the held-out set:

| seed | gap-selected | cost-selected (`mtppo_best.pt`) |
|---|---|---|
| 0 | epoch 400 → **9,696** | epoch 475 → 9,731 |
| 1 | epoch 450 → **10,028** | epoch 400 → 10,150 |
| 2 | epoch 375 → 9,863 | epoch 375 → 9,863 (same checkpoint) |

On both seeds where the criteria disagree, the gap-selected checkpoint is
*cheaper* on the held-out set than the one chosen by validation cost. With two
data points this is suggestive rather than established, but it is at least
consistent with a gap measure being less sensitive to which instances happen to
be in the validation pool: a mean of raw costs is dominated by the large-n
instances, whereas a mean of percentages weights every instance alike.

## Caveats

These must travel with the numbers.

1. **The two times answer different questions.** Archetti et al. report time to
   *prove optimality* (or hit a two-hour cutoff); their solutions are exact.
   Ours is one greedy forward pass returning a feasible but suboptimal
   solution. This is a comparison of solution *speed*, not of methods.
2. **Hardware differs by about two decades.** Their figures are from a
   single-core 2.8 GHz Pentium IV. Timing here is pinned to one thread, which
   removes the core-count difference but not the per-core one. A plausible
   10–50× single-core improvement would still leave roughly 2,000–10,000× at
   n=50 H=3 — but that adjustment is an estimate, not a measurement, and the
   scaling section above is the claim that does not depend on it.
3. **Their figures are five-instance means.** Each Table I row averages
   replicates abs1–abs5; our evaluation set holds only abs5, so each row pairs
   one instance against a five-instance average.
4. **No solution-quality gap is reported here**, for two reasons: their
   objective values are not stored in this repository, and our objective
   includes a lost-sales penalty term that their model does not have (it
   forbids stockouts by constraint). Quoting a percentage gap would require
   both their published objectives and a reconciled objective function. The
   validation gap used to select these checkpoints was computed outside this
   repository and cannot be reproduced from the logged data alone -- and was
   computed before the period-0 term was added, so it understated the true gap
   (though not in a way that changes which checkpoint it selects; see below).
5. **The published CPU times are transcribed by hand** from the paper into
   `data/benchmarks/archetti2007_vmir_ou_times.csv`, which is not otherwise
   checkable from inside this repository. Spot-check them against Table I
   before quoting any figure here.

## Period-0 holding cost

Added 2026-10-07, after re-reading the paper. Archetti et al. (2007) sum
holding cost from t=0, which charges the opening stock given in the instance
file as well as every level the policy goes on to produce. The environment was
charging only t=1..H, so the reported objective was short by

    h_0 * B_0  +  sum_i h_i * I_i^0

worth a mean of 869 per instance on the evaluation set: 5.1% of total cost from
the depot half and 3.5% from the retailer half, 8.6% together. It is largest in
relative terms on the short-horizon high-cost instances (21% on highcost H3
n=50) and smallest on the long-horizon low-cost ones (1.0% on lowcost H6 n=15),
because a longer horizon gives the decision-dependent part of the objective more
periods to accumulate while this term stays fixed.

**No retraining was needed, and no checkpoint was reselected.** Every factor in
the expression comes from the instance file, so the term is a per-instance
constant that no action can move -- asserted directly in
`test_initial_holding_cost_is_constant_under_every_policy`, which drives three
different delivery policies to the end of an episode and checks the value never
budges. Three consequences follow:

  - *The optimal policy is unchanged.* Adding a constant to an objective cannot
    reorder the policies being compared.
  - *The learning signal is unchanged.* The term is deliberately excluded from
    `r_inv`, because adding a constant to the reward would shift the critic's
    targets and disturb reward normalization while carrying no information. The
    reported `inv_cost` therefore exceeds summed `r_inv` by exactly this
    amount; the reward is a training signal and the cost is an accounting
    figure, and they are not the same quantity.
  - *Gap-based checkpoint selection is unchanged.* Mean gap over instances
    becomes `mean(cost_i/opt_i) + mean(c_i/opt_i) - 1`, and the second term
    does not depend on the checkpoint, so the argmin is exactly where it was.
    Epochs 400/450/375 remain the selected checkpoints.

Pass `--no-initial-holding` to reproduce the older accounting.
