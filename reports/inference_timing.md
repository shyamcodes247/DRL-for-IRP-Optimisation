# Solution speed vs Archetti et al. (2007)

Measured 2026-10-02. Per-instance figures in [`inference_timing.csv`](inference_timing.csv);
reproduce with:

```
python src/scripts/benchmark_inference.py \
    --run-dir src/results/seed0_20261001_172956 \
    --run-dir src/results/seed1_20261001_172956 \
    --run-dir src/results/seed2_20261001_172955 \
    --repeats 20
```

## What was measured

Each of the three seeds' best-validation checkpoints (`mtppo_best.pt`, saved at
epochs **475**, **400** and **375** for seeds 0, 1 and 2) solved all 32 held-out
evaluation instances. Timing is the median of 20 greedy rollouts per instance
after one discarded warm-up pass, single-threaded, excluding model load
(113 ms, once per process) and instance parsing (0.9 ms per instance).

Routing diagnostics are switched off while timing. The 2-opt reference tours
the evaluation path normally builds cost more than the forward passes they
exist to measure, and they are not part of producing a solution; a test pins
that the fast path returns a bit-identical solution
(`test_skipping_diagnostics_does_not_change_the_solution`).

The measurement is stable: median relative standard deviation across the 20
repeats is 0.9% (worst instance 4.0%).

## Results

Our median solve time against the branch-and-cut CPU times of Archetti et al.
(2007), Table I, VMIR-OU variant. Our column is averaged over the three seeds
(spread between them is measurement noise — the architecture is identical).

| class | H | n | ours (ms) | B&C (s) | ratio |
|---|---|---|---|---|---|
| lowcost | 3 | 5 | 1.4 | 0.0 | – |
| lowcost | 3 | 10 | 2.4 | 0.0 | – |
| lowcost | 3 | 15 | 3.4 | 1.0 | 292× |
| lowcost | 3 | 20 | 4.4 | 7.4 | 1,678× |
| lowcost | 3 | 25 | 4.9 | 27.8 | 5,718× |
| lowcost | 3 | 30 | 5.9 | 79.6 | 13,423× |
| lowcost | 3 | 35 | 7.2 | 208.6 | 29,060× |
| lowcost | 3 | 40 | 8.3 | 686.4 | 83,198× |
| lowcost | 3 | 45 | 9.9 | 1,049.6 | 106,043× |
| lowcost | 3 | 50 | 10.6 | 3,572.4 | 337,367× |
| highcost | 3 | 5 | 1.4 | 0.0 | – |
| highcost | 3 | 10 | 2.4 | 0.0 | – |
| highcost | 3 | 15 | 3.4 | 1.8 | 523× |
| highcost | 3 | 20 | 4.4 | 7.6 | 1,708× |
| highcost | 3 | 25 | 4.9 | 24.4 | 4,971× |
| highcost | 3 | 30 | 6.0 | 101.6 | 17,049× |
| highcost | 3 | 35 | 7.2 | 237.0 | 32,794× |
| highcost | 3 | 40 | 8.3 | 561.6 | 67,641× |
| highcost | 3 | 45 | 10.0 | 1,189.6 | 118,900× |
| highcost | 3 | 50 | 10.7 | 2,842.8 | 265,075× |
| lowcost | 6 | 5 | 3.4 | 0.0 | – |
| lowcost | 6 | 10 | 5.9 | 5.0 | 848× |
| lowcost | 6 | 15 | 8.5 | 29.2 | 3,437× |
| lowcost | 6 | 20 | 11.1 | 558.0 | 50,395× |
| lowcost | 6 | 25 | 13.4 | 1,063.0 | 79,481× |
| lowcost | 6 | 30 | 15.6 | 3,216.0 | 205,667× |
| highcost | 6 | 5 | 3.4 | 0.0 | – |
| highcost | 6 | 10 | 5.9 | 6.0 | 1,016× |
| highcost | 6 | 15 | 8.5 | 31.6 | 3,709× |
| highcost | 6 | 20 | 11.1 | 530.2 | 47,643× |
| highcost | 6 | 25 | 13.4 | 1,018.2 | 75,886× |
| highcost | 6 | 30 | 15.6 | 4,099.8 | 262,136× |

Whole evaluation set, one seed: **0.23 s** against **5.88 h** of reported
branch-and-cut time.

## Scaling

The ratio above mixes an algorithmic difference with twenty years of hardware,
so the scaling behaviour is the more defensible claim — it is measured within
each method and is unaffected by the hardware gap.

| | n = 5 → 50 (H=3) | H = 3 → 6 (n=30) |
|---|---|---|
| ours | 1.4 → 10.6 ms (**7.7×** for 10× the retailers) | 6.0 → 15.6 ms (**2.6×** for 2× the periods) |
| B&C (lowcost) | 0.0 → 3,572 s (**>3,500×** from n=15 alone) | 79.6 → 3,216 s (**40×**) |

Our cost is close to linear in both the number of retailers and the horizon,
which is what the architecture predicts: one inventory decision per retailer
per period, and one routing forward pass per hop. Branch-and-cut grows
exponentially in both, and at n=50 H=3 it is already within an order of
magnitude of the paper's two-hour cutoff.

The slightly super-linear H factor (2.6× rather than 2×) is expected: a longer
horizon raises the number of stops served per period as well as the number of
periods, so hops grow a little faster than H.

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
   objective values are not yet stored in this repository, and our objective
   includes a lost-sales penalty term that their model does not have (it
   forbids stockouts by constraint). Quoting a percentage gap would require
   both their published objectives and a reconciled objective function.
5. Our costs in the CSV (means of 8,862 / 9,281 / 8,994 per instance for seeds
   0/1/2) reproduce the figures from the training runs' own evaluation, which
   confirms the intended checkpoints were loaded.
