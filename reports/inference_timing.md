# Solution speed vs Archetti et al. (2007)

Measured 2026-10-07. Per-instance figures in [`inference_timing.csv`](inference_timing.csv);
reproduce with:

```
python src/scripts/benchmark_inference.py \
    --run-dir "src/results/seed0_20261007_130952:mtppo_epoch325.pt" \
    --run-dir "src/results/seed1_20261007_130952:mtppo_epoch450.pt" \
    --run-dir "src/results/seed2_20261007_130952:mtppo_epoch450.pt" \
    --repeats 20
```

## What was measured

Three checkpoints, one per seed, each solving all 32 held-out evaluation
instances: **epoch 325** (seed 0), **epoch 450** (seed 1) and **epoch 450**
(seed 2), from the 500-epoch runs of 2026-10-07.

Both checkpoint-selection criteria agree on this run -- lowest mean validation
cost and lowest mean validation gap pick the same epoch for all three seeds.
They disagreed on two of three seeds in the previous run, which is what made
the earlier epoch numbers confusing.

The objective includes holding cost on the period-0 opening stock at both
echelons, `h_0 * B_0 + sum_i h_i * I_i^0`, which Archetti et al. (2007) charge
because their holding-cost sum runs from t=0. Unlike the previous run, it was
charged throughout training here, not added to the reported cost afterwards.

Timing is the median of 20 greedy rollouts per instance after one discarded
warm-up pass, single-threaded, excluding model load (164 ms, once per
process) and instance parsing (1.6 ms per instance).

Routing diagnostics are switched off while timing. The 2-opt reference tours
the evaluation path normally builds cost more than the forward passes they
exist to measure, and they are not part of producing a solution; a test pins
that the fast path returns a bit-identical solution
(`test_skipping_diagnostics_does_not_change_the_solution`).

The measurement is stable: median relative standard deviation across the 20
repeats is 1.1%, worst instance 4.8%
(Instances_highcost_H3/abs5n50, seed 2: median 10.64 ms
against a minimum of 10.40 ms, so a few slow outliers
rather than a shifted distribution -- which is why the median, not the mean, is
the headline statistic throughout).

## Results

Our median solve time against the branch-and-cut CPU times of Archetti et al.
(2007), Table I, VMIR-OU variant. Our column is averaged over the three seeds.

| class | H | n | ours (ms) | B&C (s) | ratio |
|---|---|---|---|---|---|
| highcost | 3 | 5 | 1.4 | 0.0 | – |
| highcost | 3 | 10 | 2.4 | 0.0 | – |
| highcost | 3 | 15 | 3.4 | 1.8 | 526× |
| highcost | 3 | 20 | 4.3 | 7.6 | 1,775× |
| highcost | 3 | 25 | 4.9 | 24.4 | 4,999× |
| highcost | 3 | 30 | 5.9 | 101.6 | 17,144× |
| highcost | 3 | 35 | 7.2 | 237.0 | 32,986× |
| highcost | 3 | 40 | 8.3 | 561.6 | 67,843× |
| highcost | 3 | 45 | 9.6 | 1,189.6 | 123,500× |
| highcost | 3 | 50 | 10.6 | 2,842.8 | 268,372× |
| lowcost | 3 | 5 | 1.4 | 0.0 | – |
| lowcost | 3 | 10 | 2.4 | 0.0 | – |
| lowcost | 3 | 15 | 3.4 | 1.0 | 292× |
| lowcost | 3 | 20 | 4.3 | 7.4 | 1,723× |
| lowcost | 3 | 25 | 4.9 | 27.8 | 5,723× |
| lowcost | 3 | 30 | 5.9 | 79.6 | 13,503× |
| lowcost | 3 | 35 | 7.2 | 208.6 | 28,845× |
| lowcost | 3 | 40 | 8.3 | 686.4 | 83,135× |
| lowcost | 3 | 45 | 9.6 | 1,049.6 | 109,438× |
| lowcost | 3 | 50 | 10.7 | 3,572.4 | 335,180× |
| highcost | 6 | 5 | 3.1 | 0.0 | – |
| highcost | 6 | 10 | 5.7 | 6.0 | 1,052× |
| highcost | 6 | 15 | 8.2 | 31.6 | 3,839× |
| highcost | 6 | 20 | 10.6 | 530.2 | 50,081× |
| highcost | 6 | 25 | 12.3 | 1,018.2 | 82,741× |
| highcost | 6 | 30 | 15.0 | 4,099.8 | 272,547× |
| lowcost | 6 | 5 | 3.1 | 0.0 | – |
| lowcost | 6 | 10 | 5.7 | 5.0 | 881× |
| lowcost | 6 | 15 | 8.2 | 29.2 | 3,540× |
| lowcost | 6 | 20 | 10.5 | 558.0 | 52,950× |
| lowcost | 6 | 25 | 12.3 | 1,063.0 | 86,373× |
| lowcost | 6 | 30 | 15.0 | 3,216.0 | 214,880× |

Whole evaluation set, one seed: **0.214 s** against **5.88 h** of
reported branch-and-cut time.

## Scaling

The ratio above mixes an algorithmic difference with twenty years of hardware,
so the scaling behaviour is the more defensible claim -- it is measured within
each method and is unaffected by the hardware gap.

| | n = 5 -> 50 (H=3) | H = 3 -> 6 (n=30) |
|---|---|---|
| ours | 1.4 -> 10.6 ms (**7.8x** for 10x the retailers) | 5.9 -> 15.0 ms (**2.5x** for 2x the periods) |
| B&C (lowcost) | 0.0 -> 3,572 s (**>3,500x** from n=15 alone) | 79.6 -> 3,216 s (**40x**) |

Our cost is close to linear in both the number of retailers and the horizon,
which is what the architecture predicts: one inventory decision per retailer
per period, and one routing forward pass per hop. Branch-and-cut grows
exponentially in both, and at n=50 H=3 it is already within an order of
magnitude of the paper's two-hour cutoff.

## Comparison with the previous run

The 2026-10-01 runs were trained before inter-node distances were floored to
match Archetti's `c_ij`, and before the period-0 term was charged during
training rather than added to the reported cost afterwards. Re-running the
three seeds improved every one of them. Both sets are measured on the same
objective, so this is like-for-like:

| seed | new epoch | new eval cost | old epoch | old eval cost | change |
|---|---|---|---|---|---|
| 0 | 325 | **9,498** | 475 | 9,731 | -233 |
| 1 | 450 | **9,929** | 400 | 10,150 | -221 |
| 2 | 450 | **9,757** | 375 | 9,863 | -105 |

Improvements of 1.1-2.4%. The floored distances being active during training
rather than only at evaluation is the most likely cause, but three seeds moving
in the same direction is suggestive, not conclusive.

## Caveats

These must travel with the numbers.

1. **The two times answer different questions.** Archetti et al. report time to
   *prove optimality* (or hit a two-hour cutoff); their solutions are exact.
   Ours is one greedy forward pass returning a feasible but suboptimal
   solution. This is a comparison of solution *speed*, not of methods.
2. **Hardware differs by about two decades.** Their figures are from a
   single-core 2.8 GHz Pentium IV; ours from an Apple M5 pinned to one thread,
   which removes the core-count difference but not the per-core one. A
   plausible 10-50x single-core improvement would still leave roughly
   2,000-10,000x at n=50 H=3 -- but that adjustment is an estimate, not a
   measurement, and the scaling section is the claim that does not depend on it.
3. **Their figures are five-instance means.** Each Table I row averages
   replicates abs1-abs5; our evaluation set holds only abs5, so each row pairs
   one instance against a five-instance average.
4. **No solution-quality gap is reported here.** Their objective values are not
   stored in this repository, and quoting a percentage gap would require them.
   The validation gap used for checkpoint selection is measured against the
   2-opt-routed cost of the same solution, a different and much weaker
   reference than a proven optimum.
5. **The published CPU times are transcribed by hand** from the paper into
   `data/benchmarks/archetti2007_vmir_ou_times.csv`, which is not otherwise
   checkable from inside this repository. Spot-check them against Table I
   before quoting any figure here.
