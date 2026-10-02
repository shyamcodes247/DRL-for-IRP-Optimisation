from typing import Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt


def distance_matrix(coords: npt.NDArray) -> npt.NDArray:
    """
    Pairwise node distances, floored exactly as `IRPEnv._get_distance` floors
    them (Archetti et al., 2007's `c_ij`), so a reference tour computed here
    is directly comparable to the distance the environment actually charged.
    If the two ever disagree, every excess ratio in the diagnostics is
    measured against the wrong yardstick.

    Args:
        coords: (num_nodes, loc_dim) array, depot first (see `IRPEnv`'s node
            indexing convention).
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.floor(np.sqrt((diff ** 2).sum(axis=-1)))


def tour_length(dist: npt.NDArray, order: Sequence[int]) -> float:
    """Closed tour depot -> order... -> depot. An empty order costs nothing."""
    if len(order) == 0:
        return 0.0
    total = dist[0, order[0]] + dist[order[-1], 0]
    for a, b in zip(order[:-1], order[1:]):
        total += dist[a, b]
    return float(total)


def nearest_neighbour_tour(dist: npt.NDArray, nodes: Sequence[int]) -> List[int]:
    """Greedy nearest-neighbour ordering from the depot. O(k^2), no randomness."""
    remaining, order, current = list(nodes), [], 0
    while remaining:
        nxt = min(remaining, key=lambda n: dist[current, n])
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return order


def two_opt_tour(dist: npt.NDArray, nodes: Sequence[int], max_passes: int = 40) -> List[int]:
    """
    Nearest-neighbour start improved by 2-opt. Capped at `max_passes` sweeps
    so evaluation stays bounded on the larger instances; the cap is rarely
    reached, and a partially-improved tour is still a fair reference (it can
    only make the agent look better, never worse).
    """
    order = nearest_neighbour_tour(dist, nodes)
    if len(order) < 4:
        return order
    best = tour_length(dist, order)
    for _ in range(max_passes):
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                candidate = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                length = tour_length(dist, candidate)
                if length < best - 1e-9:
                    order, best, improved = candidate, length, True
        if not improved:
            break
    return order


def self_critical_advantage(travelled: float, reference: float, num_stops: int) -> float:
    """
    Scores one tour against a reference ordering of the same stops:
    `(reference - travelled) / reference`. Positive means the policy beat the
    reference, negative means it did worse, and the scale is the fraction of
    the reference tour saved — comparable across instances of very different
    size, unlike a raw distance difference.

    This is the self-critical baseline of Kool et al. (2019) / POMO, with a
    fixed heuristic standing in for their greedy-rollout baseline. Unlike the
    shared advantage it is deliberately *not* re-centred across the batch:
    the sign is the signal. Re-centring a batch in which every tour lost to
    the reference would relabel the least-bad tours as good.

    Returns 0 (no signal) when there was no ordering decision to make: fewer
    than two stops, or a degenerate reference.
    """
    if num_stops < 2 or reference <= 0:
        return 0.0
    return (reference - travelled) / reference


class RouteRecorder:
    """
        Accumulates routing decisions for one episode so a tour can be
        compared against what a trivial heuristic would have done on the very
        same set of stops.

        Two different failures both show up as "routing cost is too high", and
        they need different fixes, so they are measured separately:

          - visiting too often -> `visits`, the number of stops served;
          - sequencing badly -> `nn_ratio`/`opt_ratio` (how much longer the
            travelled tour is than a nearest-neighbour / 2-opt tour over the
            same stops) and `nn_rank`, below.

        `nn_rank` is the sharpest per-decision signal. At each hop the
        eligible nodes are ranked by distance from where the vehicle actually
        stands; rank 1 means the policy took the closest one. Reported
        normalized to [0, 1] via `(rank - 1) / (eligible - 1)`, which is
        comparable across hops even though the number of eligible nodes
        shrinks along a tour:

            0.0 -> always takes the nearest eligible stop
            0.5 -> indistinguishable from choosing uniformly at random
            1.0 -> always takes the farthest

        A policy that cannot see the vehicle's position cannot do better than
        ~0.5 except by luck, so this distinguishes "the routing actor has not
        learned geometry yet" from "it has, and the cost is coming from
        somewhere else".
    """

    def __init__(self, coords: npt.NDArray, delivery_cost: float) -> None:
        """
        Args:
            coords: (num_nodes, loc_dim) node coordinates, depot first.
            delivery_cost: `IRPEnv.delivery_cost`, used to convert the routing
                reward back into raw travel distance.
        """
        self.dist = distance_matrix(np.asarray(coords, dtype=float))
        self.delivery_cost = max(float(delivery_cost), 1e-12)
        self.periods: List[Dict[str, float]] = []
        self._reset_period()

    def _reset_period(self) -> None:
        self._sequence: List[int] = []
        self._travelled = 0.0
        self._ranks: List[float] = []
        self._spreads: List[float] = []

    def record_hop(
        self,
        position: int,
        eligible: npt.NDArray,
        action: int,
        r_vrp: float,
        logit_spread: Optional[float] = None,
    ) -> None:
        """
        Records one routing action, before the environment state advances.

        Args:
            position: Node the vehicle currently occupies (`vehicle_position`).
            eligible: Indices the mask left selectable at this hop.
            action: The node index chosen.
            r_vrp: Routing reward returned for the hop; converted back to raw
                distance so the total matches what the tour actually cost.
            logit_spread: Standard deviation of the selectable nodes' logits
                at this hop, from `RoutingActor.act`. Trending to zero means
                the policy is flattening towards uniform — visible well
                before it shows up in tour costs.
        """
        self._travelled += -float(r_vrp) / self.delivery_cost
        if logit_spread is not None and not np.isnan(logit_spread):
            self._spreads.append(float(logit_spread))
        if action != 0:
            self._sequence.append(int(action))
        candidates = [int(n) for n in np.asarray(eligible).ravel() if int(n) != 0]
        if action != 0 and len(candidates) > 1:
            here = self.dist[position]
            rank = 1 + sum(1 for n in candidates if here[n] < here[action])
            self._ranks.append((rank - 1) / (len(candidates) - 1))

    def close_period(self, reference: str = "nearest") -> None:
        """
        Closes the current tour and scores it against a reference ordering of
        the same stops. `reference` is "nearest" (cheap, for training) or
        "two_opt" (tighter, for evaluation).
        """
        served = sorted(set(self._sequence))
        builder = two_opt_tour if reference == "two_opt" else nearest_neighbour_tour
        ref_length = tour_length(self.dist, builder(self.dist, served)) if served else 0.0
        self.periods.append({
            "visits": float(len(served)),
            "travelled": self._travelled,
            "reference": ref_length,
            "ratio": self._travelled / ref_length if ref_length > 0 else float("nan"),
            "nn_rank": float(np.mean(self._ranks)) if self._ranks else float("nan"),
            "logit_spread": float(np.mean(self._spreads)) if self._spreads else float("nan"),
            "advantage": self_critical_advantage(self._travelled, ref_length, len(served)),
            "sequence": list(self._sequence),
        })
        self._reset_period()

    def last_advantage(self) -> float:
        """The routing advantage for the period just closed (see `self_critical_advantage`)."""
        return self.periods[-1]["advantage"] if self.periods else 0.0

    def summary(self) -> Dict[str, float]:
        """Episode-level means, safe to log directly. Empty tours are skipped."""
        if not self.periods:
            return {}
        served = [p for p in self.periods if p["visits"] > 0]
        if not served:
            served = self.periods

        def mean(key: str) -> float:
            values = [p[key] for p in served if not np.isnan(p[key])]
            return float(np.mean(values)) if values else float("nan")

        return {
            "visits_per_period": mean("visits"),
            "tour_len": mean("travelled"),
            "tour_ref_len": mean("reference"),
            "tour_ratio": mean("ratio"),
            "nn_rank": mean("nn_rank"),
            "logit_spread": mean("logit_spread"),
            "routing_advantage": mean("advantage"),
            "total_distance": float(sum(p["travelled"] for p in self.periods)),
            "total_reference": float(sum(p["reference"] for p in self.periods)),
        }


def aggregate(summaries: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Averages per-episode summaries across instances, ignoring empty ones."""
    summaries = [s for s in summaries if s]
    if not summaries:
        return {}
    keys = summaries[0].keys()
    out: Dict[str, float] = {}
    for key in keys:
        values = [s[key] for s in summaries if key in s and not np.isnan(s[key])]
        out[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")
    return out
