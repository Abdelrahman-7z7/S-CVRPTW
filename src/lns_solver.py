"""
CSE 557 — Adaptive Large Neighborhood Search (ALNS)
Built on top of the existing solver_core, sa_solver infrastructure.

LNS idea: instead of moving one customer at a time (SA), destroy a chunk
of the solution (remove 20-30% of customers) and repair it from scratch.
Each destroy-repair cycle makes a much bigger structural change than SA's
individual moves, which is exactly what's needed to eliminate vehicles.

ALNS adds adaptive operator selection on top of LNS: operators that have
recently found improvements get higher selection probability. Operators
that haven't found anything useful get lower probability. The weights
update after every iteration based on outcome.

Destroy operators (remove customers from solution):
  1. random        — remove random customers (exploration)
  2. worst         — remove customers from the highest-cost routes (exploitation)
  3. shaw          — remove a cluster of related (nearby + similar TW) customers
  4. smallest_route— remove ALL customers from the smallest route (vehicle elimination)

Repair operators (reinsert removed customers into routes):
  1. greedy        — insert each customer at cheapest feasible position
  2. regret        — insert customer with highest regret first (smarter ordering)

Acceptance: same SA criterion as existing solver (e^(-delta/T)).
This makes ALNS directly comparable to SA and easy to swap in.

Integration points:
  - lns_solve() is a drop-in replacement for sa_solve()
  - Used inside MA as local_search="lns" option
  - Route Eliminator still runs as post-processor
"""

import math
import random
import copy
import time

from solver_core import (
    build_cache, is_route_feasible, route_load,
    best_insertion, best_insertion_across_routes,
    route_travel_time, solution_travel_time,
    solution_score_proxy, get_neighbors, solution_summary
)
from construction import clarke_wright
from sa_solver import calibrate_temperature


# ---------------------------------------------------------------------------
# ALNS Operator Weight Tracker
# ---------------------------------------------------------------------------

class ALNSWeights:
    """
    Tracks performance of each destroy and repair operator.
    Uses roulette-wheel selection biased toward high-performing operators.

    Scoring per iteration outcome:
      New global best found  → +10 points
      Improvement accepted   → +3  points
      Worse move accepted    → +1  point
      Move rejected          → +0  points

    Weights decay toward uniform after each segment of seg_size iterations
    to allow adaptation as the search landscape changes.
    """

    SCORE_NEW_BEST   = 10
    SCORE_IMPROVE    = 3
    SCORE_ACCEPTED   = 1
    SCORE_REJECTED   = 0

    def __init__(self, destroy_ops, repair_ops,
                 seg_size=100, decay=0.8, min_weight=0.1):
        self.destroy_ops = destroy_ops
        self.repair_ops  = repair_ops
        self.seg_size    = seg_size
        self.decay       = decay          # weight update: w = decay*w + (1-decay)*score
        self.min_weight  = min_weight

        self.d_weights   = {op.__name__: 1.0 for op in destroy_ops}
        self.r_weights   = {op.__name__: 1.0 for op in repair_ops}
        self.d_scores    = {op.__name__: 0.0 for op in destroy_ops}
        self.r_scores    = {op.__name__: 0.0 for op in repair_ops}
        self.d_uses      = {op.__name__: 0   for op in destroy_ops}
        self.r_uses      = {op.__name__: 0   for op in repair_ops}
        self._iter       = 0

    def select_destroy(self, rng):
        names   = list(self.d_weights.keys())
        weights = [max(self.d_weights[n], self.min_weight) for n in names]
        chosen  = rng.choices(self.destroy_ops, weights=weights)[0]
        self.d_uses[chosen.__name__] += 1
        return chosen

    def select_repair(self, rng):
        names   = list(self.r_weights.keys())
        weights = [max(self.r_weights[n], self.min_weight) for n in names]
        chosen  = rng.choices(self.repair_ops, weights=weights)[0]
        self.r_uses[chosen.__name__] += 1
        return chosen

    def record(self, d_op, r_op, outcome):
        """
        outcome: 'new_best' | 'improve' | 'accepted' | 'rejected'
        """
        score_map = {
            'new_best': self.SCORE_NEW_BEST,
            'improve':  self.SCORE_IMPROVE,
            'accepted': self.SCORE_ACCEPTED,
            'rejected': self.SCORE_REJECTED,
        }
        s = score_map.get(outcome, 0)
        self.d_scores[d_op.__name__] += s
        self.r_scores[r_op.__name__] += s
        self._iter += 1

        # Update weights at end of each segment
        if self._iter % self.seg_size == 0:
            self._update_weights()

    def _update_weights(self):
        for name in self.d_weights:
            uses = max(self.d_uses[name], 1)
            avg_score = self.d_scores[name] / uses
            self.d_weights[name] = (self.decay * self.d_weights[name]
                                    + (1 - self.decay) * avg_score)
            self.d_scores[name] = 0.0
            self.d_uses[name]   = 0

        for name in self.r_weights:
            uses = max(self.r_uses[name], 1)
            avg_score = self.r_scores[name] / uses
            self.r_weights[name] = (self.decay * self.r_weights[name]
                                    + (1 - self.decay) * avg_score)
            self.r_scores[name] = 0.0
            self.r_uses[name]   = 0

    def summary(self):
        """Print current operator weights for analysis."""
        print("  ALNS Destroy weights:")
        for name, w in sorted(self.d_weights.items(), key=lambda x: -x[1]):
            print(f"    {name:<25}: {w:.3f}")
        print("  ALNS Repair weights:")
        for name, w in sorted(self.r_weights.items(), key=lambda x: -x[1]):
            print(f"    {name:<25}: {w:.3f}")


# ---------------------------------------------------------------------------
# Destroy Operators
# ---------------------------------------------------------------------------

def destroy_random(routes, instance, n_remove, rng):
    """
    Remove n_remove randomly selected customers from the solution.
    Pure random — good for exploration and avoiding search bias.

    Returns (new_routes, removed_customers).
    """
    all_customers = [cid for route in routes for cid in route]
    if len(all_customers) <= n_remove:
        return [list(r) for r in routes], []

    to_remove = set(rng.sample(all_customers, n_remove))
    new_routes = []
    removed = []

    for route in routes:
        new_route = []
        for cid in route:
            if cid in to_remove:
                removed.append(cid)
            else:
                new_route.append(cid)
        if new_route:
            new_routes.append(new_route)

    return new_routes, removed


def destroy_worst(routes, instance, n_remove, rng):
    """
    Remove the n_remove customers that contribute most to travel time.

    For each customer, compute its marginal travel time cost:
    cost(cid) = route_tt(route_with_cid) - route_tt(route_without_cid)

    Remove the most expensive customers first. This targets the customers
    that are hardest to route efficiently — removing them gives the repair
    phase a chance to find better placements.
    """
    # Compute marginal cost per customer
    costs = []
    for r_idx, route in enumerate(routes):
        for pos, cid in enumerate(route):
            route_without = route[:pos] + route[pos+1:]
            tt_with    = route_travel_time(route, instance)
            tt_without = route_travel_time(route_without, instance) if route_without else 0.0
            if tt_with == float("inf"):
                marginal = 0.0
            else:
                marginal = tt_with - tt_without
            costs.append((marginal, cid, r_idx))

    # Sort descending by cost — highest cost customers removed first
    costs.sort(key=lambda x: -x[0])

    # Add some randomness: pick from top 2*n_remove candidates
    pool_size = min(len(costs), max(n_remove * 2, n_remove + 5))
    pool = costs[:pool_size]
    rng.shuffle(pool)
    selected = pool[:n_remove]

    to_remove = {entry[1] for entry in selected}
    new_routes = []
    removed = []

    for route in routes:
        new_route = []
        for cid in route:
            if cid in to_remove:
                removed.append(cid)
            else:
                new_route.append(cid)
        if new_route:
            new_routes.append(new_route)

    return new_routes, removed


def destroy_shaw(routes, instance, n_remove, rng):
    """
    Shaw Removal: remove a cluster of RELATED customers.

    Customers are related if they are:
      - Geographically close (small distance)
      - Have overlapping or similar time windows

    Start with a random customer, then repeatedly add the most related
    unremoved customer to the removal set. This clusters the removal
    geographically, giving the repair phase a chance to find a better
    joint routing for the cluster.

    This is typically the most powerful destroy operator for VRP because
    it creates a large contiguous gap in the solution that the repair
    phase can fill more efficiently.
    """
    all_customers = [cid for route in routes for cid in route]
    if len(all_customers) <= n_remove:
        return [list(r) for r in routes], []

    dist     = instance["distance_matrix"]
    customers = instance["_customers"]

    # Relatedness between two customers: lower = more related
    def relatedness(a, b):
        d = dist[a][b]
        tw_diff = abs(customers[a]["tw_open"] - customers[b]["tw_open"])
        # Normalise: combine distance and TW similarity
        return d + 0.3 * tw_diff

    # Start with a random seed customer
    seed = rng.choice(all_customers)
    to_remove = [seed]
    candidates = [c for c in all_customers if c != seed]

    while len(to_remove) < n_remove and candidates:
        # Pick a random already-removed customer as reference
        ref = rng.choice(to_remove)
        # Sort remaining candidates by relatedness to ref
        candidates.sort(key=lambda c: relatedness(ref, c))
        # Add most related candidate (with slight randomness)
        pick_idx = int(rng.random() ** 3 * len(candidates))  # biased toward front
        pick_idx = min(pick_idx, len(candidates) - 1)
        chosen = candidates.pop(pick_idx)
        to_remove.append(chosen)

    to_remove_set = set(to_remove)
    new_routes = []
    removed = []

    for route in routes:
        new_route = []
        for cid in route:
            if cid in to_remove_set:
                removed.append(cid)
            else:
                new_route.append(cid)
        if new_route:
            new_routes.append(new_route)

    return new_routes, removed


def destroy_smallest_route(routes, instance, n_remove, rng):
    """
    Remove ALL customers from the smallest route (fewest customers).

    This is the vehicle elimination operator. By removing an entire small
    route, we give the repair phase the specific task of finding homes for
    those customers in other routes. If repair succeeds, we eliminated a
    vehicle and saved 1000 score points.

    If the smallest route has more than n_remove customers, fall back to
    removing random customers from it.
    """
    if not routes:
        return [list(r) for r in routes], []

    # Find smallest route
    target_idx = min(range(len(routes)), key=lambda i: len(routes[i]))
    target = routes[target_idx]

    # Remove target route's customers (up to n_remove)
    to_remove = set(target[:n_remove] if len(target) > n_remove else target)

    new_routes = []
    removed = []

    for i, route in enumerate(routes):
        if i == target_idx:
            remaining = [c for c in route if c not in to_remove]
            removed.extend([c for c in route if c in to_remove])
            if remaining:
                new_routes.append(remaining)
        else:
            new_routes.append(list(route))

    return new_routes, removed


# ---------------------------------------------------------------------------
# Repair Operators
# ---------------------------------------------------------------------------

def repair_greedy(routes, removed, instance):
    """
    Reinsert each removed customer at the cheapest feasible position
    across all routes.

    Insertion order: sorted by tw_open ascending (serve tightest early
    windows first — reduces infeasibility cascades).

    If no route can accommodate a customer, open a new route for them.
    """
    customers = instance["_customers"]
    # Sort by time window opening — serve early-window customers first
    removed_sorted = sorted(removed, key=lambda c: customers[c]["tw_open"])

    for cid in removed_sorted:
        r_idx, pos, delta = best_insertion_across_routes(routes, cid, instance)
        if r_idx is not None:
            routes[r_idx].insert(pos, cid)
        else:
            # Open new route — this customer couldn't fit anywhere
            routes.append([cid])

    return routes


def repair_regret(routes, removed, instance):
    """
    Regret-based insertion: at each step, insert the customer whose
    best-vs-second-best insertion cost difference (regret) is highest.

    Intuition: a customer with high regret will pay a large penalty if
    not inserted now (its second-best option is much worse than its best).
    Insert the most "urgent" customer first.

    This typically outperforms greedy repair because it avoids the situation
    where an early greedy choice blocks a later customer's best position.
    """
    customers = instance["_customers"]
    remaining = list(removed)

    while remaining:
        best_cid    = None
        best_r_idx  = None
        best_pos    = None
        best_regret = -float("inf")

        for cid in remaining:
            # Find best and second-best insertion positions across all routes
            insertion_costs = []
            for r_idx, route in enumerate(routes):
                pos, delta = best_insertion(route, cid, instance)
                if pos is not None:
                    insertion_costs.append((delta, r_idx, pos))

            # Also consider opening a new route
            new_route_cost = (instance["distance_matrix"][0][cid] +
                              instance["distance_matrix"][cid][0])
            insertion_costs.append((new_route_cost, -1, 0))  # -1 = new route

            insertion_costs.sort(key=lambda x: x[0])

            if len(insertion_costs) >= 2:
                regret = insertion_costs[1][0] - insertion_costs[0][0]
            else:
                regret = 0.0

            if regret > best_regret:
                best_regret = regret
                best_cid    = cid
                if insertion_costs[0][1] == -1:
                    best_r_idx = None
                    best_pos   = None
                else:
                    best_r_idx = insertion_costs[0][1]
                    best_pos   = insertion_costs[0][2]

        # Insert the highest-regret customer
        if best_cid is None:
            break

        remaining.remove(best_cid)
        if best_r_idx is not None:
            routes[best_r_idx].insert(best_pos, best_cid)
        else:
            routes.append([best_cid])

    return routes


# ---------------------------------------------------------------------------
# Main ALNS loop
# ---------------------------------------------------------------------------

def lns_solve(instance,
              initial_routes=None,
              n_iterations=10_000,
              destroy_fraction=0.25,
              T_initial=None,
              T_final=0.01,
              cooling_rate=0.9997,
              seg_size=100,
              weight_decay=0.8,
              seed=42,
              verbose=True,
              callback=None):
    """
    Adaptive Large Neighborhood Search for S-CVRPTW.

    At each iteration:
      1. Select destroy operator (ALNS weighted)
      2. Select repair operator (ALNS weighted)
      3. Destroy: remove destroy_fraction of customers
      4. Repair: reinsert removed customers
      5. Accept/reject with SA criterion
      6. Update ALNS weights based on outcome

    Parameters:
        instance         : loaded + cached instance dict
        initial_routes   : starting solution (None → Clarke-Wright)
        n_iterations     : total ALNS iterations
        destroy_fraction : fraction of customers removed per iteration (0.2–0.4)
        T_initial        : starting temperature (None → auto-calibrate)
        T_final          : ending temperature
        cooling_rate     : geometric cooling α per iteration
        seg_size         : ALNS weight update segment size
        weight_decay     : ALNS weight decay factor (0 = no memory, 1 = full memory)
        seed             : random seed
        verbose          : print progress
        callback         : callable(iter, best_score, T, weights) for logging

    Returns:
        best_routes (list of routes)
    """
    build_cache(instance)
    rng = random.Random(seed)

    # --- Initial solution ---
    if initial_routes is None:
        if verbose:
            print("  LNS: building initial solution (Clarke-Wright λ=0.85)...")
        routes = clarke_wright(instance, lambda_param=0.85)
    else:
        routes = copy.deepcopy(initial_routes)

    n_customers = instance["n_customers"]
    n_remove = max(2, int(n_customers * destroy_fraction))

    # --- Operator sets ---
    destroy_ops = [
        destroy_random,
        destroy_worst,
        destroy_shaw,
        destroy_smallest_route,
    ]
    repair_ops = [
        repair_greedy,
        repair_regret,
    ]

    weights = ALNSWeights(
        destroy_ops, repair_ops,
        seg_size=seg_size,
        decay=weight_decay
    )

    # --- Temperature calibration ---
    # Estimate using a few random destroy-repair cycles
    if T_initial is None:
        if verbose:
            print("  LNS: calibrating temperature...")
        deltas = []
        current_score = solution_score_proxy(routes, instance)
        for _ in range(50):
            d_op = rng.choice(destroy_ops)
            r_op = rng.choice(repair_ops)
            trial_routes, removed = d_op(
                copy.deepcopy(routes), instance, n_remove, rng
            )
            trial_routes = r_op(trial_routes, list(removed), instance)
            new_score = solution_score_proxy(trial_routes, instance)
            delta = new_score - current_score
            if delta > 0:
                deltas.append(delta)
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            T_initial = max(-avg_delta / math.log(0.8), 1.0)
        else:
            T_initial = 100.0
        if verbose:
            print(f"  LNS: T0 = {T_initial:.2f}")

    T = T_initial
    current_score = solution_score_proxy(routes, instance)
    best_routes   = copy.deepcopy(routes)
    best_score    = current_score
    t_start       = time.time()

    # Track operator usage for final report
    op_stats = {
        'new_best': 0, 'improve': 0,
        'accepted': 0, 'rejected': 0
    }

    if verbose:
        print(f"  LNS: start | {solution_summary(routes, instance)}")
        print(f"  LNS: {n_iterations:,} iterations | "
              f"destroy_fraction={destroy_fraction:.0%} "
              f"({n_remove} customers/iter)")

    for iteration in range(1, n_iterations + 1):
        # --- Select operators ---
        d_op = weights.select_destroy(rng)
        r_op = weights.select_repair(rng)

        # --- Destroy ---
        trial_routes, removed = d_op(
            copy.deepcopy(routes), instance, n_remove, rng
        )

        if not removed:
            weights.record(d_op, r_op, 'rejected')
            continue

        # --- Repair ---
        trial_routes = r_op(trial_routes, list(removed), instance)

        # --- Evaluate ---
        new_score = solution_score_proxy(trial_routes, instance)
        delta = new_score - current_score

        # --- SA acceptance ---
        if delta < 0:
            # Improvement
            routes        = trial_routes
            current_score = new_score

            if current_score < best_score:
                best_score  = current_score
                best_routes = copy.deepcopy(routes)
                outcome     = 'new_best'
                op_stats['new_best'] += 1
            else:
                outcome = 'improve'
                op_stats['improve'] += 1

        elif rng.random() < math.exp(-delta / max(T, 1e-10)):
            # Accept worse solution
            routes        = trial_routes
            current_score = new_score
            outcome       = 'accepted'
            op_stats['accepted'] += 1

        else:
            outcome = 'rejected'
            op_stats['rejected'] += 1

        # --- Update ALNS weights ---
        weights.record(d_op, r_op, outcome)

        # --- Cool down ---
        T = max(T * cooling_rate, T_final)

        # --- Logging ---
        if callback and iteration % 100 == 0:
            callback(iteration, best_score, T, weights)

        if verbose and iteration % (n_iterations // 10) == 0:
            elapsed = time.time() - t_start
            print(f"  LNS @{iteration:,} | "
                  f"{solution_summary(best_routes, instance)} | "
                  f"T={T:.3f} | {elapsed:.0f}s")

    if verbose:
        elapsed = time.time() - t_start
        print(f"  LNS: done in {elapsed:.1f}s | "
              f"{solution_summary(best_routes, instance)}")
        print(f"  LNS outcomes: "
              f"new_best={op_stats['new_best']} "
              f"improve={op_stats['improve']} "
              f"accepted={op_stats['accepted']} "
              f"rejected={op_stats['rejected']}")
        weights.summary()

    return best_routes


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys

    path  = sys.argv[1] if len(sys.argv) > 1 else "istanbul_small_100.json"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3_000

    with open(path) as f:
        instance = json.load(f)
    build_cache(instance)

    print(f"Instance: {instance['name']} ({instance['n_customers']} customers)")
    print(f"Running LNS/ALNS for {iters:,} iterations...")
    print()

    routes = lns_solve(instance, n_iterations=iters, verbose=True)

    from evaluate_solution import validate_solution, score_solution
    sol   = {"team": "LNS", "routes": routes}
    valid, errors = validate_solution(instance, sol)
    infeasible = instance.get("_infeasible_customers", set())
    real_errors = [e for e in errors
                   if not any(f"customer {c}" in e for c in infeasible)]
    print()
    if not real_errors:
        print("Validation: PASSED")
        result = score_solution(instance, sol)
        print(f"Vehicles:       {result['n_vehicles']}")
        print(f"Avg TT:         {result['avg_travel_time']:.2f}")
        print(f"Avg violations: {result['avg_tw_violations']:.2f}")
        print(f"SCORE:          {result['score']:.2f}  (baseline small: 25,127)")
    else:
        print("Validation: FAILED")
        for e in real_errors:
            print(f"  - {e}")