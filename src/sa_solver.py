"""
CSE 557 — Simulated Annealing Solver
Single-solution trajectory with 6 neighborhood operators.
Accepts worse solutions probabilistically to escape local optima.

Operators:
  relocate    — move one customer to best position in another route
  2-opt*      — swap tails between two routes
  or_opt_k    — move k consecutive customers (k=1,2,3) anywhere
  intra_2opt  — reverse segment within one route

Temperature is auto-calibrated on the starting solution.
Restart mechanism fires when stagnation is detected.
"""

import math
import random
import copy
import time

from solver_core import (
    build_cache, det_tt, is_route_feasible, route_load,
    best_insertion, route_travel_time, solution_travel_time,
    solution_score_proxy, get_neighbors, assert_valid, solution_summary
)
from construction import clarke_wright


# ---------------------------------------------------------------------------
# Move operators — each returns new_routes or None (infeasible/no change)
# ---------------------------------------------------------------------------

def _route_feasible_fast(route, instance):
    """Inline feasibility check used inside hot loops."""
    return is_route_feasible(route, instance)


def move_relocate(routes, instance, rng, n_nearest=15):
    """
    Remove a random customer from its route and insert it at the best
    feasible position in a *different* route (using nearest-neighbour
    candidate list to avoid O(N²) full search).

    Returns new routes or None if no improving/feasible move found.
    """
    if len(routes) < 2:
        return None

    n_routes = len(routes)
    # Pick a random non-empty source route
    r_from = rng.randrange(n_routes)
    if not routes[r_from]:
        return None

    pos_from = rng.randrange(len(routes[r_from]))
    cid = routes[r_from][pos_from]

    # Try inserting into nearest-neighbour routes first
    neighbors = get_neighbors(cid, instance, n_nearest)
    candidate_routes = set()
    for nb in neighbors:
        for r_idx, route in enumerate(routes):
            if r_idx != r_from and nb in route:
                candidate_routes.add(r_idx)
                break
        if len(candidate_routes) >= 5:
            break

    # Also include a few random routes for diversity
    for _ in range(3):
        candidate_routes.add(rng.randrange(n_routes))
    candidate_routes.discard(r_from)

    best_pos = None
    best_r = None
    best_delta = float("inf")

    temp_route_from = list(routes[r_from])
    temp_route_from.pop(pos_from)

    for r_to in candidate_routes:
        route_to = routes[r_to]
        # Capacity check
        if route_load(route_to, instance) + instance["_customers"][cid]["demand"] > instance["vehicle_capacity"]:
            continue
        pos, delta = best_insertion(route_to, cid, instance)
        if pos is not None and delta < best_delta:
            best_delta = delta
            best_pos = pos
            best_r = r_to

    if best_r is None:
        return None

    # Apply move
    new_routes = [list(r) for r in routes]
    new_routes[r_from].pop(pos_from)
    new_routes[best_r].insert(best_pos, cid)

    # Validate the source route (it may now be empty or infeasible due to TW shift)
    if new_routes[r_from] and not _route_feasible_fast(new_routes[r_from], instance):
        return None

    # Remove empty routes
    new_routes = [r for r in new_routes if r]
    return new_routes


def move_2opt_star(routes, instance, rng):
    """
    Inter-route 2-opt*: swap the tails of two different routes.

    Pick two routes r1, r2, and cut points i, j.
    New routes: r1[:i+1] + r2[j:] and r2[:j] + r1[i+1:]
    Accept only if both resulting routes are feasible.
    """
    if len(routes) < 2:
        return None

    n_routes = len(routes)
    r1_idx = rng.randrange(n_routes)
    r2_idx = rng.randrange(n_routes)
    if r1_idx == r2_idx:
        return None

    r1 = routes[r1_idx]
    r2 = routes[r2_idx]

    if not r1 or not r2:
        return None

    # Pick random cut points
    i = rng.randrange(len(r1))
    j = rng.randrange(len(r2))

    new_r1 = r1[:i + 1] + r2[j:]
    new_r2 = r2[:j] + r1[i + 1:]

    # Check capacity
    cap = instance["vehicle_capacity"]
    customers = instance["_customers"]
    if new_r1 and sum(customers[c]["demand"] for c in new_r1) > cap:
        return None
    if new_r2 and sum(customers[c]["demand"] for c in new_r2) > cap:
        return None

    # Check TW feasibility
    if new_r1 and not _route_feasible_fast(new_r1, instance):
        return None
    if new_r2 and not _route_feasible_fast(new_r2, instance):
        return None

    new_routes = [list(r) for r in routes]
    if new_r1:
        new_routes[r1_idx] = new_r1
    else:
        new_routes[r1_idx] = []
    if new_r2:
        new_routes[r2_idx] = new_r2
    else:
        new_routes[r2_idx] = []

    new_routes = [r for r in new_routes if r]
    return new_routes


def move_or_opt(routes, instance, rng, chain_length=1):
    """
    Or-opt: move a chain of `chain_length` consecutive customers from
    one route to the best feasible position in any other route.

    chain_length=1: single customer relocation (stronger version of relocate)
    chain_length=2,3: move pairs/triples
    """
    if len(routes) < 2:
        return None

    n_routes = len(routes)
    r_from_idx = rng.randrange(n_routes)
    r_from = routes[r_from_idx]

    if len(r_from) < chain_length:
        return None

    # Pick random start of chain
    start = rng.randrange(len(r_from) - chain_length + 1)
    chain = r_from[start:start + chain_length]

    # Remove chain from source route
    new_r_from = r_from[:start] + r_from[start + chain_length:]
    if new_r_from and not _route_feasible_fast(new_r_from, instance):
        return None

    # Find best insertion of the chain in another route
    cap = instance["vehicle_capacity"]
    customers = instance["_customers"]
    chain_demand = sum(customers[c]["demand"] for c in chain)

    best_r_to = None
    best_pos = None
    best_cost = float("inf")

    # Try a subset of candidate routes
    candidate_idxs = [i for i in range(n_routes) if i != r_from_idx]
    rng.shuffle(candidate_idxs)
    candidate_idxs = candidate_idxs[:min(len(candidate_idxs), 8)]

    for r_to_idx in candidate_idxs:
        r_to = routes[r_to_idx]
        if route_load(r_to, instance) + chain_demand > cap:
            continue
        # Try inserting the whole chain at each position
        for pos in range(len(r_to) + 1):
            candidate = r_to[:pos] + chain + r_to[pos:]
            if _route_feasible_fast(candidate, instance):
                # Use travel time delta as cost
                cost = route_travel_time(candidate, instance) - route_travel_time(r_to, instance)
                if cost < best_cost:
                    best_cost = cost
                    best_r_to = r_to_idx
                    best_pos = pos

    if best_r_to is None:
        return None

    new_routes = [list(r) for r in routes]
    new_routes[r_from_idx] = new_r_from
    r_to = new_routes[best_r_to]
    new_routes[best_r_to] = r_to[:best_pos] + chain + r_to[best_pos:]
    new_routes = [r for r in new_routes if r]
    return new_routes


def move_intra_2opt(routes, instance, rng):
    """
    Intra-route 2-opt: reverse a segment within a single route.
    Improves route sequencing without changing vehicle assignments.
    """
    if not routes:
        return None

    r_idx = rng.randrange(len(routes))
    route = routes[r_idx]

    if len(route) < 3:
        return None

    i = rng.randrange(len(route) - 1)
    j = rng.randrange(i + 1, min(i + 7, len(route)))  # limit window for speed

    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
    if not _route_feasible_fast(new_route, instance):
        return None

    new_routes = [list(r) for r in routes]
    new_routes[r_idx] = new_route
    return new_routes


def move_route_merge(routes, instance, rng):
    """
    Attempt to merge the smallest route into any other route.
    High-value move: if successful, eliminates a vehicle (saves 1000 points).
    """
    if len(routes) < 2:
        return None

    # Target the smallest route
    target_idx = min(range(len(routes)), key=lambda i: len(routes[i]))
    target = routes[target_idx]

    # Try appending to / prepending from each other route
    cap = instance["vehicle_capacity"]
    customers = instance["_customers"]
    target_demand = sum(customers[c]["demand"] for c in target)

    candidate_idxs = [i for i in range(len(routes)) if i != target_idx]
    rng.shuffle(candidate_idxs)

    for r_idx in candidate_idxs:
        other = routes[r_idx]
        if route_load(other, instance) + target_demand > cap:
            continue
        # Try inserting each customer from target into best position in other
        temp = list(other)
        success = True
        for cid in target:
            pos, _ = best_insertion(temp, cid, instance)
            if pos is None:
                success = False
                break
            temp.insert(pos, cid)
        if success and _route_feasible_fast(temp, instance):
            new_routes = [list(r) for r in routes]
            new_routes[r_idx] = temp
            new_routes.pop(target_idx)
            return new_routes

    return None


# ---------------------------------------------------------------------------
# Temperature calibration
# ---------------------------------------------------------------------------

def calibrate_temperature(routes, instance, rng, n_samples=300,
                          target_accept_prob=0.8):
    """
    Estimate a good starting temperature T0 by sampling random moves
    and measuring the average cost of worsening moves.

    T0 = -avg_delta / ln(target_accept_prob)

    This ensures P(accept average worsening) ≈ target_accept_prob at start.
    """
    move_funcs = [
        move_relocate,
        move_2opt_star,
        lambda r, i, rng: move_or_opt(r, i, rng, 1),
        lambda r, i, rng: move_or_opt(r, i, rng, 2),
        move_intra_2opt,
    ]

    current_cost = solution_travel_time(routes, instance)
    deltas = []

    for _ in range(n_samples):
        move_fn = rng.choice(move_funcs)
        new_routes = move_fn(copy.deepcopy(routes), instance, rng)
        if new_routes is None:
            continue
        new_cost = solution_travel_time(new_routes, instance)
        delta = new_cost - current_cost
        if delta > 0:  # only worsening moves
            deltas.append(delta)

    if not deltas:
        return 100.0  # fallback

    avg_delta = sum(deltas) / len(deltas)
    T0 = -avg_delta / math.log(target_accept_prob)
    return max(T0, 1.0)


# ---------------------------------------------------------------------------
# Main SA loop
# ---------------------------------------------------------------------------

def sa_solve(instance,
             initial_routes=None,
             T_initial=None,
             T_final=0.01,
             cooling_rate=0.9995,
             max_iterations=500_000,
             restart_after=50_000,
             move_weights=None,
             seed=42,
             verbose=True,
             callback=None):
    """
    Simulated Annealing for S-CVRPTW.

    Parameters:
        instance        : loaded + cached instance dict
        initial_routes  : starting solution (None → use Clarke-Wright)
        T_initial       : starting temperature (None → auto-calibrate)
        T_final         : stop temperature
        cooling_rate    : geometric cooling α (T ← T × α each iteration)
        max_iterations  : total SA iterations
        restart_after   : iterations without improvement before restart
        move_weights    : dict of move_name → weight (optional)
        seed            : random seed for reproducibility
        verbose         : print progress every 50k iterations
        callback        : callable(iter, best_score, T) for external logging

    Returns:
        best_routes (list of routes)
    """
    build_cache(instance)
    rng = random.Random(seed)

    # --- Initial solution ---
    if initial_routes is None:
        if verbose:
            print("  SA: building initial solution (Clarke-Wright λ=0.85)...")
        routes = clarke_wright(instance, lambda_param=0.85)
    else:
        routes = copy.deepcopy(initial_routes)

    # --- Move table ---
    # (function, weight, name)
    move_table = [
        (move_relocate,                                        0.30, "relocate"),
        (move_2opt_star,                                       0.20, "2opt*"),
        (lambda r, i, rng: move_or_opt(r, i, rng, 1),        0.20, "or-opt-1"),
        (lambda r, i, rng: move_or_opt(r, i, rng, 2),        0.10, "or-opt-2"),
        (lambda r, i, rng: move_or_opt(r, i, rng, 3),        0.05, "or-opt-3"),
        (move_intra_2opt,                                      0.10, "intra-2opt"),
        (move_route_merge,                                     0.05, "merge"),
    ]
    if move_weights:
        for entry in move_table:
            if entry[2] in move_weights:
                entry = (entry[0], move_weights[entry[2]], entry[2])

    move_fns = [e[0] for e in move_table]
    weights = [e[1] for e in move_table]

    # --- Temperature calibration ---
    if T_initial is None:
        if verbose:
            print("  SA: calibrating temperature...")
        T_initial = calibrate_temperature(routes, instance, rng)
        if verbose:
            print(f"  SA: T0 = {T_initial:.2f}")

    T = T_initial
    current_cost = solution_travel_time(routes, instance)
    # Track the true proxy score (includes vehicle count penalty)
    current_proxy = solution_score_proxy(routes, instance)

    best_routes = copy.deepcopy(routes)
    best_proxy = current_proxy
    no_improve = 0
    t_start = time.time()

    if verbose:
        print(f"  SA: start | {solution_summary(routes, instance)} | T={T:.2f}")

    for iteration in range(1, max_iterations + 1):
        # Select and apply move
        move_fn = rng.choices(move_fns, weights=weights)[0]
        new_routes = move_fn(copy.deepcopy(routes), instance, rng)

        if new_routes is None:
            no_improve += 1
            continue

        new_cost = solution_travel_time(new_routes, instance)
        new_proxy = solution_score_proxy(new_routes, instance)
        delta = new_proxy - current_proxy

        # SA acceptance: always accept improvements, probabilistically accept worse
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            routes = new_routes
            current_cost = new_cost
            current_proxy = new_proxy

            if current_proxy < best_proxy:
                best_proxy = current_proxy
                best_routes = copy.deepcopy(routes)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # Cool down
        T = max(T * cooling_rate, T_final)

        # Restart on stagnation: reset to best known, reheat
        if no_improve >= restart_after:
            routes = copy.deepcopy(best_routes)
            current_proxy = best_proxy
            current_cost = solution_travel_time(routes, instance)
            T = T_initial * 0.5
            no_improve = 0
            if verbose:
                elapsed = time.time() - t_start
                print(f"  SA: restart @iter {iteration:,} | "
                      f"{solution_summary(best_routes, instance)} | "
                      f"T={T:.2f} | {elapsed:.0f}s")

        # Logging
        if callback and iteration % 1000 == 0:
            callback(iteration, best_proxy, T)

        if verbose and iteration % 50_000 == 0:
            elapsed = time.time() - t_start
            print(f"  SA @{iteration:,} | "
                  f"{solution_summary(best_routes, instance)} | "
                  f"T={T:.4f} | {elapsed:.0f}s")

    if verbose:
        elapsed = time.time() - t_start
        print(f"  SA: done in {elapsed:.1f}s | {solution_summary(best_routes, instance)}")

    return best_routes


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys

    path = sys.argv[1] if len(sys.argv) > 1 else "istanbul_small_100.json"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000

    with open(path) as f:
        instance = json.load(f)
    build_cache(instance)

    print(f"Instance: {instance['name']} ({instance['n_customers']} customers)")
    print(f"Running SA for {iters:,} iterations...")
    print()

    routes = sa_solve(instance, max_iterations=iters, verbose=True)

    from evaluate_solution import validate_solution, score_solution
    sol = {"team": "SA", "routes": routes}
    valid, errors = validate_solution(instance, sol)
    print()
    if valid:
        print("Validation: PASSED")
        result = score_solution(instance, sol)
        print(f"Vehicles:      {result['n_vehicles']}")
        print(f"Avg TT:        {result['avg_travel_time']:.2f}")
        print(f"Avg violations:{result['avg_tw_violations']:.2f}")
        print(f"SCORE:         {result['score']:.2f}  (baseline: 25,127)")
    else:
        print("Validation: FAILED")
        for e in errors:
            print(f"  - {e}")