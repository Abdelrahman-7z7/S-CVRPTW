"""
CSE 557 — Construction Phase
Clarke-Wright Savings Algorithm for S-CVRPTW initial solution generation.

Clarke-Wright directly targets fewer vehicles (merges singleton routes).
It outperforms nearest-neighbour construction on vehicle count.

Reference: Clarke & Wright (1964). "Scheduling of vehicles from a central
depot to a number of delivery points." Operations Research, 12(4), 568–581.
"""

import copy
from solver_core import (
    build_cache, is_route_feasible, route_load,
    solution_summary, assert_valid, all_customer_ids
)


# ---------------------------------------------------------------------------
# Savings computation
# ---------------------------------------------------------------------------

def compute_savings(instance, lambda_param=1.0):
    """
    Compute Clarke-Wright savings for all customer pairs (i, j), i < j.

    S(i, j) = d(depot, i) + d(depot, j) - lambda * d(i, j)

    Higher savings → merging routes ending at i and starting at j saves more
    distance compared to serving them with two separate depot round-trips.

    lambda_param > 1.0: favours merging closer customers (tighter clusters).
    lambda_param < 1.0: favours merging customers far from depot (long routes).
    lambda_param = 1.0: standard Clarke-Wright formula.

    Returns list of (savings, i, j) sorted descending.
    """
    n = instance["n_customers"]
    dist = instance["distance_matrix"]

    savings = []
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue
            s = dist[0][i] + dist[0][j] - lambda_param * dist[i][j]
            savings.append((s, i, j))

    savings.sort(key=lambda x: -x[0])
    return savings


# ---------------------------------------------------------------------------
# Merge feasibility
# ---------------------------------------------------------------------------

def can_merge(route_a, route_b, instance):
    """
    Check if route_b can be appended to route_a to form a valid single route.

    For Clarke-Wright: route_a ends at some customer i (last element),
    route_b starts at some customer j (first element). We check that:
      - Combined capacity ≤ vehicle_capacity
      - The merged route satisfies all time windows and depot return

    Returns True if merge is feasible, False otherwise.
    """
    merged = route_a + route_b
    cap = instance["vehicle_capacity"]
    customers = instance["_customers"]

    load = sum(customers[cid]["demand"] for cid in merged)
    if load > cap:
        return False

    return is_route_feasible(merged, instance)


# ---------------------------------------------------------------------------
# Main Clarke-Wright solver
# ---------------------------------------------------------------------------

def clarke_wright(instance, lambda_param=1.0, verbose=False):
    """
    Build an initial solution using the Clarke-Wright Savings algorithm.

    Algorithm:
      1. Start: N singleton routes, one per customer (N vehicles).
      2. Compute savings S(i, j) for all pairs, sort descending.
      3. For each (i, j) in savings order:
           - If i is the LAST customer in its route and
             j is the FIRST customer in its route and
             they are on DIFFERENT routes:
               → Try merging. Accept if feasible (capacity + TW).
      4. Return merged routes.

    Parameters:
        lambda_param: savings formula weight (see compute_savings).
        verbose: print merge progress.

    Returns:
        List of routes (each route is a list of customer IDs).
    """
    build_cache(instance)
    n = instance["n_customers"]
    depot = instance["depot"]
    dist = instance["distance_matrix"]
    traffic = instance["traffic"]

    # ------------------------------------------------------------------
    # Pre-check: identify customers that are unreachable as the FIRST
    # stop on a route (depot → customer violates their time window).
    # These cannot form valid singleton routes, so we must ensure the
    # savings merging phase always combines them with a predecessor.
    # We flag them here; if they remain as singletons after the merge
    # phase we will forcibly insert them into the best feasible route.
    # ------------------------------------------------------------------
    from evaluate_solution import deterministic_travel_time as _dtt
    unreachable_as_first = set()
    for cid in range(1, n + 1):
        c = instance["_customers"][cid]
        d = dist[0][cid]
        tt = _dtt(d, depot["tw_open"], traffic)
        arr = depot["tw_open"] + tt
        if arr < c["tw_open"]:
            arr = c["tw_open"]
        if arr > c["tw_close"]:
            unreachable_as_first.add(cid)

    if unreachable_as_first and verbose:
        print(f"  CW: {len(unreachable_as_first)} customers unreachable "
              f"as first stop: {sorted(unreachable_as_first)}")

    # Step 1: initialise one singleton route per customer
    # route_of[cid] = index into routes list
    routes = [[cid] for cid in range(1, n + 1)]
    route_of = {cid: cid - 1 for cid in range(1, n + 1)}

    # We track route membership via route_of, and identify
    # endpoints directly from routes[r_idx][0] and routes[r_idx][-1]

    # Step 2: compute savings
    savings = compute_savings(instance, lambda_param)

    # Step 3: merge
    merges = 0
    for saving, i, j in savings:
        if saving <= 0:
            break  # no benefit in merging further

        r_i = route_of.get(i)
        r_j = route_of.get(j)

        if r_i is None or r_j is None:
            continue
        if r_i == r_j:
            continue  # already on the same route

        route_a = routes[r_i]
        route_b = routes[r_j]

        # i must be last in route_a, j must be first in route_b
        if route_a[-1] != i or route_b[0] != j:
            continue

        # Try the merge
        if not can_merge(route_a, route_b, instance):
            continue

        # Perform merge: append route_b to route_a, delete route_b
        merged = route_a + route_b
        routes[r_i] = merged
        routes[r_j] = None  # mark as consumed

        # Update route_of for all customers that moved from r_j → r_i
        for cid in route_b:
            route_of[cid] = r_i

        merges += 1

    # Step 4: collect non-None routes
    result = [r for r in routes if r is not None]

    # ------------------------------------------------------------------
    # Step 5: fix any customers that are still isolated on invalid
    # singleton routes (unreachable-as-first-stop).
    # Force-insert them into every possible position in every existing
    # route using the full feasibility simulator.
    # ------------------------------------------------------------------
    still_invalid = [
        r[0] for r in result
        if len(r) == 1 and r[0] in unreachable_as_first
    ]

    if still_invalid:
        if verbose:
            print(f"  CW: force-inserting {len(still_invalid)} "
                  f"unreachable singletons: {still_invalid}")

        # Remove their singleton routes from result
        result = [r for r in result
                  if not (len(r) == 1 and r[0] in unreachable_as_first)]

        from solver_core import best_insertion
        for cid in still_invalid:
            best_r = None
            best_pos = None
            best_delta = float("inf")

            # Try EVERY position (including 0) in EVERY route
            for r_idx, route in enumerate(result):
                pos, delta = best_insertion(route, cid, instance)
                if pos is not None and delta < best_delta:
                    best_delta = delta
                    best_r = r_idx
                    best_pos = pos

            if best_r is not None:
                result[best_r].insert(best_pos, cid)
                if verbose:
                    print(f"    Inserted cust {cid} at pos {best_pos} "
                          f"in route {best_r} (delta={best_delta:.1f})")
            else:
                # Truly cannot fit anywhere — open singleton (will fail
                # validation, but signals instance issue to the user)
                result.append([cid])
                if verbose:
                    print(f"    WARNING: cust {cid} is infeasible in this "
                          f"instance — no valid position exists in any route")

    if verbose:
        print(f"  Clarke-Wright (λ={lambda_param:.2f}): "
              f"{n} customers → {len(result)} routes "
              f"({merges} merges)")

    return result


# ---------------------------------------------------------------------------
# Population generator for MA / ACO seeding
# ---------------------------------------------------------------------------

def clarke_wright_population(instance, size=20, lambda_range=(0.8, 1.2),
                              verbose=False):
    """
    Generate a diverse population by running Clarke-Wright with different
    lambda values uniformly distributed over lambda_range.

    Returns a list of `size` route sets. Different lambdas produce
    structurally different solutions, giving MA/ACO a varied starting pool.
    """
    build_cache(instance)

    if size == 1:
        lambdas = [1.0]
    else:
        lo, hi = lambda_range
        lambdas = [lo + i * (hi - lo) / (size - 1) for i in range(size)]

    population = []
    for lam in lambdas:
        routes = clarke_wright(instance, lambda_param=lam, verbose=verbose)
        population.append(routes)

    if verbose:
        vehicle_counts = [len(r) for r in population]
        print(f"  Population: {size} solutions | "
              f"Vehicles: min={min(vehicle_counts)} "
              f"avg={sum(vehicle_counts)/len(vehicle_counts):.1f} "
              f"max={max(vehicle_counts)}")

    return population


# ---------------------------------------------------------------------------
# Greedy repair: insert unrouted customers into existing routes
# ---------------------------------------------------------------------------

def greedy_insert_unrouted(routes, unrouted, instance):
    """
    Insert each customer in `unrouted` into the best feasible position
    across all routes. If no route can accommodate a customer, open a
    new singleton route for them.

    Used as a safety net after crossover operations that may strand customers.

    Returns updated routes (modifies in-place and returns).
    """
    from solver_core import best_insertion_across_routes

    for cid in sorted(unrouted):
        r_idx, pos, delta = best_insertion_across_routes(routes, cid, instance)
        if r_idx is not None:
            routes[r_idx].insert(pos, cid)
        else:
            # Cannot fit anywhere — open a new route
            routes.append([cid])

    return routes


# ---------------------------------------------------------------------------
# Quick test / standalone run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    import time

    path = sys.argv[1] if len(sys.argv) > 1 else "istanbul_small_100.json"
    with open(path) as f:
        instance = json.load(f)

    build_cache(instance)

    print(f"Instance: {instance['name']} ({instance['n_customers']} customers)")
    print()

    # Test a few lambda values
    for lam in [0.8, 0.9, 1.0, 1.1, 1.2]:
        t0 = time.time()
        routes = clarke_wright(instance, lambda_param=lam)
        elapsed = time.time() - t0

        ok, errors = __import__("evaluate_solution").validate_solution(
            instance, {"team": "CW", "routes": routes}
        )
        status = "OK" if ok else f"INVALID ({errors[0]})"
        from solver_core import solution_score_proxy
        print(f"  λ={lam:.1f} | Vehicles={len(routes):2d} | "
              f"Proxy={solution_score_proxy(routes, instance):.1f} | "
              f"Time={elapsed:.3f}s | {status}")

    print()
    print("Best construction (λ=1.0):")
    routes = clarke_wright(instance, lambda_param=1.0)
    from solver_core import solution_summary
    print(" ", solution_summary(routes, instance))

    print()
    print("Population (size=5, λ=0.8..1.2):")
    pop = clarke_wright_population(instance, size=5, verbose=True)