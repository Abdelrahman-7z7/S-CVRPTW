"""
CSE 557 — Route Eliminator (Post-Processor)
Removes vehicles by redistributing customers from the smallest routes
into other existing routes. Each elimination saves exactly 1000 score points.

Run this after ANY algorithm has converged as a final squeeze pass.
"""

import copy
import time

from solver_core import (
    build_cache, best_insertion, is_route_feasible, route_load,
    solution_score_proxy, solution_summary, assert_valid
)


# ---------------------------------------------------------------------------
# Core elimination loop
# ---------------------------------------------------------------------------

def eliminate_routes(routes, instance, max_rounds=None, verbose=True):
    """
    Repeatedly attempt to eliminate the route with the fewest customers
    by inserting each of its customers into other routes.

    Strategy per attempt:
      1. Pick the smallest route (fewest customers).
      2. For each customer in that route, find the best feasible insertion
         across all OTHER routes.
      3. If ALL customers can be relocated → commit the elimination.
      4. Else → try the next smallest route.
      5. Repeat until no route can be eliminated.

    Parameters:
        routes     : current route set (not modified in-place — copy returned)
        instance   : loaded + cached instance
        max_rounds : max elimination attempts (None = unlimited)
        verbose    : print progress

    Returns:
        (improved_routes, n_eliminated)
    """
    build_cache(instance)
    routes = copy.deepcopy(routes)
    n_eliminated = 0
    max_rounds = max_rounds or len(routes) * 2

    for attempt in range(max_rounds):
        if len(routes) <= 1:
            break

        # Sort routes by size ascending, try smallest first
        sorted_idxs = sorted(range(len(routes)), key=lambda i: len(routes[i]))

        eliminated_this_round = False

        for target_idx in sorted_idxs:
            target_route = routes[target_idx]

            # Build working copy without target
            temp_routes = [list(r) for i, r in enumerate(routes)
                           if i != target_idx]
            success = True
            insertion_log = []  # (r_idx, pos, cid) to commit if success

            for cid in target_route:
                best_r = None
                best_pos = None
                best_delta = float("inf")

                for r_idx, route in enumerate(temp_routes):
                    pos, delta = best_insertion(route, cid, instance)
                    if pos is not None and delta < best_delta:
                        best_delta = delta
                        best_pos = pos
                        best_r = r_idx

                if best_r is None:
                    success = False
                    break

                # Tentatively insert (so subsequent customers see updated route)
                temp_routes[best_r].insert(best_pos, cid)
                insertion_log.append((best_r, best_pos, cid))

            if success:
                # Validate all modified routes
                modified_idxs = {log[0] for log in insertion_log}
                all_feasible = all(
                    is_route_feasible(temp_routes[i], instance)
                    for i in modified_idxs
                )

                if all_feasible:
                    routes = temp_routes
                    n_eliminated += 1
                    eliminated_this_round = True

                    if verbose:
                        print(f"  Eliminator: route removed! "
                              f"Now {len(routes)} vehicles "
                              f"(proxy={solution_score_proxy(routes, instance):.1f})")
                    break  # restart outer loop with updated routes

        if not eliminated_this_round:
            break  # no route could be eliminated

    return routes, n_eliminated


# ---------------------------------------------------------------------------
# Elimination with SA repair pass
# ---------------------------------------------------------------------------

def elimination_with_repair(routes, instance, sa_repair_iters=2_000,
                             verbose=True):
    """
    Run route elimination, then apply a short SA refinement pass to
    re-optimise the routes that absorbed the extra customers.

    The SA repair often finds better insertion positions than the greedy
    elimination pass, and may enable a further elimination in rare cases.

    Returns (improved_routes, n_eliminated)
    """
    routes, n_elim = eliminate_routes(routes, instance, verbose=verbose)

    if n_elim > 0 and sa_repair_iters > 0:
        if verbose:
            print(f"  Eliminator: running SA repair "
                  f"({sa_repair_iters:,} iters)...")
        from sa_solver import sa_solve
        routes = sa_solve(
            instance,
            initial_routes=routes,
            max_iterations=sa_repair_iters,
            verbose=False
        )
        # Try one more elimination after repair
        routes, n_elim2 = eliminate_routes(routes, instance, verbose=verbose)
        n_elim += n_elim2

    return routes, n_elim


# ---------------------------------------------------------------------------
# Aggressive elimination: iterate eliminate → SA → eliminate until stuck
# ---------------------------------------------------------------------------

def aggressive_elimination(routes, instance, rounds=3,
                            sa_iters_per_round=5_000,
                            verbose=True):
    """
    Alternate between elimination and SA repair for `rounds` cycles.
    Each SA round gives the search a chance to restructure routes so
    that previously un-eliminatable routes become eliminatable.

    Parameters:
        rounds            : number of eliminate-repair cycles
        sa_iters_per_round: SA iterations between eliminations

    Returns (best_routes, total_eliminated)
    """
    build_cache(instance)
    total_elim = 0
    t_start = time.time()

    for rnd in range(1, rounds + 1):
        if verbose:
            print(f"\n  [Aggressive Elim] Round {rnd}/{rounds} "
                  f"— {len(routes)} vehicles")

        # Elimination pass
        routes, n = eliminate_routes(routes, instance, verbose=verbose)
        total_elim += n

        if sa_iters_per_round > 0 and rnd < rounds:
            # SA repair between rounds
            if verbose:
                print(f"  [Aggressive Elim] SA repair ({sa_iters_per_round:,} iters)...")
            from sa_solver import sa_solve
            routes = sa_solve(
                instance,
                initial_routes=routes,
                max_iterations=sa_iters_per_round,
                verbose=False
            )

    if verbose:
        elapsed = time.time() - t_start
        print(f"\n  [Aggressive Elim] Done: {total_elim} vehicles eliminated "
              f"in {elapsed:.1f}s")
        print(f"  {solution_summary(routes, instance)}")

    return routes, total_elim


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys
    from evaluate_solution import validate_solution, score_solution

    path = sys.argv[1] if len(sys.argv) > 1 else "istanbul_small_100.json"
    sol_path = sys.argv[2] if len(sys.argv) > 2 else "baseline_istanbul_small_100.json"

    with open(path) as f:
        instance = json.load(f)
    with open(sol_path) as f:
        sol = json.load(f)

    build_cache(instance)

    print(f"Instance: {instance['name']} ({instance['n_customers']} customers)")
    print(f"Input: {sol['team']} | {len(sol['routes'])} vehicles")
    print()

    routes, n_elim = elimination_with_repair(
        sol["routes"], instance, sa_repair_iters=30_000, verbose=True
    )

    solution = {"team": "Post-Eliminator", "routes": routes}
    valid, errors = validate_solution(instance, solution)
    print()
    if valid:
        result = score_solution(instance, solution)
        print(f"Eliminated: {n_elim} routes")
        print(f"Vehicles:      {result['n_vehicles']}")
        print(f"Avg TT:        {result['avg_travel_time']:.2f}")
        print(f"Avg violations:{result['avg_tw_violations']:.2f}")
        print(f"SCORE:         {result['score']:.2f}")
    else:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")