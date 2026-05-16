"""
CSE 557 — Tabu Search Solver
Memory-based single-solution search. Forbidden moves (tabu) prevent cycling.
Aspiration criterion overrides tabu if a move finds a new global best.

Key design decisions:
  - Tabu attribute: (customer_id, origin_route_fingerprint) — forbids
    returning a customer to its origin route for `tenure` iterations.
  - Candidate list: restrict neighbourhood to nearest-K customers to keep
    O(N) per iteration instead of O(N²).
  - Diversification: when stagnating, extend tenure to push into new regions.
  - Intensification: when new best found, run a burst of exploitation.
"""

import copy
import random
import time

from solver_core import (
    build_cache, is_route_feasible, route_load,
    best_insertion, route_travel_time, solution_travel_time,
    solution_score_proxy, get_neighbors, solution_summary
)
from construction import clarke_wright


# ---------------------------------------------------------------------------
# Tabu List
# ---------------------------------------------------------------------------

class TabuList:
    """
    Stores tabu attributes as (customer_id, origin_route_id) pairs.
    A 'route_id' is the index of the route in the current solution — it's
    approximate since routes shift after each move, but works well in practice
    because we only need to prevent re-insertion into the exact same route.

    tenure is randomised each time to prevent periodic cycling.
    """

    def __init__(self, tenure_min=6, tenure_max=12, rng=None):
        self.tenure_min = tenure_min
        self.tenure_max = tenure_max
        self.rng = rng or random.Random(42)
        self._entries = {}  # (cid, route_fingerprint) → expiry_iteration

    def add(self, customer_id, route_fingerprint, current_iter):
        tenure = self.rng.randint(self.tenure_min, self.tenure_max)
        self._entries[(customer_id, route_fingerprint)] = current_iter + tenure

    def is_tabu(self, customer_id, route_fingerprint, current_iter):
        key = (customer_id, route_fingerprint)
        return self._entries.get(key, 0) > current_iter

    def cleanup(self, current_iter):
        """Remove expired entries to keep memory bounded."""
        self._entries = {
            k: v for k, v in self._entries.items() if v > current_iter
        }

    def extend_tenure(self, amount=3):
        """Diversification: push forbidden moves further into the future."""
        self.tenure_min = min(self.tenure_min + amount, 30)
        self.tenure_max = min(self.tenure_max + amount, 40)

    def reset_tenure(self, tenure_min=6, tenure_max=12):
        self.tenure_min = tenure_min
        self.tenure_max = tenure_max


# ---------------------------------------------------------------------------
# Route fingerprint — lightweight ID for a route by its sorted contents
# ---------------------------------------------------------------------------

def route_fingerprint(route):
    """
    A lightweight, hashable identifier for a route.
    We use the sorted tuple of customer IDs — this is stable even as
    route order changes due to insertions.
    """
    return tuple(sorted(route))


# ---------------------------------------------------------------------------
# Candidate move generation
# ---------------------------------------------------------------------------

def generate_relocate_candidates(routes, tabu_list, instance,
                                  current_iter, best_ever_proxy,
                                  n_nearest=15, max_candidates=40):
    """
    Generate and evaluate candidate relocate moves using the nearest-
    neighbour candidate list to restrict search to O(N * K) instead of O(N²).

    For each customer `cid`:
      - Find its K nearest neighbours.
      - For each neighbour's route, try inserting `cid` there.
      - Skip if tabu UNLESS aspiration criterion met (beats best_ever).

    Returns list of (proxy_score, new_routes, tabu_attr) sorted ascending.
    Limits to max_candidates for efficiency.
    """
    candidates = []
    cap = instance["vehicle_capacity"]
    customers = instance["_customers"]

    # Build a fast lookup: customer_id → route_index
    cid_to_route = {}
    for r_idx, route in enumerate(routes):
        for cid in route:
            cid_to_route[cid] = r_idx

    n = instance["n_customers"]
    all_cids = list(range(1, n + 1))

    # Sample a subset of customers to evaluate (full scan too slow at N=500)
    if len(all_cids) > 50:
        sample_cids = [all_cids[i] for i in
                       sorted(random.sample(range(len(all_cids)),
                                            min(30, len(all_cids))))]
    else:
        sample_cids = all_cids

    for cid in sample_cids:
        r_from_idx = cid_to_route.get(cid)
        if r_from_idx is None:
            continue

        r_from = routes[r_from_idx]
        fp_from = route_fingerprint(r_from)
        tabu_attr = (cid, fp_from)

        is_tabu_move = tabu_list.is_tabu(cid, fp_from, current_iter)

        # Find candidate destination routes via nearest neighbours
        neighbors = get_neighbors(cid, instance, n_nearest)
        candidate_route_idxs = set()
        for nb in neighbors:
            nb_route = cid_to_route.get(nb)
            if nb_route is not None and nb_route != r_from_idx:
                candidate_route_idxs.add(nb_route)
            if len(candidate_route_idxs) >= 6:
                break

        for r_to_idx in candidate_route_idxs:
            r_to = routes[r_to_idx]

            # Capacity check
            if (route_load(r_to, instance) +
                    customers[cid]["demand"] > cap):
                continue

            # Build new solution
            pos, delta = best_insertion(r_to, cid, instance)
            if pos is None:
                continue

            new_routes = [list(r) for r in routes]
            pos_in_from = new_routes[r_from_idx].index(cid)
            new_routes[r_from_idx].pop(pos_in_from)

            # Validate source route still feasible after removal
            if new_routes[r_from_idx]:
                if not is_route_feasible(new_routes[r_from_idx], instance):
                    continue

            new_routes[r_to_idx].insert(pos, cid)
            new_routes = [r for r in new_routes if r]

            proxy = solution_score_proxy(new_routes, instance)

            # Accept: not tabu, OR aspiration (new global best)
            if not is_tabu_move or proxy < best_ever_proxy:
                candidates.append((proxy, new_routes, tabu_attr))

            if len(candidates) >= max_candidates:
                candidates.sort(key=lambda x: x[0])
                return candidates

    candidates.sort(key=lambda x: x[0])
    return candidates


def generate_or_opt_candidates(routes, tabu_list, instance,
                                current_iter, best_ever_proxy,
                                n_nearest=10, chain_length=2):
    """
    Generate Or-opt moves (move chain of `chain_length` customers).
    Uses same candidate-list approach as relocate.

    Returns list of (proxy_score, new_routes, tabu_attr) sorted ascending.
    """
    candidates = []
    cap = instance["vehicle_capacity"]
    customers = instance["_customers"]

    cid_to_route = {}
    for r_idx, route in enumerate(routes):
        for cid in route:
            cid_to_route[cid] = r_idx

    # Sample source routes
    n_routes = len(routes)
    sample_routes = list(range(n_routes))
    if n_routes > 10:
        sample_routes = random.sample(sample_routes, min(8, n_routes))

    for r_from_idx in sample_routes:
        r_from = routes[r_from_idx]
        if len(r_from) < chain_length:
            continue

        # Sample a start position
        start = random.randrange(len(r_from) - chain_length + 1)
        chain = r_from[start:start + chain_length]
        fp_from = route_fingerprint(r_from)
        tabu_attr = (chain[0], fp_from)

        is_tabu_move = tabu_list.is_tabu(chain[0], fp_from, current_iter)

        chain_demand = sum(customers[c]["demand"] for c in chain)

        # Build new source route
        new_r_from = r_from[:start] + r_from[start + chain_length:]
        if new_r_from and not is_route_feasible(new_r_from, instance):
            continue

        # Find destinations via first customer's neighbours
        neighbors = get_neighbors(chain[0], instance, n_nearest)
        dest_idxs = set()
        for nb in neighbors:
            nb_r = cid_to_route.get(nb)
            if nb_r is not None and nb_r != r_from_idx:
                dest_idxs.add(nb_r)
            if len(dest_idxs) >= 5:
                break

        for r_to_idx in dest_idxs:
            r_to = routes[r_to_idx]
            if route_load(r_to, instance) + chain_demand > cap:
                continue

            for pos in range(len(r_to) + 1):
                candidate_r_to = r_to[:pos] + chain + r_to[pos:]
                if not is_route_feasible(candidate_r_to, instance):
                    continue

                new_routes = [list(r) for r in routes]
                new_routes[r_from_idx] = new_r_from
                new_routes[r_to_idx] = candidate_r_to
                new_routes = [r for r in new_routes if r]
                proxy = solution_score_proxy(new_routes, instance)

                if not is_tabu_move or proxy < best_ever_proxy:
                    candidates.append((proxy, new_routes, tabu_attr))
                break  # take first feasible position per destination

    candidates.sort(key=lambda x: x[0])
    return candidates[:20]


# ---------------------------------------------------------------------------
# Main TS loop
# ---------------------------------------------------------------------------

def ts_solve(instance,
             initial_routes=None,
             max_iterations=10_000,
             tenure_min=6,
             tenure_max=12,
             n_nearest=15,
             diversify_after=1_000,
             intensify_burst=200,
             seed=42,
             verbose=True,
             callback=None):
    """
    Tabu Search for S-CVRPTW.

    Parameters:
        instance         : loaded + cached instance dict
        initial_routes   : starting solution (None → Clarke-Wright)
        max_iterations   : total TS iterations
        tenure_min/max   : tabu tenure range (randomised each add)
        n_nearest        : candidate list size per customer
        diversify_after  : no-improvement iterations before diversification
        intensify_burst  : iterations of pure exploitation after new best
        seed             : random seed
        verbose          : print progress
        callback         : callable(iter, best_proxy) for external logging

    Returns:
        best_routes (list of routes)
    """
    build_cache(instance)
    rng = random.Random(seed)
    random.seed(seed)  # generate_*_candidates uses random internally

    # --- Initial solution ---
    if initial_routes is None:
        if verbose:
            print("  TS: building initial solution (Clarke-Wright λ=0.85)...")
        routes = clarke_wright(instance, lambda_param=0.85)
    else:
        routes = copy.deepcopy(initial_routes)

    tabu_list = TabuList(tenure_min, tenure_max, rng)

    best_routes = copy.deepcopy(routes)
    best_proxy = solution_score_proxy(routes, instance)
    current_proxy = best_proxy
    no_improve = 0
    intensify_countdown = 0
    t_start = time.time()

    if verbose:
        print(f"  TS: start | {solution_summary(routes, instance)}")

    for iteration in range(1, max_iterations + 1):
        # Cleanup tabu list periodically
        if iteration % 200 == 0:
            tabu_list.cleanup(iteration)

        # --- Generate candidates (alternating move types) ---
        use_chain = (iteration % 3 != 0)  # 2 out of 3 iters: chain moves
        chain_len = rng.choice([1, 2]) if use_chain else 1

        if chain_len == 1:
            candidates = generate_relocate_candidates(
                routes, tabu_list, instance, iteration, best_proxy,
                n_nearest=n_nearest, max_candidates=30
            )
        else:
            candidates = generate_or_opt_candidates(
                routes, tabu_list, instance, iteration, best_proxy,
                n_nearest=n_nearest // 2, chain_length=chain_len
            )

        if not candidates:
            # No candidates — try intra-2opt as fallback
            from sa_solver import move_intra_2opt
            new_routes = move_intra_2opt(copy.deepcopy(routes), instance, rng)
            if new_routes:
                candidates = [(solution_score_proxy(new_routes, instance),
                               new_routes, None)]

        if not candidates:
            no_improve += 1
            continue

        # --- Apply best candidate ---
        new_proxy, new_routes, tabu_attr = candidates[0]
        routes = new_routes
        current_proxy = new_proxy

        # Update tabu list
        if tabu_attr is not None:
            tabu_list.add(tabu_attr[0], tabu_attr[1], iteration)

        # Track best
        if current_proxy < best_proxy:
            best_proxy = current_proxy
            best_routes = copy.deepcopy(routes)
            no_improve = 0
            intensify_countdown = intensify_burst
            if verbose and iteration % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  TS @{iteration:,} NEW BEST | "
                      f"{solution_summary(best_routes, instance)} | "
                      f"{elapsed:.0f}s")
        else:
            no_improve += 1
            if intensify_countdown > 0:
                intensify_countdown -= 1

        # --- Diversification ---
        if no_improve >= diversify_after:
            tabu_list.extend_tenure(amount=2)
            no_improve = 0
            if verbose:
                elapsed = time.time() - t_start
                print(f"  TS @{iteration:,} DIVERSIFY | "
                      f"tenure now [{tabu_list.tenure_min},{tabu_list.tenure_max}] | "
                      f"{elapsed:.0f}s")

        # Callback
        if callback and iteration % 100 == 0:
            callback(iteration, best_proxy)

        if verbose and iteration % 1_000 == 0:
            elapsed = time.time() - t_start
            print(f"  TS @{iteration:,} | "
                  f"{solution_summary(best_routes, instance)} | "
                  f"no_improve={no_improve} | {elapsed:.0f}s")

    if verbose:
        elapsed = time.time() - t_start
        print(f"  TS: done in {elapsed:.1f}s | {solution_summary(best_routes, instance)}")

    return best_routes


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys

    path = sys.argv[1] if len(sys.argv) > 1 else "istanbul_small_100.json"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3_000

    with open(path) as f:
        instance = json.load(f)
    build_cache(instance)

    print(f"Instance: {instance['name']} ({instance['n_customers']} customers)")
    print(f"Running TS for {iters:,} iterations...")
    print()

    routes = ts_solve(instance, max_iterations=iters, verbose=True)

    from evaluate_solution import validate_solution, score_solution
    sol = {"team": "TS", "routes": routes}
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