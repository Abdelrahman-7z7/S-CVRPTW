"""
CSE 557 — Ant Colony Optimization (ACS variant)
Multiple ants each construct a complete route set using pheromone trails.
Best-ant pheromone update reinforces high-quality edges.
Hybrid: each ant solution is refined with SA local search (optional).

Pheromone structure: τ[i][j] = desirability of visiting j immediately after i.
Heuristic: η[i][j] = 1/distance + time-window urgency bonus.
ACS rule: with prob q0 exploit best known edge, else probabilistic selection.
"""

import copy
import random
import time

from evaluate_solution import deterministic_travel_time
from solver_core import (
    build_cache, is_route_feasible, route_load,
    solution_travel_time, solution_score_proxy, solution_summary
)
from construction import clarke_wright


# ---------------------------------------------------------------------------
# Pheromone matrix
# ---------------------------------------------------------------------------

class PheromoneMatrix:
    """
    (N+1) × (N+1) pheromone matrix. Index 0 = depot, 1..N = customers.
    Implements MMAS bounds (τ_min, τ_max) to prevent stagnation.
    """

    def __init__(self, n_nodes, tau_min=0.001, tau_max=1.0, tau_init=0.1):
        self.n = n_nodes
        self.tau_min = tau_min
        self.tau_max = tau_max
        # Flat list for speed: index = i * (n+1) + j
        size = (n_nodes + 1) ** 2
        self._tau = [tau_init] * size
        self._stride = n_nodes + 1

    def get(self, i, j):
        return self._tau[i * self._stride + j]

    def set(self, i, j, val):
        self._tau[i * self._stride + j] = max(
            self.tau_min, min(self.tau_max, val)
        )

    def global_update(self, routes, score, rho=0.1):
        """
        ACS global update: apply pheromone only on edges used by the
        best ant this iteration. Evaporate all other edges.
        delta = 1 / score (better score → stronger pheromone deposit).
        """
        if score <= 0 or score == float("inf"):
            return
        delta = 1.0 / score

        # Evaporate ALL edges
        for k in range(len(self._tau)):
            self._tau[k] = max(self.tau_min, (1 - rho) * self._tau[k])

        # Deposit on best-ant edges
        for route in routes:
            path = [0] + route + [0]
            for k in range(len(path) - 1):
                i, j = path[k], path[k + 1]
                idx = i * self._stride + j
                self._tau[idx] = min(self.tau_max,
                                     self._tau[idx] + rho * delta)

    def local_update(self, i, j, xi=0.1, tau_init=0.1):
        """
        ACS local update (applied during construction):
        reduces pheromone on just-used edge to discourage other ants
        from copying the exact same path.
        """
        idx = i * self._stride + j
        self._tau[idx] = max(
            self.tau_min,
            (1 - xi) * self._tau[idx] + xi * tau_init
        )


# ---------------------------------------------------------------------------
# Heuristic matrix (precomputed once)
# ---------------------------------------------------------------------------

def compute_heuristic(instance):
    """
    η[i][j] = base_closeness + urgency_bonus

    base_closeness = 1 / (dist[i][j] + ε)     (prefer closer customers)
    urgency_bonus  = 0.5 / (tw_width[j] + 1)  (prefer tight-window customers)

    Precomputed as (N+1) flat list for O(1) lookup.
    Returns (eta_flat, stride) where eta_flat[i*(N+1)+j] = η[i][j].
    """
    n = instance["n_customers"]
    dist = instance["distance_matrix"]
    customers = instance["_customers"]
    stride = n + 1

    eta = [0.0] * (stride * stride)
    for i in range(stride):
        for j in range(1, stride):
            if i == j:
                continue
            d = dist[i][j]
            base = 1.0 / (d + 1e-6)
            c = customers.get(j)
            if c:
                tw_width = c["tw_close"] - c["tw_open"]
                urgency = 0.5 / (tw_width + 1.0)
            else:
                urgency = 0.0
            eta[i * stride + j] = base + urgency

    return eta, stride


# ---------------------------------------------------------------------------
# Single ant construction
# ---------------------------------------------------------------------------

def construct_solution(instance, pheromone, eta, eta_stride,
                        alpha=1.0, beta=2.0, q0=0.9, rng=None,
                        xi=0.1, tau_init=0.1):
    """
    One ant builds a complete solution (set of routes) using ACS rule.

    ACS selection rule at each step:
        With probability q0: exploit → pick j* = argmax(τ[i][j]^α * η[i][j]^β)
        With probability 1-q0: explore → probabilistic selection

    Returns list of routes.
    """
    rng = rng or random.Random()
    n = instance["n_customers"]
    cap = instance["vehicle_capacity"]
    customers = instance["_customers"]
    dist = instance["distance_matrix"]
    traffic = instance["traffic"]
    depot = instance["depot"]

    unvisited = list(range(1, n + 1))
    unvisited_set = set(unvisited)
    routes = []

    while unvisited_set:
        route = []
        load = 0
        t = depot["tw_open"]
        prev = 0

        while True:
            # Find feasible candidates from unvisited
            feasible = []
            for cid in unvisited_set:
                c = customers[cid]
                if load + c["demand"] > cap:
                    continue
                tt = deterministic_travel_time(dist[prev][cid], t, traffic)
                arr = t + tt
                if arr < c["tw_open"]:
                    arr = c["tw_open"]
                if arr > c["tw_close"]:
                    continue
                dep = arr + c["service_time"]
                tt_back = deterministic_travel_time(dist[cid][0], dep, traffic)
                if dep + tt_back > depot["tw_close"]:
                    continue
                feasible.append(cid)

            if not feasible:
                break

            # ACS selection
            if rng.random() < q0:
                # Exploitation: best τ^α * η^β
                best_cid = None
                best_val = -1.0
                for cid in feasible:
                    tau = pheromone.get(prev, cid)
                    h = eta[prev * eta_stride + cid]
                    val = (tau ** alpha) * (h ** beta)
                    if val > best_val:
                        best_val = val
                        best_cid = cid
                next_cid = best_cid
            else:
                # Exploration: roulette wheel
                weights = []
                for cid in feasible:
                    tau = pheromone.get(prev, cid)
                    h = eta[prev * eta_stride + cid]
                    weights.append((tau ** alpha) * (h ** beta))
                total = sum(weights)
                if total <= 0:
                    next_cid = rng.choice(feasible)
                else:
                    probs = [w / total for w in weights]
                    next_cid = rng.choices(feasible, weights=probs)[0]

            # Apply local pheromone update (ACS)
            pheromone.local_update(prev, next_cid, xi=xi, tau_init=tau_init)

            # Move to next customer
            c = customers[next_cid]
            tt = deterministic_travel_time(dist[prev][next_cid], t, traffic)
            arr = t + tt
            if arr < c["tw_open"]:
                arr = c["tw_open"]
            t = arr + c["service_time"]
            load += c["demand"]
            route.append(next_cid)
            unvisited_set.discard(next_cid)
            prev = next_cid

        if route:
            routes.append(route)
        elif unvisited_set:
            # Fallback: force-create a singleton route for the first unroutable
            cid = next(iter(unvisited_set))
            routes.append([cid])
            unvisited_set.discard(cid)

    return routes


# ---------------------------------------------------------------------------
# Main ACS loop
# ---------------------------------------------------------------------------

def aco_solve(instance,
              n_ants=20,
              n_iterations=300,
              alpha=1.0,
              beta=2.0,
              rho=0.1,
              q0=0.9,
              tau_min=0.001,
              tau_max=1.0,
              tau_init=0.1,
              xi=0.1,
              apply_local_search=True,
              ls_iterations=200,
              seed=42,
              verbose=True,
              callback=None):
    """
    Full ACS loop with optional SA local search per ant (hybrid ACO).

    Parameters:
        n_ants           : ants per iteration (= solutions constructed)
        n_iterations     : total ACS iterations
        alpha            : pheromone weight (higher → more exploitation)
        beta             : heuristic weight (higher → more greedy)
        rho              : global pheromone evaporation rate
        q0               : ACS exploitation probability
        tau_min/max      : MMAS pheromone bounds
        tau_init         : initial pheromone value
        xi               : local pheromone update rate
        apply_local_search: if True, apply SA to each ant's solution
        ls_iterations    : SA iterations per ant (keep low for speed)
        seed             : random seed

    Returns:
        best_routes (list of routes)
    """
    build_cache(instance)
    rng = random.Random(seed)

    n = instance["n_customers"]
    pheromone = PheromoneMatrix(n, tau_min, tau_max, tau_init)
    eta, eta_stride = compute_heuristic(instance)

    # Seed pheromone with Clarke-Wright solution edges
    seed_routes = clarke_wright(instance, lambda_param=0.85)
    seed_score = solution_score_proxy(seed_routes, instance)
    pheromone.global_update(seed_routes, seed_score, rho=0.5)

    best_routes = copy.deepcopy(seed_routes)
    best_proxy = seed_score
    t_start = time.time()

    if verbose:
        print(f"  ACO: start | {solution_summary(seed_routes, instance)}")
        print(f"  ACO: n_ants={n_ants} iters={n_iterations} "
              f"α={alpha} β={beta} ρ={rho} q0={q0} ls={apply_local_search}")

    for iteration in range(1, n_iterations + 1):
        iter_best_routes = None
        iter_best_proxy = float("inf")

        for ant in range(n_ants):
            ant_seed = seed + iteration * 1000 + ant

            # Construct solution
            routes = construct_solution(
                instance, pheromone, eta, eta_stride,
                alpha=alpha, beta=beta, q0=q0,
                rng=random.Random(ant_seed),
                xi=xi, tau_init=tau_init
            )

            # Optional: quick SA refinement
            if apply_local_search and ls_iterations > 0:
                from sa_solver import sa_solve
                routes = sa_solve(
                    instance,
                    initial_routes=routes,
                    max_iterations=ls_iterations,
                    verbose=False,
                    seed=ant_seed
                )

            proxy = solution_score_proxy(routes, instance)

            if proxy < iter_best_proxy:
                iter_best_proxy = proxy
                iter_best_routes = routes

        # Global pheromone update with iteration best
        if iter_best_routes is not None:
            pheromone.global_update(iter_best_routes, iter_best_proxy, rho=rho)

            if iter_best_proxy < best_proxy:
                best_proxy = iter_best_proxy
                best_routes = copy.deepcopy(iter_best_routes)

        # Callback
        if callback:
            callback(iteration, best_proxy, iter_best_proxy)

        if verbose and iteration % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  ACO @{iteration:3d} | "
                  f"Best={best_proxy:.1f} | "
                  f"IterBest={iter_best_proxy:.1f} | "
                  f"Vehicles={len(best_routes)} | "
                  f"{elapsed:.0f}s")

    if verbose:
        elapsed = time.time() - t_start
        print(f"  ACO: done in {elapsed:.1f}s | {solution_summary(best_routes, instance)}")

    return best_routes


# ---------------------------------------------------------------------------
# Parameter sweep helper (for Person C experiments)
# ---------------------------------------------------------------------------

def aco_parameter_sweep(instance, param_grid, n_ants=10, n_iterations=30,
                         ls_iterations=100, seed=42, verbose=True):
    """
    Run ACO with each parameter combination in param_grid.
    param_grid: list of dicts, e.g. [{"alpha": 1.0, "beta": 2.0, "rho": 0.1}]

    Returns list of (params, score, n_vehicles) sorted by score ascending.
    """
    from evaluate_solution import validate_solution, score_solution
    results = []

    for i, params in enumerate(param_grid):
        if verbose:
            print(f"\n[Sweep {i+1}/{len(param_grid)}] {params}")

        routes = aco_solve(
            instance,
            n_ants=n_ants,
            n_iterations=n_iterations,
            ls_iterations=ls_iterations,
            seed=seed,
            verbose=False,
            **params
        )

        sol = {"team": "ACO-sweep", "routes": routes}
        valid, _ = validate_solution(instance, sol)
        if valid:
            res = score_solution(instance, sol)
            results.append((params, res["score"], res["n_vehicles"]))
            if verbose:
                print(f"  Score={res['score']:.1f} | Vehicles={res['n_vehicles']}")
        else:
            results.append((params, float("inf"), -1))
            if verbose:
                print("  INVALID")

    results.sort(key=lambda x: x[1])
    return results


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys

    path = sys.argv[1] if len(sys.argv) > 1 else "istanbul_small_100.json"
    n_ants = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    with open(path) as f:
        instance = json.load(f)
    build_cache(instance)

    print(f"Instance: {instance['name']} ({instance['n_customers']} customers)")
    print(f"Running ACO: {n_ants} ants × {n_iters} iterations")
    print()

    routes = aco_solve(
        instance,
        n_ants=n_ants,
        n_iterations=n_iters,
        ls_iterations=500,
        apply_local_search=True,
        verbose=True
    )

    from evaluate_solution import validate_solution, score_solution
    sol = {"team": "ACO", "routes": routes}
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