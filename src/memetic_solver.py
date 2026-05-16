"""
CSE 557 — Memetic Algorithm (MA)
Evolutionary population search where each individual receives local
optimisation (SA or TS) before and after genetic operators.

"Memetic" = GA (global exploration) + local search (local exploitation).
This combines the diversity of population-based search with the precision
of single-solution refinement.

Population seeding: Clarke-Wright with diverse lambda values.
Crossover: Route-Based Crossover (RBX) — preserves feasible routes intact.
Mutation: 4 operators targeting vehicle reduction and travel time.
Local search: SA (fast) or TS (strong) per individual, per generation.
Diversity control: inject fresh Clarke-Wright solutions on convergence.
"""

import copy
import random
import time

from solver_core import (
    build_cache, best_insertion, best_insertion_across_routes,
    is_route_feasible, route_load, route_travel_time,
    solution_travel_time, solution_score_proxy, solution_summary
)
from construction import clarke_wright, clarke_wright_population, greedy_insert_unrouted


# ---------------------------------------------------------------------------
# Crossover — Route-Based Crossover (RBX)
# ---------------------------------------------------------------------------

def route_based_crossover(parent1, parent2, instance, rng):
    """
    Route-Based Crossover for VRP:

    1. Select parent1's route with the lowest travel time (best quality route).
    2. Copy it intact into the offspring.
    3. Remove those customers from parent2's routes.
    4. Copy remaining (pruned) routes from parent2.
    5. Re-insert any missing customers greedily.

    This preserves at least one high-quality sequence from parent1 while
    inheriting the global route structure from parent2.

    Returns: offspring (list of routes)
    """
    # Step 1: select best route from parent1 by travel time
    def rt(r):
        return route_travel_time(r, instance)

    best_route = min(parent1, key=rt)
    offspring = [copy.deepcopy(best_route)]
    taken = set(best_route)

    # Step 2 & 3: copy pruned routes from parent2
    for route in parent2:
        cleaned = [c for c in route if c not in taken]
        if cleaned:
            offspring.append(cleaned)
            taken.update(cleaned)

    # Step 4: repair any stranded customers (safety net)
    all_customers = set(range(1, instance["n_customers"] + 1))
    missing = all_customers - taken
    if missing:
        offspring = greedy_insert_unrouted(offspring, missing, instance)

    # Step 5: remove empty routes and validate
    offspring = [r for r in offspring if r]
    return offspring


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

def mutate_relocate(routes, instance, rng):
    """Move one random customer to the best feasible position in any route."""
    if not routes:
        return routes
    r_from_idx = rng.randrange(len(routes))
    if not routes[r_from_idx]:
        return routes
    pos = rng.randrange(len(routes[r_from_idx]))
    cid = routes[r_from_idx][pos]

    new_routes = [list(r) for r in routes]
    new_routes[r_from_idx].pop(pos)
    temp_routes = [r for r in new_routes if r]

    r_idx, ins_pos, _ = best_insertion_across_routes(temp_routes, cid, instance)
    if r_idx is not None:
        temp_routes[r_idx].insert(ins_pos, cid)
        return temp_routes
    # Fallback: put back
    return routes


def mutate_shuffle(routes, instance, rng):
    """Randomly reorder customers within one route, accept if feasible."""
    if not routes:
        return routes
    r_idx = rng.randrange(len(routes))
    route = list(routes[r_idx])
    rng.shuffle(route)
    if is_route_feasible(route, instance):
        new_routes = [list(r) for r in routes]
        new_routes[r_idx] = route
        return new_routes
    return routes


def mutate_merge_attempt(routes, instance, rng):
    """Try to merge the two smallest routes into one."""
    if len(routes) < 2:
        return routes

    sorted_idxs = sorted(range(len(routes)), key=lambda i: len(routes[i]))
    r1_idx, r2_idx = sorted_idxs[0], sorted_idxs[1]

    merged = routes[r1_idx] + routes[r2_idx]
    cap = instance["vehicle_capacity"]
    customers = instance["_customers"]

    if sum(customers[c]["demand"] for c in merged) <= cap:
        if is_route_feasible(merged, instance):
            new_routes = [list(r) for i, r in enumerate(routes)
                          if i not in (r1_idx, r2_idx)]
            new_routes.append(merged)
            return new_routes

    # Try reversed combination
    merged_rev = routes[r2_idx] + routes[r1_idx]
    if sum(customers[c]["demand"] for c in merged_rev) <= cap:
        if is_route_feasible(merged_rev, instance):
            new_routes = [list(r) for i, r in enumerate(routes)
                          if i not in (r1_idx, r2_idx)]
            new_routes.append(merged_rev)
            return new_routes

    return routes


def mutate_2opt_star(routes, instance, rng):
    """Apply one random inter-route 2-opt* swap."""
    from sa_solver import move_2opt_star
    result = move_2opt_star(copy.deepcopy(routes), instance, rng)
    return result if result is not None else routes


def mutate(routes, instance, rng, mutation_rate=0.15):
    """
    Apply one of 4 mutation operators if random draw < mutation_rate.
    Operator weights favour vehicle-reducing moves (merge, relocate).
    """
    if rng.random() > mutation_rate:
        return routes

    op = rng.choices(
        ["relocate", "shuffle", "merge", "2opt*"],
        weights=[0.35, 0.15, 0.35, 0.15]
    )[0]

    if op == "relocate":
        return mutate_relocate(routes, instance, rng)
    elif op == "shuffle":
        return mutate_shuffle(routes, instance, rng)
    elif op == "merge":
        return mutate_merge_attempt(routes, instance, rng)
    else:
        return mutate_2opt_star(routes, instance, rng)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def tournament_select(population, scores, k=3, rng=None):
    """
    Tournament selection: sample k individuals, return the index of
    the one with the lowest (best) proxy score.
    """
    contestants = rng.sample(range(len(population)), min(k, len(population)))
    return min(contestants, key=lambda i: scores[i])


# ---------------------------------------------------------------------------
# Local search wrappers (called on each individual)
# ---------------------------------------------------------------------------

def apply_local_search(routes, instance, method="sa",
                        iterations=500, seed=42):
    """
    Apply SA or TS local search to a single individual.
    Returns improved routes.
    """
    if method == "sa":
        from sa_solver import sa_solve
        return sa_solve(
            instance,
            initial_routes=routes,
            max_iterations=iterations,
            verbose=False,
            seed=seed
        )
    elif method == "ts":
        from ts_solver import ts_solve
        # TS iterations ≈ SA iterations / 50 (TS is more expensive per iter)
        ts_iters = max(50, iterations // 50)
        return ts_solve(
            instance,
            initial_routes=routes,
            max_iterations=ts_iters,
            verbose=False,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown local search method: {method}")


# ---------------------------------------------------------------------------
# Diversity metric
# ---------------------------------------------------------------------------

def population_diversity(scores):
    """
    Returns the coefficient of variation (std/mean) of proxy scores.
    Low diversity (< 0.005) triggers injection of fresh individuals.
    """
    if len(scores) < 2:
        return 1.0
    mean = sum(scores) / len(scores)
    if mean == 0:
        return 0.0
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance ** 0.5
    return std / mean


# ---------------------------------------------------------------------------
# Main MA loop
# ---------------------------------------------------------------------------

def ma_solve(instance,
             pop_size=20,
             n_generations=200,
             local_search="sa",
            #  ls_iterations=500, # for the small data
             ls_iterations=5000, # for the large data
             crossover_prob=0.85,
             mutation_rate=0.15,
             tournament_k=3,
             elitism=2,
            #  diversity_threshold=0.005, # for the small data
            diversity_threshold=0.002, # for the large data
            #  inject_count=3, # for the small data
             inject_count=6, # for the large data
             lambda_range=(0.8, 1.2),
             seed=42,
             verbose=True,
             callback=None):
    """
    Memetic Algorithm for S-CVRPTW.

    Parameters:
        instance           : loaded + cached instance dict
        pop_size           : population size
        n_generations      : number of generations
        local_search       : 'sa' or 'ts' (applied to each individual)
        ls_iterations      : local search iterations per individual
        crossover_prob     : probability of crossover (vs copy parent)
        mutation_rate      : probability of applying mutation
        tournament_k       : tournament selection size
        elitism            : number of elite individuals copied unchanged
        diversity_threshold: CoV below this triggers fresh injection
        inject_count       : number of fresh solutions to inject on low diversity
        lambda_range       : Clarke-Wright lambda range for population seeding
        seed               : random seed
        verbose            : print per-generation progress
        callback           : callable(gen, best_proxy, avg_proxy, n_vehicles)

    Returns:
        best_routes (list of routes)
    """
    build_cache(instance)
    rng = random.Random(seed)
    t_start = time.time()

    # -----------------------------------------------------------------------
    # Step 1: Initialise population
    # -----------------------------------------------------------------------
    if verbose:
        print(f"  MA: initialising population (size={pop_size}, "
              f"ls={local_search}, ls_iters={ls_iterations})...")

    population = clarke_wright_population(
        instance, size=pop_size, lambda_range=lambda_range
    )

    # Apply initial local search to each individual
    if verbose:
        print("  MA: applying initial local search to population...")

    for i in range(len(population)):
        population[i] = apply_local_search(
            population[i], instance,
            method=local_search,
            iterations=ls_iterations,
            seed=seed + i
        )

    # Score population
    scores = [solution_score_proxy(ind, instance) for ind in population]
    best_idx = min(range(len(scores)), key=lambda i: scores[i])
    best_routes = copy.deepcopy(population[best_idx])
    best_proxy = scores[best_idx]

    if verbose:
        print(f"  MA Gen 0 | Best={best_proxy:.1f} | "
              f"Avg={sum(scores)/len(scores):.1f} | "
              f"Vehicles={len(best_routes)} | "
              f"Diversity={population_diversity(scores):.4f}")

    # -----------------------------------------------------------------------
    # Step 2: Evolution loop
    # -----------------------------------------------------------------------
    for gen in range(1, n_generations + 1):
        new_population = []

        # Elitism: copy top E individuals unchanged
        elite_idxs = sorted(range(len(scores)),
                             key=lambda i: scores[i])[:elitism]
        for ei in elite_idxs:
            new_population.append(copy.deepcopy(population[ei]))

        # Fill rest of population through selection → crossover → mutation → LS
        child_seed = seed + gen * 1000
        while len(new_population) < pop_size:
            # Selection
            p1_idx = tournament_select(population, scores, tournament_k, rng)
            p2_idx = tournament_select(population, scores, tournament_k, rng)

            # Crossover
            if rng.random() < crossover_prob:
                child = route_based_crossover(
                    population[p1_idx], population[p2_idx], instance, rng
                )
            else:
                child = copy.deepcopy(population[p1_idx])

            # Mutation
            child = mutate(child, instance, rng, mutation_rate)

            # Local search (the memetic step)
            child = apply_local_search(
                child, instance,
                method=local_search,
                iterations=ls_iterations,
                seed=child_seed + len(new_population)
            )

            new_population.append(child)

        population = new_population
        scores = [solution_score_proxy(ind, instance) for ind in population]

        # -----------------------------------------------------------------------
        # Diversity control: inject fresh solutions if population converged
        # -----------------------------------------------------------------------
        div = population_diversity(scores)
        if div < diversity_threshold:
            if verbose:
                print(f"  MA Gen {gen:3d} | LOW DIVERSITY ({div:.5f}) "
                      f"— injecting {inject_count} fresh solutions")
            fresh_lambdas = [rng.uniform(0.7, 1.3) for _ in range(inject_count)]
            for lam in fresh_lambdas:
                fresh = clarke_wright(instance, lambda_param=lam)
                fresh = apply_local_search(fresh, instance, method=local_search,
                                            iterations=ls_iterations,
                                            seed=rng.randint(0, 99_999))
                # Replace worst individual
                worst_idx = max(range(len(scores)), key=lambda i: scores[i])
                population[worst_idx] = fresh
                scores[worst_idx] = solution_score_proxy(fresh, instance)

        # Track global best
        gen_best_idx = min(range(len(scores)), key=lambda i: scores[i])
        if scores[gen_best_idx] < best_proxy:
            best_proxy = scores[gen_best_idx]
            best_routes = copy.deepcopy(population[gen_best_idx])

        # Callback
        avg_score = sum(scores) / len(scores)
        if callback:
            callback(gen, best_proxy, avg_score, len(best_routes))

        if verbose and (gen % 10 == 0 or gen <= 5):
            elapsed = time.time() - t_start
            print(f"  MA Gen {gen:3d} | Best={best_proxy:.1f} | "
                  f"Avg={avg_score:.1f} | "
                  f"Vehicles={len(best_routes)} | "
                  f"Div={div:.4f} | {elapsed:.0f}s")

    if verbose:
        elapsed = time.time() - t_start
        print(f"  MA: done in {elapsed:.1f}s | {solution_summary(best_routes, instance)}")

    return best_routes


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys

    path = sys.argv[1] if len(sys.argv) > 1 else "istanbul_small_100.json"
    ls_method = sys.argv[2] if len(sys.argv) > 2 else "sa"
    gen = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    with open(path) as f:
        instance = json.load(f)
    build_cache(instance)

    print(f"Instance: {instance['name']} ({instance['n_customers']} customers)")
    print(f"Running MA({ls_method}) for {gen} generations...")
    print()

    routes = ma_solve(
        instance,
        pop_size=10,
        n_generations=gen,
        local_search=ls_method,
        ls_iterations=3_000,
        verbose=True
    )

    from evaluate_solution import validate_solution, score_solution
    sol = {"team": f"MA({ls_method})", "routes": routes}
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