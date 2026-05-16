"""
CSE 557 — Baseline S-CVRPTW Solver
Nearest-neighbour construction + intra-route 2-opt improvement.
Uses the same deterministic travel-time model as the official validator.

Usage:
    python baseline_solver.py <instance.json> [--output solution.json]
"""
import json
import os
import argparse
import time

from evaluate_solution import (
    load_json, deterministic_travel_time, validate_solution, score_solution
)


def _det_tt(dist_km, departure_min, traffic):
    """Deterministic travel time wrapper (same model as the validator)."""
    return deterministic_travel_time(dist_km, departure_min, traffic)


def solve(instance):
    """Build routes with nearest-neighbour, then improve with 2-opt."""
    n = instance["n_customers"]
    cap = instance["vehicle_capacity"]
    depot = instance["depot"]
    custs = {c["id"]: c for c in instance["customers"]}
    dist = instance["distance_matrix"]
    traffic = instance["traffic"]
    depot_open = depot["tw_open"]
    depot_close = depot["tw_close"]

    unvisited = set(range(1, n + 1))
    routes = []

    while unvisited:
        route = []
        load = 0
        cur_t = depot_open
        prev = 0

        while True:
            best_id = None
            best_d = float("inf")

            for cid in list(unvisited):
                c = custs[cid]
                if load + c["demand"] > cap:
                    continue
                d = dist[prev][cid]
                tt = _det_tt(d, cur_t, traffic)
                arr = cur_t + tt
                if arr < c["tw_open"]:
                    arr = c["tw_open"]
                if arr > c["tw_close"]:
                    continue
                dep = arr + c["service_time"]
                tt_back = _det_tt(dist[cid][0], dep, traffic)
                if dep + tt_back > depot_close:
                    continue
                if d < best_d:
                    best_d = d
                    best_id = cid

            if best_id is None:
                break

            c = custs[best_id]
            tt = _det_tt(dist[prev][best_id], cur_t, traffic)
            arr = cur_t + tt
            if arr < c["tw_open"]:
                arr = c["tw_open"]
            cur_t = arr + c["service_time"]
            load += c["demand"]
            route.append(best_id)
            unvisited.discard(best_id)
            prev = best_id

        if route:
            routes.append(route)
        else:
            print(f"    WARNING: {len(unvisited)} customers could not be routed")
            break

    print(f"    NN: {len(routes)} routes, "
          f"{sum(len(r) for r in routes)}/{n} customers")

    # 2-opt with deterministic TW feasibility
    def route_time(r):
        t = depot_open
        total_tt = 0.0
        prev_node = 0
        for cid in r:
            c = custs[cid]
            tt = _det_tt(dist[prev_node][cid], t, traffic)
            total_tt += tt
            arr = t + tt
            if arr < c["tw_open"]:
                arr = c["tw_open"]
            t = arr + c["service_time"]
            prev_node = cid
        tt_back = _det_tt(dist[prev_node][0], t, traffic)
        total_tt += tt_back
        return total_tt

    def tw_feasible(r):
        t = depot_open
        prev_node = 0
        for cid in r:
            c = custs[cid]
            tt = _det_tt(dist[prev_node][cid], t, traffic)
            arr = t + tt
            if arr < c["tw_open"]:
                arr = c["tw_open"]
            if arr > c["tw_close"]:
                return False
            t = arr + c["service_time"]
            prev_node = cid
        tt_back = _det_tt(dist[prev_node][0], t, traffic)
        return t + tt_back <= depot_close

    improved_count = 0
    for ri in range(len(routes)):
        r = routes[ri]
        best_cost = route_time(r)
        changed = True
        iters = 0
        while changed and iters < 30:
            changed = False
            iters += 1
            for i in range(len(r) - 1):
                for j in range(i + 2, min(i + 6, len(r))):
                    nr = r[:i] + r[i:j+1][::-1] + r[j+1:]
                    if not tw_feasible(nr):
                        continue
                    nc = route_time(nr)
                    if nc < best_cost - 0.01:
                        r = nr
                        best_cost = nc
                        changed = True
                        improved_count += 1
                        break
                if changed:
                    break
        routes[ri] = r

    print(f"    2-opt: {improved_count} improvements applied")
    return routes


def main():
    parser = argparse.ArgumentParser(description="CSE 557 Baseline Solver")
    parser.add_argument("instance", help="Path to instance JSON file")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    instance = load_json(args.instance)
    print(f"Instance: {instance['name']}  ({instance['n_customers']} customers)")

    t0 = time.time()
    routes = solve(instance)
    elapsed = time.time() - t0
    print(f"  Solved in {elapsed:.2f}s")

    solution = {"team": "Baseline (NN + 2-opt)", "routes": routes}

    valid, errors = validate_solution(instance, solution)
    if not valid:
        print("  VALIDATION FAILED:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  Validation: PASSED")

    print("  Scoring (100 stochastic scenarios)...")
    result = score_solution(instance, solution)
    print(f"  Vehicles: {result['n_vehicles']}")
    print(f"  Avg travel time: {result['avg_travel_time']:.2f} min")
    print(f"  Avg TW violations: {result['avg_tw_violations']:.2f}")
    print(f"  SCORE: {result['score']:.2f}  (teams must beat this)")

    if args.output is None:
        name = os.path.splitext(os.path.basename(args.instance))[0]
        args.output = f"baseline_{name}.json"
    out = os.path.join(os.path.dirname(args.instance), args.output)
    with open(out, "w") as f:
        json.dump(solution, f, indent=2)
    print(f"  Solution saved: {out}")


if __name__ == "__main__":
    main()
