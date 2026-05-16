"""
CSE 557 — Official S-CVRPTW Solution Evaluator
Validates a solution against an instance, then scores it under
100 stochastic traffic scenarios.

Usage:
    python evaluate_solution.py <instance.json> <solution.json> [--scenarios N]

Solution format (JSON):
    {
      "team": "Team Name",
      "routes": [
          [3, 17, 5, 42],     // route 1: depot -> 3 -> 17 -> 5 -> 42 -> depot
          [8, 12, 1],          // route 2: depot -> 8 -> 12 -> 1 -> depot
          ...
      ]
    }
"""
import json
import math
import random
import sys
import argparse


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_traffic_multiplier(departure_min, traffic):
    """Return (multiplier, sigma) for the given departure time."""
    regimes = traffic["regimes"]
    t = departure_min % 1440
    for reg in regimes:
        s, e = reg["start_min"], reg["end_min"]
        if s < e:
            if s <= t < e:
                return reg["multiplier"], reg["sigma"]
        else:  # wraps midnight
            if t >= s or t < e:
                return reg["multiplier"], reg["sigma"]
    return 1.0, 0.12


def stochastic_travel_time(base_dist, departure_min, traffic, rng):
    """Compute a single stochastic travel time sample."""
    speed = traffic["speed_km_per_min"]
    base_time = base_dist / speed
    mult, sigma = get_traffic_multiplier(departure_min, traffic)
    noise = rng.lognormvariate(0, sigma)
    return base_time * mult * noise


def deterministic_travel_time(base_dist, departure_min, traffic):
    """Expected travel time (no noise, multiplier only)."""
    speed = traffic["speed_km_per_min"]
    base_time = base_dist / speed
    mult, _ = get_traffic_multiplier(departure_min, traffic)
    return base_time * mult


def validate_solution(instance, solution):
    """Check hard constraints. Returns (is_valid, list_of_errors)."""
    errors = []
    n = instance["n_customers"]
    capacity = instance["vehicle_capacity"]
    depot = instance["depot"]
    customers = {c["id"]: c for c in instance["customers"]}
    dist = instance["distance_matrix"]
    traffic = instance["traffic"]

    visited = set()
    routes = solution.get("routes", [])

    if not routes:
        errors.append("No routes provided.")
        return False, errors

    for ri, route in enumerate(routes):
        if not route:
            errors.append(f"Route {ri+1}: empty route.")
            continue

        load = 0
        for cid in route:
            if cid < 1 or cid > n:
                errors.append(f"Route {ri+1}: invalid customer id {cid}.")
                continue
            if cid in visited:
                errors.append(f"Route {ri+1}: customer {cid} visited more than once.")
            visited.add(cid)
            load += customers[cid]["demand"]

        if load > capacity:
            errors.append(f"Route {ri+1}: capacity exceeded ({load} > {capacity}).")

        # Time feasibility check (deterministic)
        current_time = depot["tw_open"]
        prev = 0
        for cid in route:
            tt = deterministic_travel_time(dist[prev][cid], current_time, traffic)
            arrival = current_time + tt
            c = customers[cid]
            if arrival < c["tw_open"]:
                arrival = c["tw_open"]  # wait
            if arrival > c["tw_close"]:
                errors.append(
                    f"Route {ri+1}: customer {cid} time window violated "
                    f"(arrive {arrival:.1f} > close {c['tw_close']}).")
            current_time = arrival + c["service_time"]
            prev = cid

        # Return to depot
        tt = deterministic_travel_time(dist[prev][0], current_time, traffic)
        return_time = current_time + tt
        if return_time > depot["tw_close"]:
            errors.append(
                f"Route {ri+1}: depot closing time violated "
                f"(return {return_time:.1f} > close {depot['tw_close']}).")

    # Check all customers visited
    all_ids = set(range(1, n + 1))
    missing = all_ids - visited
    if missing:
        errors.append(f"Customers not visited: {sorted(missing)}")

    return len(errors) == 0, errors


def evaluate_scenario(instance, solution, rng):
    """Simulate one stochastic scenario. Returns (total_travel_time, violation_count)."""
    depot = instance["depot"]
    customers = {c["id"]: c for c in instance["customers"]}
    dist = instance["distance_matrix"]
    traffic = instance["traffic"]

    total_time = 0.0
    tw_violations = 0

    for route in solution["routes"]:
        current_time = depot["tw_open"]
        prev = 0
        for cid in route:
            tt = stochastic_travel_time(dist[prev][cid], current_time, traffic, rng)
            arrival = current_time + tt
            c = customers[cid]
            if arrival < c["tw_open"]:
                arrival = c["tw_open"]
            if arrival > c["tw_close"]:
                tw_violations += 1
            current_time = arrival + c["service_time"]
            total_time += tt
            prev = cid

        tt_return = stochastic_travel_time(dist[prev][0], current_time, traffic, rng)
        total_time += tt_return
        return_time = current_time + tt_return
        if return_time > depot["tw_close"]:
            tw_violations += 1

    return total_time, tw_violations


def score_solution(instance, solution, n_scenarios=100, seed=42):
    """
    Score = n_vehicles * 1000 + avg_travel_time + 50 * avg_tw_violations

    Lower is better.
    """
    rng = random.Random(seed)
    n_vehicles = len(solution["routes"])
    travel_times = []
    violation_counts = []

    for _ in range(n_scenarios):
        tt, viol = evaluate_scenario(instance, solution, rng)
        travel_times.append(tt)
        violation_counts.append(viol)

    avg_tt = sum(travel_times) / len(travel_times)
    avg_viol = sum(violation_counts) / len(violation_counts)
    std_tt = (sum((t - avg_tt) ** 2 for t in travel_times) / len(travel_times)) ** 0.5

    score = n_vehicles * 1000 + avg_tt + 50 * avg_viol
    return {
        "score": round(score, 2),
        "n_vehicles": n_vehicles,
        "avg_travel_time": round(avg_tt, 2),
        "std_travel_time": round(std_tt, 2),
        "avg_tw_violations": round(avg_viol, 2),
        "n_scenarios": n_scenarios,
    }


def main():
    parser = argparse.ArgumentParser(description="CSE 557 S-CVRPTW Evaluator")
    parser.add_argument("instance", help="Path to instance JSON file")
    parser.add_argument("solution", help="Path to solution JSON file")
    parser.add_argument("--scenarios", type=int, default=100,
                        help="Number of stochastic scenarios (default: 100)")
    args = parser.parse_args()

    instance = load_json(args.instance)
    solution = load_json(args.solution)

    print(f"Instance: {instance['name']}  ({instance['n_customers']} customers)")
    print(f"Team: {solution.get('team', 'Unknown')}")
    print(f"Routes: {len(solution['routes'])}")
    print()

    # Validate
    valid, errors = validate_solution(instance, solution)
    if not valid:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        print(f"\nScore: INVALID (fix constraint violations first)")
        sys.exit(1)

    print("Validation: PASSED (all hard constraints satisfied)")
    print()

    # Score
    result = score_solution(instance, solution, n_scenarios=args.scenarios)
    print(f"--- RESULTS ({result['n_scenarios']} stochastic scenarios) ---")
    print(f"  Vehicles used:           {result['n_vehicles']}")
    print(f"  Avg travel time:         {result['avg_travel_time']:.2f} min")
    print(f"  Std travel time:         {result['std_travel_time']:.2f} min")
    print(f"  Avg TW violations:       {result['avg_tw_violations']:.2f}")
    print(f"  ---------------------------------")
    print(f"  SCORE:                   {result['score']:.2f}")
    print(f"  (lower is better)")
    print(f"  Formula: vehicles*1000 + avg_time + 50*avg_violations")


if __name__ == "__main__":
    main()
