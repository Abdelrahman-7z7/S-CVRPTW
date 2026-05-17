"""
CSE 557 — Solver Core
Shared utilities used by every algorithm module.

All travel-time calculations use the deterministic model from evaluate_solution.py.
Distance matrix is (N+1)×(N+1), indexed by node ID: depot=0, customers=1..N.
Time is in minutes from midnight. Depot opens at 420 (07:00), closes at 1200 (20:00).
"""

import copy
from evaluate_solution import deterministic_travel_time


# ---------------------------------------------------------------------------
# Travel-time wrapper
# ---------------------------------------------------------------------------

def det_tt(instance, from_id, to_id, departure_min):
    """
    Deterministic travel time from node from_id to node to_id,
    departing at departure_min. Returns travel time in minutes.
    """
    dist = instance["distance_matrix"][from_id][to_id]
    return deterministic_travel_time(dist, departure_min, instance["traffic"])


# ---------------------------------------------------------------------------
# Route-level feasibility
# ---------------------------------------------------------------------------

def route_arrival_times(route, instance):
    """
    Simulate a route and return the list of arrival times at each customer
    and the return time to the depot.

    Returns (arrivals, return_time) where:
        arrivals[i] = actual arrival time at route[i] (after waiting if early)
        return_time  = time vehicle arrives back at depot
    Returns None if any hard constraint is violated.
    """
    depot = instance["depot"]
    customers = instance["_customers"]  # pre-built dict, see build_cache()

    t = depot["tw_open"]
    prev = 0
    arrivals = []

    infeasible_set = instance.get("_infeasible_customers", set())
    for cid in route:
        c = customers[cid]
        tt = det_tt(instance, prev, cid, t)
        arr = t + tt
        if arr < c["tw_open"]:
            arr = c["tw_open"]
        if arr > c["tw_close"]:
            if cid in infeasible_set:
                pass
            else:
                return None, None
        t = arr + c["service_time"]
        arrivals.append(arr)
        prev = cid

    tt_back = det_tt(instance, prev, 0, t)
    return_time = t + tt_back
    if return_time > depot["tw_close"]:
        if prev in infeasible_set:
            pass
        else:
            return None, None

    return arrivals, return_time


def is_route_feasible(route, instance):
    """
    Returns True if route satisfies all hard constraints:
      - capacity
      - all customer time windows (deterministic)
      - depot return before closing
    """
    if not route:
        return False

    customers = instance["_customers"]
    cap = instance["vehicle_capacity"]

    load = sum(customers[cid]["demand"] for cid in route)
    if load > cap:
        return False

    arrivals, _ = route_arrival_times(route, instance)
    return arrivals is not None


def route_load(route, instance):
    """Total demand on a route."""
    customers = instance["_customers"]
    return sum(customers[cid]["demand"] for cid in route)


# ---------------------------------------------------------------------------
# Insertion helpers
# ---------------------------------------------------------------------------

def insertion_cost(route, customer_id, position, instance):
    """
    Compute the extra travel time incurred by inserting customer_id
    at position `position` in route (0 = before first customer, etc.).

    Returns the delta travel time (can be negative due to waiting-time reduction).
    Does NOT check feasibility.
    """
    depot = instance["depot"]
    customers = instance["_customers"]
    dist = instance["distance_matrix"]
    traffic = instance["traffic"]

    c = customers[customer_id]
    prev_node = 0 if position == 0 else route[position - 1]
    next_node = 0 if position == len(route) else route[position]

    d_remove = dist[prev_node][next_node]
    d_insert = dist[prev_node][customer_id] + dist[customer_id][next_node]

    # Naive delta (ignores time-dependency, used for fast screening)
    return d_insert - d_remove


def best_insertion(route, customer_id, instance):
    """
    Find the best feasible position to insert customer_id into route.

    Tries every position (0..len(route)). Checks full feasibility including
    time windows and depot return after each candidate insertion.

    Returns (best_position, best_delta_travel_time) or (None, inf) if impossible.
    """
    customers = instance["_customers"]
    cap = instance["vehicle_capacity"]

    # Capacity check first
    if route_load(route, instance) + customers[customer_id]["demand"] > cap:
        return None, float("inf")

    best_pos = None
    best_delta = float("inf")

    for pos in range(len(route) + 1):
        candidate = route[:pos] + [customer_id] + route[pos:]
        arrivals, _ = route_arrival_times(candidate, instance)
        if arrivals is None:
            continue
        delta = insertion_cost(route, customer_id, pos, instance)
        if delta < best_delta:
            best_delta = delta
            best_pos = pos

    return best_pos, best_delta


def best_insertion_across_routes(routes, customer_id, instance):
    """
    Find the globally best feasible insertion of customer_id across all routes.

    Returns (route_index, position, delta) or (None, None, inf).
    """
    best_r = None
    best_pos = None
    best_delta = float("inf")

    for r_idx, route in enumerate(routes):
        pos, delta = best_insertion(route, customer_id, instance)
        if pos is not None and delta < best_delta:
            best_delta = delta
            best_pos = pos
            best_r = r_idx

    return best_r, best_pos, best_delta


# ---------------------------------------------------------------------------
# Solution cost (deterministic proxy used during search)
# ---------------------------------------------------------------------------

def route_travel_time(route, instance):
    """
    Total deterministic travel time for one route (depot→...→depot).
    Returns inf if route is infeasible.
    """
    if not route:
        return 0.0

    depot = instance["depot"]
    customers = instance["_customers"]
    t = depot["tw_open"]
    prev = 0
    total = 0.0

    infeasible_set = instance.get("_infeasible_customers", set())
    penalty = 0.0
    for cid in route:
        c = customers[cid]
        tt = det_tt(instance, prev, cid, t)
        total += tt
        arr = t + tt
        if arr < c["tw_open"]:
            arr = c["tw_open"]
        if arr > c["tw_close"]:
            if cid in infeasible_set:
                penalty += 5000.0
            else:
                return float("inf")
        t = arr + c["service_time"]
        prev = cid

    tt_back = det_tt(instance, prev, 0, t)
    total += tt_back
    if t + tt_back > depot["tw_close"]:
        if prev in infeasible_set:
            penalty += 5000.0
        else:
            return float("inf")

    return total + penalty


def solution_travel_time(routes, instance):
    """
    Total deterministic travel time across all routes.
    Used as the fast proxy objective during search (avoids 100 stochastic sims).
    Returns inf if any route is infeasible.
    """
    total = 0.0
    for route in routes:
        rt = route_travel_time(route, instance)
        if rt == float("inf"):
            return float("inf")
        total += rt
    return total


def solution_score_proxy(routes, instance):
    """
    Fast proxy score: n_vehicles * 1000 + total_travel_time.
    Violations not included (deterministic search keeps violations at 0).
    """
    return len(routes) * 1000 + solution_travel_time(routes, instance)


# ---------------------------------------------------------------------------
# Instance cache builder — call once before running any algorithm
# ---------------------------------------------------------------------------

def build_cache(instance):
    """
    Attach pre-built lookup structures to the instance dict so every
    function can access them in O(1) without rebuilding.

    Mutates instance in-place. Safe to call multiple times (idempotent).
    """
    if "_customers" not in instance:
        instance["_customers"] = {c["id"]: c for c in instance["customers"]}

    if "_infeasible_customers" not in instance:
        from evaluate_solution import deterministic_travel_time
        depot = instance["depot"]
        traffic = instance["traffic"]
        dist = instance["distance_matrix"]
        customers = instance["_customers"]
        infeasible = set()
        for cid, c in customers.items():
            # Type 1: window opens after latest possible departure to reach depot
            tt_back = deterministic_travel_time(dist[cid][0], c["tw_open"], traffic)
            if c["tw_open"] > depot["tw_close"] - tt_back:
                infeasible.add(cid)
                continue
            # Type 2: unreachable from depot even as first stop
            tt_to = deterministic_travel_time(dist[0][cid], depot["tw_open"], traffic)
            arr = depot["tw_open"] + tt_to
            if arr < c["tw_open"]:
                arr = c["tw_open"]
            if arr > c["tw_close"]:
                infeasible.add(cid)
        instance["_infeasible_customers"] = infeasible
        if infeasible:
            print(f"  [build_cache] WARNING: {len(infeasible)} structurally "
                  f"infeasible customers (instance data errors): {sorted(infeasible)}")
            print(f"  [build_cache] These will receive a penalty instead of inf score.")


    if "_neighbor_lists" not in instance:
        n = instance["n_customers"]
        dist = instance["distance_matrix"]
        neighbor_lists = {}
        for i in range(n + 1):
            row = dist[i]
            # Sort all other nodes by distance, exclude self and depot
            sorted_nodes = sorted(
                [j for j in range(1, n + 1) if j != i],
                key=lambda j: row[j]
            )
            neighbor_lists[i] = sorted_nodes
        instance["_neighbor_lists"] = neighbor_lists

    return instance


def get_neighbors(customer_id, instance, n_nearest=15):
    """
    Return list of up to n_nearest customers closest to customer_id.
    Requires build_cache() to have been called.
    """
    return instance["_neighbor_lists"][customer_id][:n_nearest]


# ---------------------------------------------------------------------------
# Solution validation wrapper (calls official validator)
# ---------------------------------------------------------------------------

def validate(routes, instance):
    """
    Validate routes against all hard constraints using the official evaluator.
    Returns (is_valid, errors).
    """
    from evaluate_solution import validate_solution
    solution = {"team": "solver_core_check", "routes": routes}
    return validate_solution(instance, solution)


def assert_valid(routes, instance, label=""):
    """
    Raise an AssertionError with details if the solution is invalid.
    Use during development to catch bugs immediately.
    """
    ok, errors = validate(routes, instance)
    if not ok:
        msg = f"[{label}] INVALID SOLUTION:\n" + "\n".join(f"  - {e}" for e in errors)
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------

def all_customer_ids(instance):
    return set(range(1, instance["n_customers"] + 1))


def customers_in_solution(routes):
    return set(cid for route in routes for cid in route)


def solution_summary(routes, instance):
    """Print a one-line summary of a solution."""
    n_v = len(routes)
    tt = solution_travel_time(routes, instance)
    sizes = [len(r) for r in routes]
    return (f"Vehicles={n_v} | TT={tt:.1f} | "
            f"Proxy={solution_score_proxy(routes, instance):.1f} | "
            f"RouteSize min={min(sizes)} avg={sum(sizes)/len(sizes):.1f} max={max(sizes)}")