"""
CSE 557 — Experiment Runner
Unified benchmarking harness. Runs any solver function, validates the output,
scores it with 100 stochastic scenarios, and logs everything to CSV.

Usage (standalone):
    python experiment_runner.py <instance.json> [--quick] [--full]

    --quick : fast smoke-test (SA 50k, TS 1k, MA 3 gen, ACO 3 iter)
    --full  : overnight run on full_500 with best configuration
"""

import csv
import json
import os
import sys
import time
import copy

from evaluate_solution import validate_solution, score_solution
from solver_core import build_cache, solution_summary


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_experiment(solver_func, instance, label,
                   log_file="results.csv", verbose=True):
    """
    Run solver_func(instance), validate, score, and log.

    Always scores, logs to CSV, and saves the solution JSON — even if
    validation fails due to known instance data errors (e.g. structurally
    infeasible customers). Hard constraint violations are noted in the
    output but do not block recording.

    Parameters:
        solver_func : callable(instance) → routes
        instance    : loaded + cached instance dict
        label       : string identifier for this run (e.g. "SA_200k")
        log_file    : CSV file to append results to
        verbose     : print result to stdout

    Returns:
        result dict always (never None)
    """
    t0 = time.time()
    routes = solver_func(instance)
    elapsed = time.time() - t0

    solution = {"team": label, "routes": routes}
    valid, errors = validate_solution(instance, solution)

    # Separate known instance-data errors from real solver bugs
    infeasible_set = instance.get("_infeasible_customers", set())
    known_errors = []
    real_errors = []
    for e in errors:
        # Check if the error mentions one of the known infeasible customers
        is_known = any(f"customer {cid}" in e for cid in infeasible_set)
        if is_known:
            known_errors.append(e)
        else:
            real_errors.append(e)

    if real_errors:
        # Genuine solver bug — print clearly but still log
        print(f"[{label}] VALIDATION FAILED (solver errors):")
        for e in real_errors:
            print(f"  - {e}")

    if known_errors:
        # Known instance data errors — note them quietly
        print(f"[{label}] NOTE: {len(known_errors)} known instance data "
              f"error(s) for customers {sorted(infeasible_set)} "
              f"(unavoidable, noted in report)")

    # Always score — score_solution uses stochastic simulation and
    # simply counts violations, so it works fine on partially invalid solutions
    result = score_solution(instance, solution)
    result["label"] = label
    result["runtime_sec"] = round(elapsed, 1)
    result["instance"] = instance["name"]
    result["validation"] = "PASSED" if valid else (
        "KNOWN_ERRORS" if not real_errors else "FAILED"
    )

    # Always log to CSV
    _append_csv(result, log_file)

    if verbose:
        status = "✓" if valid else ("~" if not real_errors else "✗")
        print(f"[{label}] {status} "
              f"Score={result['score']:.2f} | "
              f"Vehicles={result['n_vehicles']} | "
              f"AvgTT={result['avg_travel_time']:.1f} | "
              f"Violations={result['avg_tw_violations']:.2f} | "
              f"Time={elapsed:.1f}s")

    return result


def _append_csv(result, log_file):
    """Append a result dict as a CSV row."""
    fieldnames = [
        "label", "instance", "validation", "score", "n_vehicles",
        "avg_travel_time", "std_travel_time",
        "avg_tw_violations", "n_scenarios", "runtime_sec"
    ]
    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


# ---------------------------------------------------------------------------
# Save solution JSON
# ---------------------------------------------------------------------------

def save_solution(routes, instance, team_name, output_dir="solutions"):
    """Save a solution to JSON in the solutions/ directory."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = team_name.replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"solution_{instance['name']}_{safe_name}.json"
    path = os.path.join(output_dir, filename)
    sol = {"team": team_name, "routes": routes}
    with open(path, "w") as f:
        json.dump(sol, f, indent=2)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------

def run_benchmark(instance, mode="quick", log_file="results.csv"):
    """
    Run all algorithms on the instance and produce a comparison table.

    mode="quick"    — fast smoke test (suitable for CI / development)
    mode="standard" — moderate run (a few minutes per algorithm)
    mode="full"     — long overnight run for final competition submission
    """
    build_cache(instance)
    name = instance["name"]
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name} ({instance['n_customers']} customers)")
    print(f"Mode: {mode}")
    print(f"{'='*60}\n")

    results = []
    baseline_score = {
        "istanbul_small_100": 25_127,
        "istanbul_medium_300": 67_399,
        "istanbul_full_500": 119_428,
    }.get(name, None)

    # -----------------------------------------------------------------------
    # Configuration per mode
    # -----------------------------------------------------------------------
    if mode == "quick":
        sa_iters    = 50_000
        ts_iters    = 1_000
        ma_gen      = 3
        ma_ls_iters = 2_000
        aco_ants    = 5
        aco_iters   = 5
        aco_ls      = 200
    elif mode == "standard":
        sa_iters    = 300_000
        ts_iters    = 5_000
        ma_gen      = 20
        ma_ls_iters = 3_000
        aco_ants    = 15
        aco_iters   = 50
        aco_ls      = 500
    else:  # full
        sa_iters    = 2_000_000
        ts_iters    = 30_000
        ma_gen      = 200
        ma_ls_iters = 2_000
        aco_ants    = 20
        aco_iters   = 300
        aco_ls      = 500

    # -----------------------------------------------------------------------
    # 1. Clarke-Wright (construction only)
    # -----------------------------------------------------------------------
    print("--- Clarke-Wright (construction) ---")
    from construction import clarke_wright
    cw_routes = clarke_wright(instance, lambda_param=0.85)
    r = run_experiment(
        lambda inst: clarke_wright(inst, lambda_param=0.85),
        instance, f"CW_λ0.85", log_file
    )
    results.append(r)
    save_solution(cw_routes, instance, "CW")

    # -----------------------------------------------------------------------
    # 2. SA
    # -----------------------------------------------------------------------
    print("\n--- Simulated Annealing ---")
    from sa_solver import sa_solve

    sa_routes = [None]

    def sa_runner(inst):
        sa_routes[0] = sa_solve(inst, max_iterations=sa_iters, verbose=True)
        return sa_routes[0]

    r = run_experiment(sa_runner, instance, f"SA_{sa_iters//1000}k", log_file)
    results.append(r)
    save_solution(sa_routes[0], instance, "SA")

    # -----------------------------------------------------------------------
    # 3. Tabu Search
    # -----------------------------------------------------------------------
    print("\n--- Tabu Search ---")
    from ts_solver import ts_solve

    ts_routes = [None]

    def ts_runner(inst):
        ts_routes[0] = ts_solve(inst, max_iterations=ts_iters, verbose=True)
        return ts_routes[0]

    r = run_experiment(ts_runner, instance, f"TS_{ts_iters}", log_file)
    results.append(r)
    save_solution(ts_routes[0], instance, "TS")

    # -----------------------------------------------------------------------
    # 4. MA (SA local search)
    # -----------------------------------------------------------------------
    print("\n--- Memetic Algorithm (SA) ---")
    from memetic_solver import ma_solve

    ma_sa_routes = [None]

    def ma_sa_runner(inst):
        ma_sa_routes[0] = ma_solve(
            inst, pop_size=10, n_generations=ma_gen,
            local_search="sa", ls_iterations=ma_ls_iters,
            verbose=True
        )
        return ma_sa_routes[0]

    r = run_experiment(ma_sa_runner, instance, f"MA_SA_g{ma_gen}", log_file)
    results.append(r)
    save_solution(ma_sa_routes[0], instance, "MA_SA")

    # -----------------------------------------------------------------------
    # 5. MA (TS local search)
    # -----------------------------------------------------------------------
    print("\n--- Memetic Algorithm (TS) ---")
    ma_ts_routes = [None]

    def ma_ts_runner(inst):
        ma_ts_routes[0] = ma_solve(
            inst, pop_size=10, n_generations=ma_gen,
            local_search="ts", ls_iterations=ma_ls_iters,
            verbose=True
        )
        return ma_ts_routes[0]

    r = run_experiment(ma_ts_runner, instance, f"MA_TS_g{ma_gen}", log_file)
    results.append(r)
    save_solution(ma_ts_routes[0], instance, "MA_TS")

    # -----------------------------------------------------------------------
    # 6. ACO
    # -----------------------------------------------------------------------
    print("\n--- Ant Colony Optimization ---")
    from aco_solver import aco_solve

    aco_routes = [None]

    def aco_runner(inst):
        aco_routes[0] = aco_solve(
            inst, n_ants=aco_ants, n_iterations=aco_iters,
            ls_iterations=aco_ls, apply_local_search=True,
            verbose=True
        )
        return aco_routes[0]

    r = run_experiment(aco_runner, instance, f"ACO_a{aco_ants}_i{aco_iters}",
                       log_file)
    results.append(r)
    save_solution(aco_routes[0], instance, "ACO")

    # -----------------------------------------------------------------------
    # 7. Best algorithm + Route Eliminator
    # -----------------------------------------------------------------------
    print("\n--- Route Eliminator (applied to best result) ---")
    best_result = min(results, key=lambda r: r["score"])
    best_label = best_result["label"]

    # Reload the best solution file
    best_sol_path = os.path.join(
        "solutions",
        f"solution_{name}_{best_label.split('_')[0]}.json"
    )
    if os.path.exists(best_sol_path):
        with open(best_sol_path) as f:
            best_sol = json.load(f)
        best_input_routes = best_sol["routes"]
    else:
        # Fallback: use best available routes in memory
        best_input_routes = (ma_ts_routes[0] or ma_sa_routes[0]
                             or sa_routes[0] or cw_routes)

    from route_eliminator import elimination_with_repair
    elim_routes = [None]

    def elim_runner(inst):
        routes, n = elimination_with_repair(
            best_input_routes, inst,
            sa_repair_iters=min(50_000, sa_iters // 4),
            verbose=True
        )
        elim_routes[0] = routes
        return routes

    r = run_experiment(elim_runner, instance, f"Best+Elim", log_file)
    results.append(r)
    save_solution(elim_routes[0], instance, "Best_Eliminator")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*75}")
    print(f"RESULTS SUMMARY — {name}")
    print(f"{'='*75}")
    header = (f"{'Algorithm':<20} {'Valid':>8} {'Vehicles':>8} "
              f"{'AvgTT':>10} {'Violations':>11} {'Score':>10} {'vs Baseline':>12}")
    print(header)
    print("-" * 75)

    if baseline_score:
        print(f"{'Baseline (NN+2opt)':<20} "
              f"{'—':>8} {'—':>8} {'—':>10} {'—':>11} "
              f"{baseline_score:>10,.0f} {'—':>12}")

    results = [r for r in results if r is not None]
    for r in sorted(results, key=lambda x: x["score"]):
        improvement = ""
        if baseline_score:
            pct = (baseline_score - r["score"]) / baseline_score * 100
            improvement = f"{pct:+.1f}%"
        val_symbol = ("✓" if r.get("validation") == "PASSED"
                      else "~" if r.get("validation") == "KNOWN_ERRORS"
                      else "✗")
        print(f"{r['label']:<20} "
              f"{val_symbol:>8} "
              f"{r['n_vehicles']:>8} "
              f"{r['avg_travel_time']:>10.1f} "
              f"{r['avg_tw_violations']:>11.2f} "
              f"{r['score']:>10.2f} "
              f"{improvement:>12}")

    print(f"{'='*75}")
    print(f"  ✓ = fully valid   ~ = known instance errors only   ✗ = solver bug")
    print(f"Results logged to: {log_file}")

    return results


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSE 557 Experiment Runner")
    parser.add_argument("instance", help="Path to instance JSON")
    parser.add_argument("--mode", choices=["quick", "standard", "full"],
                        default="quick", help="Benchmark mode")
    parser.add_argument("--log", default="results.csv",
                        help="Output CSV file")
    args = parser.parse_args()

    with open(args.instance) as f:
        instance = json.load(f)

    build_cache(instance)
    run_benchmark(instance, mode=args.mode, log_file=args.log)