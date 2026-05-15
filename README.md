# CSE 557 — S-CVRPTW Solver

**Istanbul Stochastic Capacitated Vehicle Routing Problem with Time Windows**

## File Structure

```
src/
├── solver_core.py        # Shared utilities, feasibility checkers, insertion logic
├── construction.py       # Clarke-Wright Savings construction algorithm
├── sa_solver.py          # Simulated Annealing (6 move operators)
├── ts_solver.py          # Tabu Search (candidate lists, aspiration, diversification)
├── memetic_solver.py     # Memetic Algorithm (GA + SA/TS local search)
├── aco_solver.py         # Ant Colony System (ACS variant, hybrid with SA)
├── route_eliminator.py   # Post-processor: eliminate vehicles by route redistribution
└── experiment_runner.py  # Unified benchmarking harness

# Provided by course:
evaluate_solution.py      # Official evaluator (DO NOT MODIFY)
```

## Dependencies

```
Python 3.9+
No external libraries required — uses only the Python standard library.
evaluate_solution.py must be in the same directory as the src/ files.
```

## Quick Start

```bash
# 1. Place all files from src/ and evaluate_solution.py in the same folder.
# 2. Place instance JSON files in the same folder.

# Run quick benchmark (all algorithms, fast settings):
python experiment_runner.py istanbul_small_100.json --mode quick

# Run standard benchmark:
python experiment_runner.py istanbul_small_100.json --mode standard

# Run individual algorithms:
python construction.py istanbul_small_100.json
python sa_solver.py istanbul_small_100.json 500000
python ts_solver.py istanbul_small_100.json 10000
python memetic_solver.py istanbul_small_100.json sa 50
python aco_solver.py istanbul_small_100.json 20 100
python route_eliminator.py istanbul_small_100.json baseline_istanbul_small_100.json
```

## Reproducing the Final Competition Solution

```bash
# Full run on the competition instance (run overnight):
python experiment_runner.py istanbul_full_500.json --mode full --log results_full.csv

# The best solution is saved automatically to:
#   solutions/solution_istanbul_full_500_Best_Eliminator.json
#
# Rename and submit:
cp solutions/solution_istanbul_full_500_Best_Eliminator.json solution_full_500.json

# Verify:
python evaluate_solution.py istanbul_full_500.json solution_full_500.json
```

## Algorithm Overview

| Module                | Algorithm             | Key Parameters                                |
| --------------------- | --------------------- | --------------------------------------------- |
| `construction.py`     | Clarke-Wright Savings | λ ∈ [0.8, 1.2]                                |
| `sa_solver.py`        | Simulated Annealing   | T₀ (auto-calibrated), α=0.9995, 500k–2M iters |
| `ts_solver.py`        | Tabu Search           | tenure ∈ [6,12], candidate list K=15          |
| `memetic_solver.py`   | Memetic Algorithm     | pop=20, gen=200, SA/TS local search           |
| `aco_solver.py`       | Ant Colony System     | 20 ants, α=1, β=2, ρ=0.1, q₀=0.9              |
| `route_eliminator.py` | Route Elimination     | Greedy + SA repair                            |

## Score Formula

```
Score = N_vehicles × 1000 + Avg_travel_time + 50 × Avg_violations
```

Lower is better. Reducing one vehicle saves 1000 points.

## Baseline Results to Beat

| Instance     | Vehicles | Score   |
| ------------ | -------- | ------- |
| small (100)  | 20       | 25,127  |
| medium (300) | 54       | 67,399  |
| full (500)   | 98       | 119,428 |
