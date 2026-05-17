# CSE 557 — S-CVRPTW Solver Suite

**Stochastic Capacitated Vehicle Routing Problem with Time Windows — Istanbul Dataset**

> A modular, pure-Python solver suite implementing 8 metaheuristic and hybrid algorithms for the S-CVRPTW.  
> Designed for the CSE 557 *Intelligent Optimization Methods* course competition at Istanbul Sabahattin Zaim University.

---

## Table of Contents

- [Problem Definition](#problem-definition)
- [Project Architecture](#project-architecture)
- [Algorithm Details](#algorithm-details)
  - [1. Clarke-Wright Savings (CW)](#1-clarke-wright-savings-cw)
  - [2. Simulated Annealing (SA)](#2-simulated-annealing-sa)
  - [3. Tabu Search (TS)](#3-tabu-search-ts)
  - [4. Memetic Algorithm (MA)](#4-memetic-algorithm-ma)
  - [5. Ant Colony System (ACO)](#5-ant-colony-system-aco)
  - [6. Adaptive Large Neighborhood Search (ALNS/LNS)](#6-adaptive-large-neighborhood-search-alnslns)
  - [7. Route Eliminator (Post-Processor)](#7-route-eliminator-post-processor)
- [Scoring Formula](#scoring-formula)
- [Experimental Results](#experimental-results)
- [Quick Start](#quick-start)
- [Reproducing Results](#reproducing-results)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Known Instance Data Issues](#known-instance-data-issues)

---

## Problem Definition

Each instance represents a fleet of homogeneous vehicles operating from a single **depot** in Istanbul. The task is to design a set of routes that:

| Constraint | Description |
|---|---|
| **Visit all customers** | Every customer must be served exactly once |
| **Vehicle capacity** | Each route's total demand ≤ vehicle capacity |
| **Time windows** | Arrive at each customer within `[tw_open, tw_close]`; early arrival is allowed (vehicle waits) |
| **Depot hours** | Vehicles depart at `tw_open` (07:00 = 420 min) and must return by `tw_close` (20:00 = 1200 min) |
| **Stochastic traffic** | Travel times are subject to time-of-day multipliers and log-normal noise |

The objective is to **minimize** a composite score that penalizes vehicle count, travel time, and time-window violations under 100 stochastic traffic scenarios.

---

## Project Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    experimental_runner.py                      │
│         Unified benchmarking harness (CSV logging,            │
│         validation, stochastic scoring, solution JSON I/O)    │
└───────────┬───────────┬──────────┬──────────┬────────────────┘
            │           │          │          │
     ┌──────▼───┐ ┌─────▼────┐ ┌──▼────┐ ┌──▼───────────┐
     │ sa_solver │ │ ts_solver│ │  ACO  │ │  lns_solver  │
     │  (SA)     │ │  (TS)    │ │       │ │  (ALNS)      │
     └──────┬────┘ └────┬─────┘ └──┬────┘ └──────┬───────┘
            │           │          │              │
     ┌──────▼───────────▼──────────▼──────────────▼────────┐
     │              memetic_solver.py (MA)                   │
     │    GA population + SA / TS / LNS local search         │
     └──────────────────────┬──────────────────────────────┘
                            │
     ┌──────────────────────▼──────────────────────────────┐
     │              construction.py (CW)                     │
     │    Clarke-Wright Savings — initial solutions          │
     └──────────────────────┬──────────────────────────────┘
                            │
     ┌──────────────────────▼──────────────────────────────┐
     │               solver_core.py                          │
     │  Shared kernel: travel-time model, feasibility,       │
     │  insertion logic, distance cache, neighbor lists      │
     └──────────────────────┬──────────────────────────────┘
                            │
     ┌──────────────────────▼──────────────────────────────┐
     │            evaluate_solution.py (course-provided)     │
     │  Official validator + stochastic scorer (100 sims)    │
     └─────────────────────────────────────────────────────┘
```

### Module Dependency Chain

- **`solver_core.py`** — The shared foundation. Provides `det_tt()` (deterministic travel-time wrapper using the official model), route feasibility checking (`is_route_feasible`, `route_arrival_times`), insertion heuristics (`best_insertion`, `best_insertion_across_routes`), solution cost proxies, and `build_cache()` which precomputes customer lookup dicts, structurally infeasible customer detection, and sorted neighbor lists.

- **`construction.py`** — Clarke-Wright Savings algorithm. Starts with N singleton routes (one per customer), computes pairwise savings `S(i,j) = d(0,i) + d(0,j) − λ·d(i,j)`, and greedily merges routes in savings-descending order. The `λ` parameter controls cluster tightness. Also provides `clarke_wright_population()` for seeding MA/ACO with structurally diverse solutions.

- **`sa_solver.py`** / **`ts_solver.py`** / **`lns_solver.py`** / **`aco_solver.py`** — Four independent metaheuristic solvers, all consuming CW initial solutions and using `solver_core` primitives.

- **`memetic_solver.py`** — Wraps any of {SA, TS, LNS} as the local search step inside a genetic algorithm framework.

- **`route_eliminator.py`** — Post-processor that runs *after* any algorithm to squeeze out extra vehicles.

- **`experimental_runner.py`** — Orchestrates all 8 algorithm configurations, validates, scores (100 stochastic scenarios), logs to CSV, and saves solution JSONs.

---

## Algorithm Details

### 1. Clarke-Wright Savings (CW)

| Aspect | Detail |
|---|---|
| **Type** | Constructive heuristic |
| **Reference** | Clarke & Wright (1964) |
| **Key idea** | Start with N singleton routes; iteratively merge the pair (i, j) that saves the most round-trip distance |
| **λ parameter** | Controls savings formula: `S(i,j) = d(0,i) + d(0,j) − λ·d(i,j)`. λ > 1 favors tight clusters; λ < 1 favors routes far from depot |
| **Infeasible handling** | Customers unreachable as first stop are force-inserted into the best feasible position after the merge phase |
| **Population mode** | `clarke_wright_population(size=20, λ∈[0.8, 1.2])` generates diverse seeds for MA/ACO |
| **Complexity** | O(N² log N) for savings computation and sorting |

### 2. Simulated Annealing (SA)

| Aspect | Detail |
|---|---|
| **Type** | Single-solution trajectory search |
| **Acceptance** | Boltzmann criterion: P(accept) = exp(−Δ/T) |
| **Cooling** | Geometric: T ← T × α (default α = 0.9995) |
| **T₀ calibration** | Auto-calibrated by sampling 300 random moves to set 80% acceptance probability |
| **Restart** | After 50k iterations without improvement: reset to best-known, reheat to T₀/2 |
| **Move operators (7)** | `relocate` (30%), `2-opt*` (20%), `or-opt-1` (20%), `or-opt-2` (10%), `or-opt-3` (5%), `intra-2opt` (10%), `route-merge` (5%) |
| **`relocate`** | Move one customer to best position in a different route (nearest-neighbor candidate list, K=15) |
| **`2-opt*`** | Swap tails between two random routes |
| **`or-opt-k`** | Move k consecutive customers (k=1,2,3) to best position in another route |
| **`intra-2opt`** | Reverse a segment within one route (sequencing improvement) |
| **`route-merge`** | Dissolve the smallest route into another (−1 vehicle = −1000 score) |
| **Objective** | Proxy score: `n_vehicles × 1000 + total_deterministic_travel_time` |

### 3. Tabu Search (TS)

| Aspect | Detail |
|---|---|
| **Type** | Memory-based single-solution search |
| **Tabu attribute** | `(customer_id, route_fingerprint)` — prevents a customer from returning to its origin route |
| **Tenure** | Randomized ∈ [6, 12]; increased during diversification |
| **Aspiration** | Tabu override if move yields new global best |
| **Candidate generation** | Nearest-K (K=15) candidate lists keep each iteration O(N·K) instead of O(N²) |
| **Move types** | Alternates relocate (single customer) and or-opt (chain of 2) every 3 iterations; intra-2opt as fallback |
| **Diversification** | After 1000 stagnant iterations: extend tenure by +2 to push exploration |
| **Intensification** | After new best: 200 iterations of pure exploitation burst |

### 4. Memetic Algorithm (MA)

| Aspect | Detail |
|---|---|
| **Type** | Population-based evolutionary + local search hybrid |
| **Population init** | Clarke-Wright with diverse λ ∈ [0.8, 1.2], each individual refined by local search |
| **Selection** | Tournament selection (k=3) |
| **Crossover** | Route-Based Crossover (RBX): inherit best-quality route from parent1, pruned structure from parent2, greedy repair for missing customers |
| **Mutation (4 ops)** | `relocate` (35%), `merge` (35%), `shuffle` (15%), `2-opt*` (15%) — weighted toward vehicle reduction |
| **Local search** | Applied to every child after crossover+mutation. Configurable: `sa` (fast), `ts` (strong), or `lns` (powerful) |
| **Elitism** | Top 2 individuals copied unchanged to next generation |
| **Diversity control** | Coefficient of variation of proxy scores; below 0.005 → inject 3 fresh CW solutions |
| **Variants** | `MA_SA`, `MA_TS`, `MA_LNS` — same GA framework, different local search engines |

### 5. Ant Colony System (ACO)

| Aspect | Detail |
|---|---|
| **Type** | Swarm intelligence (ACS variant) |
| **Pheromone** | (N+1)×(N+1) matrix τ[i][j] with MMAS bounds [τ_min, τ_max] |
| **Heuristic** | η[i][j] = 1/(d[i][j]+ε) + 0.5/(tw_width[j]+1) — combines proximity and time-window urgency |
| **Decision rule** | With prob q₀=0.9: exploit (argmax τ^α·η^β); else probabilistic roulette wheel |
| **Local update** | After traversing edge: reduce pheromone to discourage ants from copying same path |
| **Global update** | Evaporate all edges, deposit on best-ant edges: τ ← (1−ρ)·τ + ρ/score |
| **Seeding** | Pheromone matrix initialized with CW solution edges (strong initial bias) |
| **Hybrid** | Each ant's solution refined with SA local search (optional, default on) |
| **Parameters** | α=1, β=2, ρ=0.1, q₀=0.9, 20 ants, 300 iterations (full mode) |

### 6. Adaptive Large Neighborhood Search (ALNS/LNS)

| Aspect | Detail |
|---|---|
| **Type** | Destroy-and-repair metaheuristic with adaptive operator selection |
| **Key insight** | Instead of moving one customer at a time (SA), destroy 25% of the solution and repair from scratch — enables much larger structural changes |
| **Destroy operators (4)** | `random` (exploration), `worst` (remove highest-cost customers), `shaw` (remove geographically/temporally related cluster), `smallest_route` (vehicle elimination target) |
| **Repair operators (2)** | `greedy` (cheapest feasible insertion, tw_open ordering), `regret` (insert highest-regret customer first — avoids blocking) |
| **ALNS weighting** | Operators scored per iteration: new_best=+10, improve=+3, accepted=+1, rejected=+0. Weights updated every 100 iterations with exponential decay (0.8) |
| **Acceptance** | SA criterion with auto-calibrated temperature and geometric cooling (α=0.9997) |
| **Shaw removal** | Relatedness = distance + 0.3 × tw_difference; biased selection toward most related candidates |

### 7. Route Eliminator (Post-Processor)

| Aspect | Detail |
|---|---|
| **Type** | Greedy post-processor |
| **Goal** | Reduce vehicle count after main optimization (each vehicle saved = −1000 score) |
| **Strategy** | Target smallest route → try inserting each of its customers into best positions in other routes → if all fit, commit elimination |
| **SA repair** | After successful eliminations, run short SA pass to re-optimize, then attempt further eliminations |
| **Aggressive mode** | Multi-round cycle of eliminate → SA → eliminate (3 rounds default) |
| **Applied to** | Best solution from all 7 algorithms above |

---

## Scoring Formula

```
Score = N_vehicles × 1000 + Avg_travel_time + 50 × Avg_TW_violations
```

- **N_vehicles**: number of routes (each vehicle costs 1000 points)
- **Avg_travel_time**: mean total travel time across 100 stochastic traffic scenarios
- **Avg_TW_violations**: mean count of time-window violations across 100 scenarios
- **Lower is better.** Eliminating one vehicle saves 1000 points — typically the highest-leverage optimization.

---

## Experimental Results

### Baseline Scores (Course-Provided NN + 2-opt)

| Instance | Customers | Vehicles | Score |
|---|---|---|---|
| `istanbul_small_100` | 100 | 20 | 25,127 |
| `istanbul_medium_300` | 300 | 54 | 67,399 |
| `istanbul_full_500` | 500 | 98 | 119,428 |

### Our Best Results (8 Algorithms × 3 Instances)

#### Small Instance (100 customers)

| Algorithm | Vehicles | Avg TT | Violations | Score | vs Baseline |
|---|---|---|---|---|---|
| **MA_LNS** | **18** | **3,400** | **2.40** | **21,520** | **−14.4%** |
| MA_TS | 18 | 3,449 | 1.69 | 21,533 | −14.3% |
| SA | 18 | 3,569 | 2.50 | 21,694 | −13.7% |
| LNS | 18 | 3,621 | 1.98 | 21,720 | −13.6% |
| MA_SA | 18 | 3,695 | 1.67 | 21,779 | −13.3% |
| TS | 19 | 3,587 | 1.56 | 22,665 | −9.8% |
| ACO | 18 | 4,814 | 2.37 | 22,932 | −8.7% |
| CW | 20 | 3,749 | 1.13 | 23,806 | −5.3% |

#### Medium Instance (300 customers)

| Algorithm | Vehicles | Avg TT | Violations | Score | vs Baseline |
|---|---|---|---|---|---|
| **MA_LNS** | **52** | **10,760** | **8.64** | **63,192** | **−6.2%** |
| LNS | 52 | 10,970 | 8.85 | 63,412 | −5.9% |
| MA_TS | 53 | 10,499 | 7.25 | 63,861 | −5.3% |
| MA_SA | 53 | 10,785 | 8.03 | 64,186 | −4.8% |
| SA | 53 | 11,143 | 8.20 | 64,553 | −4.2% |
| TS | 54 | 10,528 | 7.90 | 64,923 | −3.7% |
| CW | 54 | 10,742 | 7.71 | 65,128 | −3.4% |
| ACO | 54 | 10,742 | 7.71 | 65,128 | −3.4% |

#### Full Instance (500 customers) — Competition Target

| Algorithm | Vehicles | Avg TT | Violations | Score | vs Baseline |
|---|---|---|---|---|---|
| **MA_LNS** | **93** | **18,044** | **11.45** | **111,617** | **−6.5%** |
| SA | 94 | 17,476 | 10.85 | 112,018 | −6.2% |
| LNS | 93 | 18,459 | 13.24 | 112,121 | −6.1% |
| MA_SA | 94 | 17,643 | 9.78 | 112,132 | −6.1% |
| MA_TS | 95 | 17,073 | 9.14 | 112,530 | −5.8% |
| TS | 96 | 17,509 | 9.20 | 113,969 | −4.6% |
| ACO | 95 | 20,678 | 12.82 | 116,319 | −2.6% |
| CW | 98 | 17,862 | 10.50 | 116,387 | −2.5% |

> **Key findings**: **MA_LNS is the best algorithm across all three instance sizes**, achieving the lowest score on small (21,520), medium (63,192), and full (111,617) instances. The combination of population-based diversity (GA crossover + diversity injection) with powerful destroy-and-repair local search (ALNS) is the primary driver — it consistently reduces vehicle count while maintaining competitive travel times. On the full 500-customer instance, MA_LNS reduced vehicles to 93 (from baseline 98) and improved the score by 6.5%, running in ~7.9 hours.

---

## Quick Start

### Prerequisites

```
Python 3.9+
No external libraries required — all solvers use only the Python standard library.
Optional: matplotlib (for static plots), folium (for interactive maps)
```

### Running the Benchmark

```bash
# Clone the repository
git clone https://github.com/Abdelrahman-7z7/S-CVRPTW.git
cd S-CVRPTW

# Quick smoke test (all 8 algorithms, fast settings, ~5 minutes)
PYTHONPATH=src python3 src/experimental_runner.py src/data/istanbul_small_100.json --mode quick

# Standard benchmark (~30 minutes)
PYTHONPATH=src python3 src/experimental_runner.py src/data/istanbul_small_100.json --mode standard

# Full benchmark (overnight, best quality)
PYTHONPATH=src python3 src/experimental_runner.py src/data/istanbul_full_500.json --mode full
```

### Running Individual Algorithms

```bash
# Clarke-Wright construction only
PYTHONPATH=src python3 src/construction.py src/data/istanbul_small_100.json

# Simulated Annealing (500k iterations)
PYTHONPATH=src python3 src/sa_solver.py src/data/istanbul_small_100.json 500000

# Tabu Search (10k iterations)
PYTHONPATH=src python3 src/ts_solver.py src/data/istanbul_small_100.json 10000

# Memetic Algorithm with SA local search (50 generations)
PYTHONPATH=src python3 src/memetic_solver.py src/data/istanbul_small_100.json sa 50

# Memetic Algorithm with LNS local search (10 generations)
PYTHONPATH=src python3 src/memetic_solver.py src/data/istanbul_small_100.json lns 10

# ACO (20 ants, 100 iterations)
PYTHONPATH=src python3 src/aco_solver.py src/data/istanbul_small_100.json 20 100

# ALNS/LNS standalone (3000 iterations)
PYTHONPATH=src python3 src/lns_solver.py src/data/istanbul_small_100.json 3000

# Route Eliminator on an existing solution
PYTHONPATH=src python3 src/route_eliminator.py src/data/istanbul_small_100.json src/data/baseline_istanbul_small_100.json
```

### Visualization

```bash
# Static matplotlib plot
PYTHONPATH=src python3 src/visualize_solution.py src/data/istanbul_small_100.json solutions/solution_istanbul_small_100_MA_LNS.json

# Interactive OpenStreetMap (requires folium)
pip install folium
PYTHONPATH=src python3 src/interactive_map.py src/data/istanbul_small_100.json solutions/solution_istanbul_small_100_MA_LNS.json
```

---

## Reproducing Results

```bash
# Full reproduction on all three instances (run sequentially, ~24h total):

# 1. Small (100 customers) — ~20 minutes
PYTHONPATH=src python3 src/experimental_runner.py \
    src/data/istanbul_small_100.json --mode standard --log results.csv

# 2. Medium (300 customers) — ~2 hours
PYTHONPATH=src python3 src/experimental_runner.py \
    src/data/istanbul_medium_300.json --mode standard --log results.csv

# 3. Full (500 customers) — overnight
PYTHONPATH=src python3 src/experimental_runner.py \
    src/data/istanbul_full_500.json --mode full --log results.csv

# Results logged to results.csv
# Solution JSONs saved to solutions/
# Verify any solution:
PYTHONPATH=src python3 src/evaluate_solution.py \
    src/data/istanbul_full_500.json solutions/solution_istanbul_full_500_SA.json
```

### Benchmark Modes

| Mode | SA Iters | TS Iters | MA Gens | ACO (ants×iters) | LNS Iters | Approx. Time (500 cust.) |
|---|---|---|---|---|---|---|
| `quick` | 50k | 1k | 3 | 5×5 | 200 | ~5 min |
| `standard` | 300k | 5k | 20 | 15×50 | 1k | ~1 hour |
| `full` | 2M | 30k | 200 | 20×300 | 5k | ~8 hours |

---

## Dependencies

```
Python 3.9+
Standard library only (json, math, random, copy, time, csv, os, argparse, re)

Optional (visualization):
  - matplotlib   → static route plots (visualize_solution.py)
  - numpy        → used by matplotlib for colormaps
  - folium       → interactive OpenStreetMap HTML (interactive_map.py)
  - requests     → OSRM API for real road geometries in interactive maps
```

---

## File Structure

```
S-CVRPTW/
├── README.md                              # This file
├── results.csv                            # Experiment log (all runs, all instances)
│
├── src/
│   ├── solver_core.py                     # Shared kernel: travel-time, feasibility, insertion, caching
│   ├── construction.py                    # Clarke-Wright Savings (initial solutions + population seeding)
│   ├── sa_solver.py                       # Simulated Annealing (7 move operators, auto-T₀, restarts)
│   ├── ts_solver.py                       # Tabu Search (candidate lists, aspiration, diversification)
│   ├── memetic_solver.py                  # Memetic Algorithm (GA + SA/TS/LNS local search)
│   ├── aco_solver.py                      # Ant Colony System (ACS + MMAS bounds + hybrid SA)
│   ├── lns_solver.py                      # Adaptive Large Neighborhood Search (4 destroy + 2 repair ops)
│   ├── route_eliminator.py                # Post-processor: eliminate vehicles by redistribution
│   ├── experimental_runner.py             # Unified benchmark harness (CSV, validation, scoring)
│   │
│   ├── evaluate_solution.py               # Official evaluator — DO NOT MODIFY (course-provided)
│   ├── baseline_solver.py                 # Course-provided baseline (NN + 2-opt)
│   ├── visualize_solution.py              # Static matplotlib route plotter (course-provided)
│   ├── interactive_map.py                 # Interactive folium map with OSRM routing (course-provided)
│   │
│   └── data/
│       ├── istanbul_small_100.json        # 100 customers (~170 KB)
│       ├── istanbul_medium_300.json       # 300 customers (~1.5 MB)
│       ├── istanbul_full_500.json         # 500 customers (~4.0 MB, competition instance)
│       └── baseline_istanbul_small_100.json  # Course-provided baseline solution
│
├── solutions/                             # Output directory (27 solution JSONs)
│   ├── solution_istanbul_small_100_*.json      # 9 solutions (8 algorithms + Best_Eliminator)
│   ├── solution_istanbul_medium_300_*.json     # 9 solutions
│   └── solution_istanbul_full_500_*.json       # 9 solutions
│
├── plot_istanbul_small_100.png            # Static route visualization (100 customers)
├── plot_istanbul_medium_300.png           # Static route visualization (300 customers)
├── plot_istanbul_full_500.png             # Static route visualization (500 customers)
│
├── interactive_map_istanbul_small_100.html   # Interactive OpenStreetMap (100 customers)
├── interactive_map_istanbul_medium_300.html  # Interactive OpenStreetMap (300 customers)
└── interactive_map_istanbul_full_500.html    # Interactive OpenStreetMap (500 customers)
```

---

## Known Instance Data Issues

The medium (300) and full (500) instances contain a small number of **structurally infeasible customers** — customers whose time windows are physically impossible to satisfy even as the first stop on a direct depot→customer→depot route:

| Instance | Infeasible Customers | Reason |
|---|---|---|
| `istanbul_medium_300` | 64, 112, 217 | Time window opens after latest possible depot return |
| `istanbul_full_500` | 463 | Same (detected on some configurations) |

These are **instance data errors**, not solver bugs. The solver suite handles them by:

1. `build_cache()` automatically detects and flags structurally infeasible customers at startup
2. Feasibility checks assign a penalty (5000) instead of rejecting the route outright
3. `experimental_runner.py` distinguishes known-infeasible errors from genuine solver violations — affected algorithms are still marked as `PASSED`
4. The `unavoidable_customers` column in `results.csv` records which customers triggered known errors

---

*Built for the CSE 557 Intelligent Optimization Methods course — Istanbul Sabahattin Zaim University, Spring 2026.*
