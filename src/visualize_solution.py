"""
CSE 557 — S-CVRPTW Solution Visualizer
Plots the depot, customers, and routes on a 2D map.

Usage:
    python visualize_solution.py <instance.json> [solution.json]
"""
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Visualize S-CVRPTW Instance and Solution")
    parser.add_argument("instance", help="Path to instance JSON file")
    parser.add_argument("solution", nargs="?", default=None, help="Path to optional solution JSON file")
    args = parser.parse_args()

    instance = load_json(args.instance)
    solution = load_json(args.solution) if args.solution else None

    # Extract coordinates
    depot = instance["depot"]
    customers = {c["id"]: c for c in instance["customers"]}
    
    # Setup plot
    plt.figure(figsize=(12, 10))
    
    # Plot customers
    c_lons = [c["lon"] for c in customers.values()]
    c_lats = [c["lat"] for c in customers.values()]
    plt.scatter(c_lons, c_lats, c='gray', marker='o', s=50, label='Customers', zorder=2)
    
    # Plot customer IDs
    for cid, c in customers.items():
        plt.annotate(str(cid), (c["lon"], c["lat"]), xytext=(3, 3), 
                     textcoords="offset points", fontsize=8, color='darkslategray', zorder=3)

    # Plot depot
    plt.scatter([depot["lon"]], [depot["lat"]], c='red', marker='*', s=300, 
                edgecolors='black', label='Depot', zorder=4)
    plt.annotate("Depot", (depot["lon"], depot["lat"]), xytext=(5, 5), 
                 textcoords="offset points", fontsize=10, fontweight='bold', color='darkred', zorder=5)

    # Plot routes if solution provided
    if solution and "routes" in solution:
        routes = solution["routes"]
        colors = cm.tab20(np.linspace(0, 1, len(routes)))
        
        for ri, route in enumerate(routes):
            if not route: continue
            
            # Build sequence of coordinates: Depot -> C1 -> C2 -> ... -> Depot
            seq_lons = [depot["lon"]]
            seq_lats = [depot["lat"]]
            
            for cid in route:
                if cid in customers:
                    seq_lons.append(customers[cid]["lon"])
                    seq_lats.append(customers[cid]["lat"])
            
            seq_lons.append(depot["lon"])
            seq_lats.append(depot["lat"])
            
            # Plot the route lines
            plt.plot(seq_lons, seq_lats, c=colors[ri], linestyle='-', linewidth=2, 
                     alpha=0.7, label=f"Route {ri+1}", zorder=1)

    # Add titles and labels
    title = f"Instance: {instance['name']} ({instance['n_customers']} customers)"
    if solution:
        team_name = solution.get("team", "Unknown")
        n_routes = len(solution.get("routes", []))
        title += f"\nTeam: {team_name} | Vehicles: {n_routes}"
        
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    
    # Only show legend if not too many items
    if not solution or len(solution.get("routes", [])) <= 20:
        # Put legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
    else:
        # Simplify legend
        handles, labels = plt.gca().get_legend_handles_labels()
        base_handles = [h for h, l in zip(handles, labels) if l in ['Customers', 'Depot']]
        base_labels = [l for l in labels if l in ['Customers', 'Depot']]
        plt.legend(base_handles, base_labels, loc='best')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    out_filename = f"plot_{instance['name']}.png"
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {out_filename}")
    
    # Do not call plt.show() to prevent blocking when run in terminal

if __name__ == "__main__":
    main()
