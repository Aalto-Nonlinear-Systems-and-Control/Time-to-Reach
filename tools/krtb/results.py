import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from .simulation import simulate_trajectories


def save_krtb_results(
    system,
    config,
    initial_sets,
    target_sets,
    reach_time_bounds,
    pts_x0,
    output_dir_root,
    timings,
    verification_result,
    dt=0.01,
    max_trajectories=100,
):
    """
    Save KRTB analysis results to JSON and plot, following standard format.
    """
    # Create timestamp and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_root, "results", "krtb")
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
    plot_path = os.path.join(output_dir, f"plot_{timestamp}.png")

    # 1. Create summary dictionary
    results_data = {
        "benchmark": config["benchmark"]["name"],
        "timestamp": timestamp,
        "tool": "krtb",
        "reachable": verification_result,
        "expected_result": config["verification"]["expected_result"],
        "verification_passed": verification_result
        == config["verification"]["expected_result"],
        "time_horizon": config["verification"]["time_horizon"],
        "reach_time_bounds": [
            {"lower": b[0], "upper": b[1]} for b in reach_time_bounds
        ],
        "timings_sec": timings,
    }

    # 2. Save summary to JSON
    with open(summary_path, "w") as f:
        json.dump(results_data, f, indent=4)

    # 3. Generate and save plot
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot initial and target sets
    for s in initial_sets:
        if s["type"] == "box":
            bounds = np.array(s["bounds"])
            rect = plt.Rectangle(
                (bounds[0, 0], bounds[1, 0]),
                bounds[0, 1] - bounds[0, 0],
                bounds[1, 1] - bounds[1, 0],
                color="blue",
                alpha=0.2,
                label="Initial set",
            )
            ax.add_patch(rect)
        elif s["type"] == "level_set":
            func_str = s["function"].replace("^", "**")
            domain = s.get("domain", [[-2, 2], [-2, 2]])
            nx = np.linspace(domain[0][0], domain[0][1], 100)
            ny = np.linspace(domain[1][0], domain[1][1], 100)
            Xs, Ys = np.meshgrid(nx, ny)
            allowed_funcs = {
                k: getattr(np, k)
                for k in ["exp", "sin", "cos", "tan", "log", "sqrt", "fabs", "power"]
                if hasattr(np, k)
            }

            def level_func(*args):
                var_dict = {f"x{i+1}": args[i] for i in range(2)}
                return eval(func_str, allowed_funcs, var_dict)

            Z = np.vectorize(level_func)(Xs, Ys)
            ax.contour(Xs, Ys, Z, levels=[0], colors=["blue"], alpha=0.7)

    for s in target_sets:
        if s["type"] == "box":
            bounds = np.array(s["bounds"])
            rect = plt.Rectangle(
                (bounds[0, 0], bounds[1, 0]),
                bounds[0, 1] - bounds[0, 0],
                bounds[1, 1] - bounds[1, 0],
                color="red",
                alpha=0.2,
                label="Target set",
            )
            ax.add_patch(rect)
        elif s["type"] == "level_set":
            func_str = s["function"].replace("^", "**")
            domain = s.get("domain", [[-2, 2], [-2, 2]])
            nx = np.linspace(domain[0][0], domain[0][1], 100)
            ny = np.linspace(domain[1][0], domain[1][1], 100)
            Xs, Ys = np.meshgrid(nx, ny)
            allowed_funcs = {
                k: getattr(np, k)
                for k in ["exp", "sin", "cos", "tan", "log", "sqrt", "fabs", "power"]
                if hasattr(np, k)
            }

            def level_func(*args):
                var_dict = {f"x{i+1}": args[i] for i in range(2)}
                return eval(func_str, allowed_funcs, var_dict)

            Z = np.vectorize(level_func)(Xs, Ys)
            ax.contour(Xs, Ys, Z, levels=[0], colors=["red"], alpha=0.7)

    # Plot vector field (only for 2D systems)
    if system.dimension == 2:
        grid_x = np.linspace(-2, 2, 30)
        grid_y = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(grid_x, grid_y)
        points = np.stack([X.ravel(), Y.ravel()], axis=-1)
        vecs = np.array([system.ff(*pt) for pt in points])
        U = vecs[:, 0].reshape(X.shape)
        V = vecs[:, 1].reshape(X.shape)
        ax.streamplot(X, Y, U, V, color="gray", density=1.0, linewidth=0.5, arrowsize=1)
    elif system.dimension > 2:
        # For higher dimensional systems, we can only plot trajectories in the first 2 dimensions
        # Skip vector field plotting as it's not meaningful for projections
        pass

    # Simulate and plot trajectories
    if reach_time_bounds and len(pts_x0) > 0:
        T_max = max(
            b[1] for b in reach_time_bounds if b[1] != np.inf and b[1] is not None
        )
        if T_max > 0 and T_max != np.inf:
            num_trajectories = min(max_trajectories, len(pts_x0))
            trajectories, t_vec = simulate_trajectories(
                system, pts_x0[:num_trajectories], T_max, dt
            )
            bound_colors = plt.cm.viridis(
                np.linspace(0, 1, max(1, len(reach_time_bounds)))
            )
            for i, traj in enumerate(trajectories):
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    color="gray",
                    alpha=0.5,
                    linewidth=1,
                    zorder=1,
                )
                for j, (t0, t1) in enumerate(reach_time_bounds):
                    mask = (t_vec >= t0) & (t_vec <= t1)
                    if np.any(mask):
                        ax.plot(
                            traj[mask, 0],
                            traj[mask, 1],
                            color=bound_colors[j],
                            alpha=0.9,
                            linewidth=2,
                            zorder=2,
                        )

    # Final plot adjustments
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"KRTB Analysis: {config['benchmark']['name']}")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

    print(f"\n✓ KRTB results saved to: {output_dir}")
