"""
Results handling for HJ reachability analysis.

This module provides functions for saving and visualizing HJ reachability analysis results.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def save_hj_results(
    results: Dict[str, Any],
    benchmark_name: str,
    output_dir: str,
    config: Dict[str, Any],
) -> Dict[str, str]:
    """
    Save HJ_Reachability analysis results.

    Args:
        results: Analysis results dictionary
        benchmark_name: Name of the benchmark
        output_dir: Output directory path
        config: Original configuration
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create basic summary
    summary = {
        "benchmark": benchmark_name,
        "timestamp": timestamp,
        "reachable": results.get("reachable", "unknown"),
        "expected_result": config.get("verification", {}).get(
            "expected_result", "unknown"
        ),
        "n_trajectories": results.get("n_trajectories", 0),
        "time_horizon": results.get("time_horizon", []),
        "domain_bounds": results.get("domain_bounds", []),
        "reachability_type": results.get("reachability_type", "unknown"),
        "verification_passed": (
            results.get("reachable", False)
            == (config.get("verification", {}).get("expected_result") == "reachable")
        ),
    }

    # Add HJ computation time if available
    if "hj_computation_time" in results:
        hj_time = results["hj_computation_time"]
        summary["hj_computation_time_seconds"] = round(hj_time, 3)
        summary["hj_computation_time_formatted"] = f"{hj_time:.3f}s"

    summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Create visualization
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot domain bounds
    domain_bounds = results["domain_bounds"]
    ax.set_xlim(domain_bounds[0])
    ax.set_ylim(domain_bounds[1])

    # Plot backward reachable set (negative values show what can reach the target)
    if "level_set" in results and "X" in results and "Y" in results:
        level_set = results["level_set"]
        X = results["X"]
        Y = results["Y"]

        # Plot the backward reachable set (negative values)
        ax.contourf(
            X,
            Y,
            level_set,
            levels=[-np.inf, 0],
            colors=["lightblue"],
            alpha=0.7,
        )

        # Plot zero level set (boundary of reachable set) prominently
        ax.contour(X, Y, level_set, levels=[0], colors="blue", linewidths=2)

    # Plot initial set
    initial_set = results.get("initial_set")
    if isinstance(initial_set, dict) and initial_set.get("type") == "ball":
        # Plot as circle for Ball sets
        center = initial_set["center"]
        radius = initial_set["radius"]
        initial_circle = plt.Circle(
            (center[0], center[1]),
            radius,
            fill=False,
            edgecolor="green",
            linewidth=3,
            label="Initial Set",
        )
        ax.add_patch(initial_circle)
    else:
        # Plot as rectangle for Box sets
        initial_bounds = results["initial_bounds"]
        initial_rect = plt.Rectangle(
            (initial_bounds[0][0], initial_bounds[1][0]),
            initial_bounds[0][1] - initial_bounds[0][0],
            initial_bounds[1][1] - initial_bounds[1][0],
            fill=False,
            edgecolor="green",
            linewidth=3,
            label="Initial Set",
        )
        ax.add_patch(initial_rect)

    # Plot target set(s)
    target_set = results.get("target_set")
    if isinstance(target_set, dict) and target_set.get("type") == "union":
        # Plot multiple target sets for union
        for i, sub_target in enumerate(target_set["sets"]):
            label = (
                f"Target Set {i+1}" if i == 0 else ""
            )  # Only label the first one for legend

            if isinstance(sub_target, dict) and sub_target.get("type") == "ball":
                # Plot as circle for Ball sets
                center = sub_target["center"]
                radius = sub_target["radius"]
                target_circle = plt.Circle(
                    (center[0], center[1]),
                    radius,
                    fill=True,
                    facecolor="red",
                    alpha=0.8,
                    edgecolor="darkred",
                    linewidth=3,
                    label=label,
                )
                ax.add_patch(target_circle)
            else:
                # Plot as rectangle for Box sets
                sub_target_bounds = (
                    np.array(sub_target) if isinstance(sub_target, list) else sub_target
                )
                target_rect = plt.Rectangle(
                    (sub_target_bounds[0][0], sub_target_bounds[1][0]),
                    sub_target_bounds[0][1] - sub_target_bounds[0][0],
                    sub_target_bounds[1][1] - sub_target_bounds[1][0],
                    fill=True,
                    facecolor="red",
                    alpha=0.8,
                    edgecolor="darkred",
                    linewidth=3,
                    label=label,
                )
                ax.add_patch(target_rect)
    elif isinstance(target_set, dict) and target_set.get("type") == "ball":
        # Plot single ball target set
        center = target_set["center"]
        radius = target_set["radius"]
        target_circle = plt.Circle(
            (center[0], center[1]),
            radius,
            fill=True,
            facecolor="red",
            alpha=0.8,
            edgecolor="darkred",
            linewidth=3,
            label="Target Set",
        )
        ax.add_patch(target_circle)
    else:
        # Plot single box target set
        target_bounds = results["target_bounds"]
        target_rect = plt.Rectangle(
            (target_bounds[0][0], target_bounds[1][0]),
            target_bounds[0][1] - target_bounds[0][0],
            target_bounds[1][1] - target_bounds[1][0],
            fill=True,
            facecolor="red",
            alpha=0.8,
            edgecolor="darkred",
            linewidth=3,
            label="Target Set",
        )
        ax.add_patch(target_rect)

    ax.set_xlabel("x1 (Population 1)")
    ax.set_ylabel("x2 (Population 2)")

    # Update title to include timing information
    title = f"{benchmark_name} - HJ {results.get('reachability_type', 'Unknown').title()} Reachable Set"
    if "hj_computation_time" in results:
        title += f" (HJ Time: {results['hj_computation_time']:.2f}s)"
    ax.set_title(title)

    # Add manual legend entries for reachable set
    import matplotlib.patches as patches

    reachable_patch = patches.Patch(
        color="lightblue", alpha=0.7, label="Backward Reachable Set"
    )
    boundary_patch = patches.Patch(color="blue", label="Reachable Set Boundary")

    # Get existing legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([reachable_patch, boundary_patch])
    labels.extend(["Backward Reachable Set", "Reachable Set Boundary"])

    ax.legend(handles, labels)
    ax.grid(True, alpha=0.3)

    # Save plot
    plot_path = os.path.join(output_dir, f"plot_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Results saved to {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Plot: {plot_path}")

    # Print timing information if available
    if "hj_computation_time" in results:
        hj_time = results["hj_computation_time"]
        print("\nTiming Information:")
        print(f"  HJ solver time: {hj_time:.2f}s")

    return {"summary": summary_path, "plot": plot_path}
