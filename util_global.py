# Global utility functions for benchmark checking
"""
Benchmark Configuration Checking Framework

This module provides utilities for:
1. Loading and parsing benchmark configurations
2. Simulating dynamical system trajectories
3. Checking reachability against target sets
4. Visualizing results

Key Design Principles:
- Modular functions with clear interfaces
- Support for different system types (linear, nonlinear)
- Support for different set types (box, level_set)
- Comprehensive error handling and validation
- Rich visualization capabilities

Usage:
    from util_global import *

    config = load_config("benchmark_00.json")
    dynamics = parse_dynamic(config["system"])
    trajectories = simulate_trajectories(dynamics, ...)
    result = check_reachability(trajectories, ...)
    save_plot(config_path, config, dynamics, trajectories, result)
"""

import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# Configuration validation and constants
SUPPORTED_SYSTEM_TYPES = ["linear", "nonlinear"]
SUPPORTED_SET_TYPES = ["box", "level_set"]
SUPPORTED_RESULTS = ["reachable", "unreachable"]
DEFAULT_NUM_TRAJECTORIES = 20
DEFAULT_OUTPUT_DIR = "benchmark_visualization"


def load_config(config_path):
    """
    Load benchmark configuration from JSON file

    Args:
        config_path (str): Path to the JSON configuration file

    Returns:
        dict: Parsed configuration dictionary
    """
    with open(config_path, "r") as f:
        return json.load(f)


def parse_dynamic(system_config):
    """
    Parse system dynamics configuration and return a dynamics function

    Args:
        system_config (dict): System configuration containing type and dynamics

    Returns:
        callable: Function that computes dx/dt = f(t, x) for integration

    Example:
        For a nonlinear system with equations ["1.5*x1 - x1*x2", "-3*x2 + x1*x2"]
        Returns a function that can be used with scipy.integrate.solve_ivp
    """
    system_type = system_config["type"]
    dynamics = system_config["dynamics"]

    if system_type == "linear":
        if "A" in dynamics:
            # Linear system: dx/dt = A * x, we only consider autonomous systems
            A = np.array(dynamics["A"])

            def linear_dynamics(t, x):
                return A @ np.array(x)

            return linear_dynamics

        else:
            raise ValueError("Linear system must have either 'A' matrix or 'equations'")

    elif system_type == "nonlinear":
        # Nonlinear system with string equations
        equations = dynamics["equations"]

        # Pre-compile lambdified functions for efficiency
        compiled_functions = []
        n_dims = len(equations)  # Assume number of equations equals dimensions
        var_names = [f"x{j+1}" for j in range(n_dims)]
        symbols = [sp.Symbol(name) for name in var_names]

        for equation in equations:
            try:
                # Parse equation and convert ^ to **
                expr_str = equation.replace("^", "**")
                expr = sp.sympify(expr_str)
                func = sp.lambdify(symbols, expr, modules=["numpy"])
                compiled_functions.append(func)
            except Exception as e:
                print(
                    f"Warning: Failed to compile equation '{equation}' with sympy: {e}"
                )
                # Store the string equation for fallback
                compiled_functions.append(equation)

        def nonlinear_dynamics(t, x):
            dx = np.zeros(len(x))

            for i, func_or_eq in enumerate(compiled_functions):
                try:
                    if callable(func_or_eq):
                        # Use pre-compiled lambdified function
                        dx[i] = func_or_eq(*x)
                    else:
                        # Fallback to string replacement for this equation
                        expr = func_or_eq
                        for j in range(len(x)):
                            var_name = f"x{j+1}"
                            expr = expr.replace(var_name, str(x[j]))
                        expr = expr.replace("^", "**")
                        dx[i] = eval(expr)
                except Exception as e:
                    print(f"Warning: Error evaluating equation {i}: {e}")
                    dx[i] = 0.0

            return dx

        return nonlinear_dynamics

    else:
        raise ValueError(f"Unsupported system type: {system_type}")


def sample_from_set(set_spec, num_samples, random_seed=None):
    """
    Sample points from a given set specification

    Args:
        set_spec (dict): Set specification with 'type' and parameters
        num_samples (int): Number of samples to generate
        random_seed (int, optional): Random seed for reproducibility

    Returns:
        np.ndarray: Array of sampled points (shape: [num_samples, n_dims])
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if set_spec["type"] == "box":
        return _sample_from_box(set_spec["bounds"], num_samples)

    elif set_spec["type"] == "level_set":
        # Domain is required for level sets
        if "domain" not in set_spec:
            raise ValueError("Level set definition must include a 'domain' property")
        domain = set_spec["domain"]
        return _sample_from_level_set(set_spec["function"], domain, num_samples)

    else:
        raise ValueError(f"Unsupported set type: {set_spec['type']}")


def simulate_trajectories(dynamics_func, initial_points, time_horizon):
    """
    Simulate trajectories from given initial points

    Args:
        dynamics_func (callable): Function that computes dx/dt = f(t, x)
        initial_points (np.ndarray): Array of initial conditions (shape: [num_points, n_dims])
        time_horizon (list): [start_time, end_time] for simulation

    Returns:
        list: List of trajectory dictionaries, each containing:
            - 'time': time points array
            - 'states': state values array (shape: [n_states, n_time_points])
            - 'initial': initial condition used
    """
    trajectories = []
    t_span = time_horizon
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    failed_count = 0
    print(f"    🔄 Starting simulation of {len(initial_points)} trajectories...")

    for i, initial_point in enumerate(initial_points):
        if (i + 1) % 20 == 0 or i == 0:  # Progress every 20 trajectories or first one
            print(f"      Progress: {i+1}/{len(initial_points)} trajectories")

        try:
            # Simulate trajectory using solve_ivp
            sol = solve_ivp(
                dynamics_func,
                t_span,
                initial_point,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-8,
                atol=1e-10,
            )

            if sol.success:
                trajectories.append(
                    {"time": sol.t, "states": sol.y, "initial": initial_point}
                )
            else:
                failed_count += 1
                if failed_count <= 3:  # Only show first few failures
                    print(
                        f"      ⚠️  Trajectory {i+1} simulation failed for initial condition {initial_point}"
                    )

        except Exception as e:
            failed_count += 1
            if failed_count <= 3:  # Only show first few errors
                print(
                    f"      ❌ Trajectory {i+1} simulation error for initial condition {initial_point}: {e}"
                )

    if failed_count > 3:
        print(f"      ⚠️  ... and {failed_count - 3} more failures (suppressed)")

    success_rate = len(trajectories) / len(initial_points) * 100
    print(
        f"    ✓ Simulation completed: {len(trajectories)}/{len(initial_points)} successful ({success_rate:.1f}%)"
    )

    return trajectories


def check_reachability(trajectories, target_sets, expected_result):
    """
    Check if trajectories reach target sets and validate against expected result

    Args:
        trajectories (list): List of trajectory dictionaries from simulate_trajectories
        target_sets (list): List of target set specifications
        expected_result (str): Expected result ("reachable" or "unreachable")

    Returns:
        dict: Result dictionary containing:
            - 'reached': boolean indicating if any trajectory reached target
            - 'correct': boolean indicating if result matches expectation
            - 'reaching_trajectories': list of indices of trajectories that reached target
            - 'expected': the expected result
            - 'actual': the actual result

    Example:
        For box target: check if trajectory enters the box bounds
        For level set target: check if trajectory satisfies the constraint
    """
    reached = False
    reaching_trajectories = []

    print(
        f"    🎯 Checking {len(trajectories)} trajectories against {len(target_sets)} target set(s)"
    )

    for target_idx, target_set in enumerate(target_sets):
        print(f"      Target set {target_idx + 1}: {target_set['type']}")

    for traj_idx, trajectory in enumerate(trajectories):
        if (traj_idx + 1) % 50 == 0 or traj_idx == 0:  # Progress every 50 trajectories
            print(
                f"      Progress: checking trajectory {traj_idx+1}/{len(trajectories)}"
            )

        states = trajectory["states"]
        n_time_points = states.shape[1]

        # Check if this trajectory reaches any target set
        trajectory_reaches_target = False

        for target_set in target_sets:
            if target_set["type"] == "box":
                bounds = target_set["bounds"]

                # Check each point along the trajectory
                for t_idx in range(n_time_points):
                    point = [states[i, t_idx] for i in range(states.shape[0])]

                    if _point_in_box(point, bounds):
                        trajectory_reaches_target = True
                        break

            elif target_set["type"] == "level_set":
                function_str = target_set["function"]

                # Vectorized level set evaluation for efficiency
                try:
                    # Create lambdified function once for all trajectory points
                    func, _ = _create_level_set_function(function_str, ["x1", "x2"])

                    # Extract all trajectory points (x1, x2 coordinates)
                    x1_points = states[0, :]  # All x1 coordinates along trajectory
                    x2_points = states[1, :]  # All x2 coordinates along trajectory

                    # Vectorized evaluation of the level set function
                    func_values = func(x1_points, x2_points)

                    # Check if any point satisfies f(x) <= 0
                    if np.any(func_values <= 0):
                        trajectory_reaches_target = True
                        break

                except Exception as e:
                    # Fallback to point-by-point evaluation if vectorized fails
                    print(
                        f"      ⚠️  Vectorized level set evaluation failed, using fallback: {e}"
                    )
                    for t_idx in range(n_time_points):
                        point = [states[i, t_idx] for i in range(states.shape[0])]
                        if _point_in_level_set(point, function_str):
                            trajectory_reaches_target = True
                            break

            if trajectory_reaches_target:
                break

        if trajectory_reaches_target:
            reached = True
            reaching_trajectories.append(traj_idx)

    # Determine actual result
    actual_result = "reachable" if reached else "unreachable"

    # Check if result matches expectation
    correct = actual_result == expected_result

    print("    ✓ Reachability analysis complete:")
    print(f"      - Reaching trajectories: {len(reaching_trajectories)}")
    print(f"      - Actual result: {actual_result}")
    print(f"      - Expected: {expected_result}")
    print(f"      - Match: {'✓' if correct else '✗'}")

    return {
        "reached": reached,
        "correct": correct,
        "reaching_trajectories": reaching_trajectories,
        "expected": expected_result,
        "actual": actual_result,
        "num_reaching": len(reaching_trajectories),
        "total_trajectories": len(trajectories),
    }


def save_plot(config_path, config, dynamic, trajectories, check_result):
    """
    Create and save visualization for a benchmark configuration

    Args:
        config_path (str): Path to the configuration file (for naming output)
        config (dict): Full configuration dictionary
        dynamic (callable): Dynamics function for vector field visualization
        trajectories (list): Simulated trajectory data
        check_result (dict): Result from reachability checking

    Returns:
        list: List of paths to saved plot files

    Visualization includes:
        - Multiple phase plots for each vis_dims pair
        - Time series plot showing all dimensions over time
        - Vector field/streamlines of the dynamical system
        - Initial sets (colored regions)
        - Target sets (colored regions)
        - Simulated trajectories (colored by reachability)
    """
    import matplotlib.pyplot as plt

    # Create output directory if it doesn't exist
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    benchmark_name = get_benchmark_name(config_path)

    # Get checking settings
    checking_settings = config.get("checking", {})
    all_vis_dims = checking_settings.get("vis_dims", [[0, 1]])  # Get all pairs
    full_domain = checking_settings.get("domain", [[-5, 5], [-5, 5]])

    # Determine system dimension from config
    system_dim = len(config["system"]["dynamics"]["equations"])

    saved_files = []

    # Create phase plots for each vis_dims pair
    for i, vis_dims in enumerate(all_vis_dims):
        # Extract plot domain for the visualization dimensions
        plot_domain = [full_domain[vis_dims[0]], full_domain[vis_dims[1]]]

        # Create figure for this dimension pair
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Set plot domain
        ax.set_xlim(plot_domain[0])
        ax.set_ylim(plot_domain[1])

        # Plot vector field
        _plot_vector_field(ax, dynamic, plot_domain, vis_dims, system_dim=system_dim)

        # Plot initial sets
        for j, initial_set in enumerate(config["initial_sets"]):
            # Unique label for each initial set
            set_label = (
                f"Initial Set {j+1}"
                if len(config["initial_sets"]) > 1
                else "Initial Set"
            )
            _plot_set(
                ax, initial_set, vis_dims, color="green", alpha=0.3, label=set_label
            )

        # Plot target sets
        for j, target_set in enumerate(config["verification"]["target_sets"]):
            # Unique label for each target set
            set_label = (
                f"Target Set {j+1}"
                if len(config["verification"]["target_sets"]) > 1
                else "Target Set"
            )
            _plot_set(ax, target_set, vis_dims, color="red", alpha=0.3, label=set_label)

        # Plot trajectories
        reaching_indices = set(check_result["reaching_trajectories"])

        for j, traj in enumerate(trajectories):
            if len(traj["states"]) >= max(vis_dims) + 1:
                x_data = traj["states"][vis_dims[0], :]
                y_data = traj["states"][vis_dims[1], :]

                if j in reaching_indices:
                    ax.plot(
                        x_data,
                        y_data,
                        "r-",
                        alpha=0.7,
                        linewidth=0.5,
                        label=(
                            "Reaching Trajectory" if j == min(reaching_indices) else ""
                        ),
                    )
                    # Mark initial point for reaching trajectories
                    ax.plot(x_data[0], y_data[0], "ro", markersize=4)
                else:
                    ax.plot(
                        x_data,
                        y_data,
                        "b-",
                        alpha=0.5,
                        linewidth=0.3,
                        label=(
                            "Non-reaching Trajectory"
                            if j == 0 and 0 not in reaching_indices
                            else ""
                        ),
                    )
                    # Mark initial point for non-reaching trajectories
                    ax.plot(x_data[0], y_data[0], "bo", markersize=3)

        # Set labels and title
        ax.set_xlabel(f"x{vis_dims[0]+1}")
        ax.set_ylabel(f"x{vis_dims[1]+1}")

        result_text = f"{'✓' if check_result['correct'] else '✗'}"
        title = f"{benchmark_name} (x{vis_dims[0]+1}-x{vis_dims[1]+1}): {check_result['actual']} (expected: {check_result['expected']}) {result_text}"
        ax.set_title(title)

        # Add legend
        ax.legend(loc="upper right")

        # Add grid
        ax.grid(True, alpha=0.3)

        # Save phase plot
        output_path = (
            output_dir / f"{benchmark_name}_phase_x{vis_dims[0]+1}_x{vis_dims[1]+1}.png"
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_files.append(str(output_path))

    # Create time series plot for all dimensions
    fig, axes = plt.subplots(system_dim, 1, figsize=(12, 3 * system_dim), sharex=True)
    if system_dim == 1:
        axes = [axes]  # Ensure axes is always a list

    reaching_indices = set(check_result["reaching_trajectories"])

    for dim in range(system_dim):
        for j, traj in enumerate(trajectories):
            if j in reaching_indices:
                axes[dim].plot(
                    traj["time"],
                    traj["states"][dim, :],
                    "r-",
                    alpha=0.7,
                    linewidth=0.8,
                    label="Reaching Trajectory" if j == min(reaching_indices) else "",
                )
            else:
                axes[dim].plot(
                    traj["time"],
                    traj["states"][dim, :],
                    "b-",
                    alpha=0.3,
                    linewidth=0.5,
                    label=(
                        "Non-reaching Trajectory"
                        if j == 0 and 0 not in reaching_indices
                        else ""
                    ),
                )

        axes[dim].set_ylabel(f"x{dim+1}")
        axes[dim].grid(True, alpha=0.3)
        if dim == 0:
            axes[dim].legend(loc="upper right")

    axes[-1].set_xlabel("Time")
    result_text = f"{'✓' if check_result['correct'] else '✗'}"
    fig.suptitle(
        f"{benchmark_name} - Time Series: {check_result['actual']} (expected: {check_result['expected']}) {result_text}"
    )

    # Save time series plot
    time_output_path = output_dir / f"{benchmark_name}_timeseries.png"
    plt.savefig(time_output_path, dpi=150, bbox_inches="tight")
    plt.close()
    saved_files.append(str(time_output_path))

    print(f"Visualizations saved:")
    for path in saved_files:
        print(f"  - {path}")

    return saved_files
    print(f"  - Simulated {len(trajectories)} trajectories")
    print(
        f"  - Result: {check_result['actual']} (expected: {check_result['expected']})"
    )
    print(f"  - Correct: {check_result['correct']}")
    print(
        f"  - Reaching trajectories: {check_result['num_reaching']}/{check_result['total_trajectories']}"
    )

    # Add timing info for plot components
    print("  📊 Plot components:")
    print("     - Vector field: ✓")
    print(f"     - Initial sets: {len(config['initial_sets'])} set(s)")
    print(f"     - Target sets: {len(config['verification']['target_sets'])} set(s)")
    print(f"     - Trajectories: {len(trajectories)} plotted")

    return str(output_path)


# Helper functions for implementation
def _create_level_set_function(function_str, var_names=None):
    """
    Create a lambdified function from a string expression using sympy

    Args:
        function_str (str): String representation of the function (e.g., "x1^2 + x2^2 - 1")
        var_names (list, optional): Variable names to use. Defaults to ['x1', 'x2']

    Returns:
        callable: Lambdified function that can be evaluated on numpy arrays
    """
    if var_names is None:
        var_names = ["x1", "x2"]

    # Create sympy symbols
    symbols = [sp.Symbol(name) for name in var_names]

    # Parse the function string and convert ^ to ** for sympy
    expr_str = function_str.replace("^", "**")

    try:
        # Parse the expression
        expr = sp.sympify(expr_str)

        # Create lambdified function
        func = sp.lambdify(symbols, expr, modules=["numpy"])

        return func, symbols
    except Exception as e:
        raise ValueError(f"Failed to parse function '{function_str}': {e}")


def _sample_from_box(bounds, num_samples, random_seed=None):
    """
    Sample random points from a box (hyperrectangle)

    Args:
        bounds (list): List of [min, max] pairs for each dimension
        num_samples (int): Number of samples to generate
        random_seed (int, optional): Random seed for reproducibility

    Returns:
        np.ndarray: Array of sampled points (shape: [num_samples, n_dims])
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Handle edge cases
    if num_samples <= 0:
        return np.empty((0, len(bounds)))

    if not bounds:
        return np.empty((num_samples, 0))

    samples = []
    for _ in range(num_samples):
        point = []
        for bound in bounds:
            value = np.random.uniform(bound[0], bound[1])
            point.append(value)
        samples.append(point)

    result = np.array(samples)
    # Ensure the result is always 2D, even for single samples
    if result.ndim == 1:
        result = result.reshape(1, -1)
    return result


def _sample_from_level_set(function_str, bounds, num_samples, random_seed=None):
    """
    Sample points from a level set defined by f(x) <= 0

    Args:
        function_str (str): String representation of the function f(x1, x2)
        bounds (list): Bounding box to search within
        num_samples (int): Target number of samples
        random_seed (int, optional): Random seed for reproducibility

    Returns:
        np.ndarray: Array of valid points satisfying the constraint
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    print(f"        🎯 Level set sampling: target={num_samples}")
    print(f"           Function: {function_str}")
    print(f"           Domain: {bounds}")

    # Create lambdified function for the level set
    try:
        level_set_func, symbols = _create_level_set_function(function_str)
    except Exception as e:
        print(f"        ❌ Failed to create lambdified function: {e}")
        return np.array([])

    valid_points = []
    batch_size = 5000  # Process points in batches for memory efficiency
    attempts = 0

    while (
        len(valid_points) < num_samples and attempts < num_samples * 20
    ):  # Limit total attempts
        # Generate a batch of candidate points
        current_batch_size = min(
            batch_size, max(1, num_samples * 10 - attempts)
        )  # Adaptive batch size, ensure at least 1
        candidates = _sample_from_box(bounds, current_batch_size)
        attempts += current_batch_size

        # Skip if no candidates were generated
        if len(candidates) == 0:
            continue

        try:
            # Vectorized evaluation of all candidates at once
            x1_coords = candidates[:, 0]
            x2_coords = candidates[:, 1]
            func_values = level_set_func(x1_coords, x2_coords)

            # Find valid points (f(x) <= 0)
            valid_mask = func_values <= 0
            valid_candidates = candidates[valid_mask]

            # Add valid points to our collection
            for point in valid_candidates:
                if len(valid_points) >= num_samples:
                    break
                valid_points.append(point)

            if attempts % 10000 == 0:  # Progress every 10k attempts
                print(
                    f"           Progress: {len(valid_points)}/{num_samples} found after {attempts} attempts"
                )

        except Exception as e:
            # Fallback to individual point checking if vectorized evaluation fails
            print(f"           ⚠️  Vectorized evaluation failed, using fallback: {e}")
            for candidate in candidates:
                if len(valid_points) >= num_samples:
                    break
                if _point_in_level_set(candidate, function_str):
                    valid_points.append(candidate)

    success_rate = len(valid_points) / attempts * 100 if attempts > 0 else 0
    print(
        f"        ✓ Level set sampling complete: {len(valid_points)}/{num_samples} in {attempts} attempts ({success_rate:.1f}% success)"
    )

    if len(valid_points) < num_samples:
        print(
            f"        ⚠️  Warning: Only found {len(valid_points)} valid points out of {num_samples} requested"
        )

    return np.array(valid_points)


def _point_in_box(point, bounds):
    """
    Check if a point is inside a box

    Args:
        point (array-like): Point coordinates
        bounds (list): List of [min, max] pairs for each dimension

    Returns:
        bool: True if point is inside the box
    """
    for i, bound in enumerate(bounds):
        if point[i] < bound[0] or point[i] > bound[1]:
            return False
    return True


def _point_in_level_set(point, function_str):
    """
    Check if a point satisfies a level set constraint f(x) <= 0

    Args:
        point (array-like): Point coordinates
        function_str (str): String representation of the function

    Returns:
        bool: True if point satisfies the constraint
    """
    try:
        # Create lambdified function for 2D level sets
        func, _ = _create_level_set_function(function_str, ["x1", "x2"])

        # Evaluate function at the point
        # For 2D level sets, use first two coordinates
        x1, x2 = point[0], point[1]
        func_value = func(x1, x2)

        return func_value <= 0
    except Exception:
        return False


def validate_config(config):
    """
    Validate a benchmark configuration for completeness and correctness

    Args:
        config (dict): Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
        KeyError: If required keys are missing

    Returns:
        bool: True if configuration is valid
    """
    # TODO: Implement comprehensive config validation
    # 1. Check required top-level keys
    # 2. Validate system type and dynamics format
    # 3. Validate initial_sets and target_sets formats
    # 4. Check time_horizon format
    # 5. Validate expected_result value
    pass


def get_benchmark_name(config_path):
    """Extract benchmark name from config file path"""
    return Path(config_path).stem


def _plot_vector_field(
    ax, dynamics_func, domain, vis_dims, grid_density=20, system_dim=None
):
    """Plot vector field for the dynamical system"""
    x1_range, x2_range = domain
    x1_grid = np.linspace(x1_range[0], x1_range[1], grid_density)
    x2_grid = np.linspace(x2_range[0], x2_range[1], grid_density)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    DX1 = np.zeros_like(X1)
    DX2 = np.zeros_like(X2)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            try:
                # Create state vector with appropriate dimensions
                if system_dim is not None:
                    state = np.zeros(system_dim)
                else:
                    state = np.zeros(max(vis_dims) + 1)
                state[vis_dims[0]] = X1[i, j]
                state[vis_dims[1]] = X2[i, j]

                dx = dynamics_func(0, state)
                DX1[i, j] = dx[vis_dims[0]]
                DX2[i, j] = dx[vis_dims[1]]
            except Exception:
                DX1[i, j] = 0
                DX2[i, j] = 0

    # Plot streamlines
    ax.streamplot(X1, X2, DX1, DX2, density=1.5, color="gray", linewidth=0.8)


def _plot_set(ax, set_spec, vis_dims, color, alpha, label):
    """Plot a set (box or level_set) on the axes"""
    if set_spec["type"] == "box":
        bounds = set_spec["bounds"]
        if len(bounds) >= max(vis_dims) + 1:
            x_bounds = bounds[vis_dims[0]]
            y_bounds = bounds[vis_dims[1]]

            rect = patches.Rectangle(
                (x_bounds[0], y_bounds[0]),
                x_bounds[1] - x_bounds[0],
                y_bounds[1] - y_bounds[0],
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
                label=label,
            )
            ax.add_patch(rect)

    elif set_spec["type"] == "level_set":
        function_str = set_spec["function"]

        # Domain is required for level sets (consistent with sampling)
        if "domain" not in set_spec:
            raise ValueError(
                "Level set definition must include a 'domain' property for plotting"
            )

        domain = set_spec["domain"]

        # Validate domain dimensions - level sets are always 2D (x1, x2)
        if len(domain) < 2:
            raise ValueError(
                "Level set domain must have at least 2 dimensions for 2D plotting"
            )

        # For level sets, always use x1 (domain[0]) and x2 (domain[1])
        # No need to consider vis_dims as level sets are always 2D
        x1_range = domain[0]  # Domain for x1
        x2_range = domain[1]  # Domain for x2

        x1_grid = np.linspace(x1_range[0], x1_range[1], 200)
        x2_grid = np.linspace(x2_range[0], x2_range[1], 200)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)

        Z = np.zeros_like(X1)

        # Create lambdified function for efficient evaluation
        try:
            func, _ = _create_level_set_function(function_str, ["x1", "x2"])

            # Vectorized evaluation over the entire grid
            Z = func(X1, X2)

        except Exception as e:
            raise ValueError(
                f"Failed to create lambdified function for level set '{function_str}': {e}"
            )

        # Plot level set (f(x) <= 0)
        contour_set = ax.contourf(
            X1, X2, Z, levels=[float("-inf"), 0], colors=[color], alpha=alpha
        )
        ax.contour(X1, X2, Z, levels=[0], colors=[color], linewidths=2)

        # Add label manually for legend (with safety check for different matplotlib versions)
        try:
            if hasattr(contour_set, "collections") and contour_set.collections:
                contour_set.collections[0].set_label(label)
            elif hasattr(contour_set, "legend_elements"):
                # For newer matplotlib versions, use legend_elements
                pass  # The contour itself will be used for legend
        except (AttributeError, IndexError):
            # Fallback: create a dummy patch for legend
            import matplotlib.patches as mpatches

            ax.add_patch(
                mpatches.Rectangle(
                    (0, 0), 0, 0, facecolor=color, alpha=alpha, label=label
                )
            )
