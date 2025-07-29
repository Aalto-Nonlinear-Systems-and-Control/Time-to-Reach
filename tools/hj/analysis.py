"""
Analysis functions for HJ reachability analysis.

This module provides functions for running HJ reachability analysis and checking reachability.
"""

import time
from typing import Any, Callable, Dict, List

import numpy as np
import scipy.integrate

# Optional imports for HJ reachability library
try:
    import hj_reachability as hj
    import jax
    import jax.numpy as jnp

    HJ_AVAILABLE = True
except ImportError:
    HJ_AVAILABLE = False
    print(
        "Warning: hj_reachability library not available. Some features may be limited."
    )


def run_hj_analysis(
    dynamics_func: Callable,
    initial_set: Any,
    target_set: Any,
    domain_bounds: List[List[float]],
    time_horizon: List[float],
    grid_size: int = 101,
    accuracy: str = "medium",
    reachability_type: str = "backward",
) -> Dict[str, Any]:
    """
    Run HJ_Reachability analysis using the hj_reachability library.

    This function supports both box and ball sets and computes the reachable level set.
    Use check_reachability_from_hj_values() separately to check reachability.

    Args:
        dynamics_func: System dynamics function (should be an HJ dynamics class)
        initial_set: Initial set (box bounds array or ball dict)
        target_set: Target set (box bounds array or ball dict)
        domain_bounds: Domain bounds for analysis
        time_horizon: [start_time, end_time] for analysis
        grid_size: Grid resolution
        accuracy: Solver accuracy ("low", "medium", "high", "very_high")
        reachability_type: "forward" or "backward" reachability

    Returns:
        Dictionary containing analysis results with level set computation and core HJ timing
    """
    if not HJ_AVAILABLE:
        raise ImportError("hj_reachability library is required for this function")

    # Check system dimension and warn about computational complexity
    system_dim = len(domain_bounds)
    if system_dim > 4:
        print(
            f"⚠️  WARNING: {system_dim}D system detected. HJ reachability analysis becomes computationally prohibitive for dimensions > 4."
        )
        print(
            "    This may result in out-of-memory errors or very long computation times."
        )
        print(
            "    Consider using other tools (CORA, JuliaReach, KRTB) for high-dimensional systems."
        )
    elif system_dim > 2:
        print(
            f"⚠️  WARNING: {system_dim}D system detected. This may require significant memory and computation time."
        )
        print(
            f"    Grid size will be {grid_size}^{system_dim} = {grid_size**system_dim:,} points."
        )

        # Estimate memory usage (rough approximation)
        estimated_memory_gb = (grid_size**system_dim * 8 * 2) / (
            1024**3
        )  # 8 bytes per float, 2 arrays
        print(f"    Estimated memory usage: ~{estimated_memory_gb:.1f} GB")

        if estimated_memory_gb > 16:
            print("    ⚠️  This may exceed typical system memory limits!")

        response = input("    Do you want to continue? (y/N): ")
        if response.lower() != "y":
            raise RuntimeError("User cancelled high-dimensional HJ analysis")

    if reachability_type not in ["forward", "backward"]:
        raise ValueError("reachability_type must be 'forward' or 'backward'")

    # Import get_set_bounds for bounding box computation
    from .sets import get_set_bounds

    # Start timing for core HJ computation
    hj_computation_start = time.time()

    # Create HJ grid (support arbitrary dimensions)
    lo = jnp.array([bound[0] for bound in domain_bounds])
    hi = jnp.array([bound[1] for bound in domain_bounds])
    grid_sizes = tuple([grid_size] * system_dim)

    print(f"Creating {system_dim}D grid with shape {grid_sizes}...")
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(lo, hi),
        grid_sizes,
    )

    # Create level set function based on reachability type
    if reachability_type == "backward":
        # Backward reachability: compute states that can reach target set
        reference_set = target_set
        time_start = time_horizon[1]  # Start from end time
        time_end = time_horizon[0]  # Go back to start time
        postprocessor = hj.solver.backwards_reachable_tube
    else:  # forward reachability
        # Forward reachability: compute states reachable from initial set
        reference_set = initial_set
        time_start = time_horizon[0]  # Start from start time
        time_end = time_horizon[1]  # Go forward to end time
        postprocessor = hj.solver.identity

        # Create signed distance function based on set type

    def reference_set_signed_distance(state):
        """Compute signed distance to reference set (ball, box, or union of sets)."""
        if isinstance(reference_set, dict) and reference_set.get("type") == "ball":
            # Ball set - use native HJ Ball support
            # Distance to center minus radius (negative inside, positive outside)
            center = jnp.array(reference_set["center"])
            radius = reference_set["radius"]

            distance_to_center = jnp.linalg.norm(state - center)
            return distance_to_center - radius
        elif isinstance(reference_set, dict) and reference_set.get("type") == "union":
            # Union of sets - compute minimum distance to any set
            distances = []
            for sub_set in reference_set["sets"]:
                if isinstance(sub_set, dict) and sub_set.get("type") == "ball":
                    # Ball set
                    center = jnp.array(sub_set["center"])
                    radius = sub_set["radius"]
                    distance_to_center = jnp.linalg.norm(state - center)
                    distance = distance_to_center - radius
                else:
                    # Box set
                    sub_bounds = get_set_bounds(sub_set)

                    # Check if point is inside this sub-set
                    inside_dims = []
                    dist_to_boundary_dims = []

                    for i in range(len(sub_bounds)):
                        coord = state[i]
                        coord_min, coord_max = sub_bounds[i]

                        inside_dim = (coord >= coord_min) & (coord <= coord_max)
                        inside_dims.append(inside_dim)

                        # Distance to boundary in this dimension
                        dist_to_boundary_dim = jnp.minimum(
                            coord - coord_min, coord_max - coord
                        )
                        dist_to_boundary_dims.append(dist_to_boundary_dim)

                    # Point is inside if it's inside in all dimensions
                    inside = jnp.all(jnp.array(inside_dims))

                    # Distance to boundary is minimum across all dimensions
                    dist_to_boundary = jnp.minimum.reduce(
                        jnp.array(dist_to_boundary_dims)
                    )

                    # Distance to set for points outside
                    dist_outside_dims = []
                    for i in range(len(sub_bounds)):
                        coord = state[i]
                        coord_min, coord_max = sub_bounds[i]
                        dist_dim = jnp.maximum(
                            0, jnp.maximum(coord_min - coord, coord - coord_max)
                        )
                        dist_outside_dims.append(dist_dim)

                    dist_outside = jnp.sqrt(jnp.sum(jnp.array(dist_outside_dims) ** 2))

                    # Return negative distance for inside, positive for outside
                    distance = jnp.where(inside, -dist_to_boundary, dist_outside)

                distances.append(distance)

            # For union, take minimum distance (closest to any set)
            # If any distance is negative (inside), that's the minimum
            return jnp.minimum.reduce(jnp.array(distances))
        else:
            # Box set - generalized for arbitrary dimensions
            reference_bounds = get_set_bounds(reference_set)

            # Check if point is inside reference set for all dimensions
            inside_dims = []
            dist_to_boundary_dims = []

            for i in range(len(reference_bounds)):
                coord = state[i]
                coord_min, coord_max = reference_bounds[i]

                inside_dim = (coord >= coord_min) & (coord <= coord_max)
                inside_dims.append(inside_dim)

                # Distance to boundary in this dimension
                dist_to_boundary_dim = jnp.minimum(coord - coord_min, coord_max - coord)
                dist_to_boundary_dims.append(dist_to_boundary_dim)

            # Point is inside if it's inside in all dimensions
            inside = jnp.all(jnp.array(inside_dims))

            # Distance to boundary is minimum across all dimensions
            dist_to_boundary = jnp.minimum.reduce(jnp.array(dist_to_boundary_dims))

            # Distance to reference set for points outside (generalized)
            dist_outside_dims = []
            for i in range(len(reference_bounds)):
                coord = state[i]
                coord_min, coord_max = reference_bounds[i]
                dist_dim = jnp.maximum(
                    0, jnp.maximum(coord_min - coord, coord - coord_max)
                )
                dist_outside_dims.append(dist_dim)

            dist_outside = jnp.sqrt(jnp.sum(jnp.array(dist_outside_dims) ** 2))

            # Return negative distance for inside, positive for outside
            return jnp.where(inside, -dist_to_boundary, dist_outside)

    # Vectorize the distance function for the grid (generalized for N-D)
    # Need to apply vmap for each grid dimension
    reference_distance_vmap = reference_set_signed_distance
    for _ in range(system_dim):
        reference_distance_vmap = jax.vmap(reference_distance_vmap, in_axes=0)

    # Initial values (signed distance to reference set)
    values = reference_distance_vmap(grid.states)

    # Use the dynamics function (should be an HJ dynamics class)
    dynamics = dynamics_func

    # Set up solver
    solver_settings = hj.SolverSettings.with_accuracy(
        accuracy, hamiltonian_postprocessor=postprocessor
    )

    # Run HJ reachability analysis with error handling
    print(f"Running HJ solver from time {time_start} to {time_end}...")
    print(
        "⚠️  This may take a very long time or run out of memory for high-dimensional systems..."
    )

    try:
        hj_values = hj.step(
            solver_settings, dynamics, grid, time_start, values, time_end
        )
        print("✓ HJ solver completed successfully!")
    except Exception as e:
        if "out of memory" in str(e).lower() or "memory" in str(e).lower():
            raise RuntimeError(
                f"HJ solver ran out of memory for {system_dim}D system with grid size {grid_size}. "
                f"Try reducing grid_size or using other tools."
            ) from e
        else:
            raise RuntimeError(f"HJ solver failed: {e}") from e

    # End timing for core HJ computation
    hj_computation_time = time.time() - hj_computation_start

    # Sample trajectories for visualization (not timed)
    # Get bounding boxes for trajectory sampling
    initial_bounds = get_set_bounds(initial_set)
    target_bounds = get_set_bounds(target_set)

    trajectories = _sample_trajectories_from_dynamics_class(
        dynamics, initial_bounds, target_bounds, time_horizon
    )

    # Create meshgrid for visualization (only for 2D systems)
    if system_dim == 2:
        X, Y = np.meshgrid(grid.coordinate_vectors[0], grid.coordinate_vectors[1])
        return_dict = {
            "X": X,
            "Y": Y,
            "level_set": np.array(hj_values.T),  # Transpose for matplotlib
        }
    else:
        # For higher dimensions, we can't create simple 2D meshgrids
        return_dict = {
            "level_set": np.array(hj_values),  # Raw N-D array
        }

        # Add common fields to return dictionary
    return_dict["trajectories"] = trajectories
    return_dict["grid_points"] = grid.coordinate_vectors
    return_dict["hj_grid"] = grid
    return_dict["hj_values"] = hj_values
    return_dict["initial_set"] = initial_set
    return_dict["target_set"] = target_set
    return_dict["initial_bounds"] = initial_bounds
    return_dict["target_bounds"] = target_bounds
    return_dict["domain_bounds"] = domain_bounds
    return_dict["time_horizon"] = time_horizon
    return_dict["reachability_type"] = reachability_type
    return_dict["reference_set"] = reference_set
    return_dict["n_trajectories"] = len(trajectories)
    return_dict["hj_computation_time"] = hj_computation_time

    return return_dict


def _sample_trajectories_from_dynamics_class(
    dynamics: Any,
    initial_bounds: np.ndarray,
    target_bounds: np.ndarray,
    time_horizon: List[float],
    n_samples: int = 50,
) -> List[np.ndarray]:
    """Sample trajectories from the initial set using HJ dynamics class."""
    x1_samples = np.random.uniform(
        initial_bounds[0][0], initial_bounds[0][1], n_samples
    )
    x2_samples = np.random.uniform(
        initial_bounds[1][0], initial_bounds[1][1], n_samples
    )

    trajectories = []

    for i in range(n_samples):
        x0 = jnp.array([x1_samples[i], x2_samples[i]])

        # Integrate trajectory using the dynamics class
        t_span = time_horizon
        t_eval = np.linspace(t_span[0], t_span[1], 100)

        def dynamics_func(t, x):
            """Convert HJ dynamics to scipy.integrate format."""
            state = jnp.array(x)
            return np.array(dynamics.open_loop_dynamics(state, t))

        try:
            sol = scipy.integrate.solve_ivp(
                dynamics_func,
                t_span,
                np.array(x0),
                t_eval=t_eval,
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
            )

            if sol.success:
                traj = sol.y.T
                trajectories.append(traj)

        except Exception as e:
            print(f"Warning: Simulation failed for initial condition {x0}: {e}")
            continue

    return trajectories


def check_reachability_from_hj_values(
    query_set: Any,
    grid: Any,
    hj_values: Any,
    reachability_type: str = "backward",
) -> bool:
    """
    Check reachability using HJ values.

    For backward reachability: Check if query_set (initial set) intersects
    with backward reachable set (states that can reach target).

    For forward reachability: Check if query_set (target set) intersects
    with forward reachable set (states reachable from initial).

    Args:
        query_set: Set to check for intersection (box bounds array or ball dict)
        grid: HJ grid object
        hj_values: HJ reachability values
        reachability_type: "forward" or "backward" reachability

    Returns:
        True if reachable, False otherwise
    """
    if reachability_type not in ["forward", "backward"]:
        raise ValueError("reachability_type must be 'forward' or 'backward'")

    # Import get_set_bounds for bounding box computation
    from .sets import get_set_bounds

    # Get bounding box for sampling
    query_bounds = get_set_bounds(query_set)

    # Sample points from query set
    n_samples = 20
    x1_samples = np.linspace(query_bounds[0][0], query_bounds[0][1], n_samples)
    x2_samples = np.linspace(query_bounds[1][0], query_bounds[1][1], n_samples)

    # Check if any point in query set is in reachable set
    for x1 in x1_samples:
        for x2 in x2_samples:
            # For ball sets, also check if the point is actually inside the ball
            if isinstance(query_set, dict) and query_set.get("type") == "ball":
                center = np.array(query_set["center"])
                radius = query_set["radius"]
                point = np.array([x1, x2])

                # Skip if point is outside the ball
                if np.linalg.norm(point - center) > radius:
                    continue

            # Find grid indices for this point
            i = np.argmin(np.abs(grid.coordinate_vectors[0] - x1))
            j = np.argmin(np.abs(grid.coordinate_vectors[1] - x2))

            # Check if this point is in reachable set (negative HJ value)
            if hj_values[i, j] <= 0:
                return True

    return False
