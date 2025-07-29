#!/usr/bin/env python3
"""
KRTB Benchmark Runner for benchmark_DUFF_FIN_REA_DISJOINT_BOX

This script runs the KRTB (Koopman Reach-Time Bounds) method for reachability analysis.
"""

import os
import sys
import time

import numpy as np

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

# Import from the new modular krtb package
from krtb import (
    compute_reach_time_bounds,
    create_system_from_config,
    init_principle_eigenfunction_eval,
    load_benchmark_config,
    path_integral_eigeval,
    sample_sets,
    save_krtb_results,
    verify_reachability_with_simulation,
)

# --- Verification Parameters ---
NUM_SIM_SAMPLES = 1000  # Number of trajectories to simulate for verification


def main():
    """Main function for benchmark_DUFF_FIN_REA_BOX KRTB analysis."""

    # Parameters
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "configs",
        "benchmark_DUFF_FIN_REA_DISJOINT_BOX.json",
    )
    equilibrium = np.array([1, 0])  # Equilibrium point
    num_samples_initial = 1000  # Reduced for debugging
    num_samples_target = 1000  # Reduced for debugging

    print("=" * 60)
    print("KRTB (Koopman Reach-Time Bounds) Analysis")
    print("=" * 60)
    print(f"Config file: {config_path}")
    print(f"Equilibrium point: {equilibrium}")
    print(f"Samples - Initial: {num_samples_initial}, Target: {num_samples_target}")
    print()

    import sys

    sys.stdout.flush()  # Force output

    try:
        # Step 1: Load configuration
        print("Step 1: Loading configuration...")
        sys.stdout.flush()
        config = load_benchmark_config(config_path)
        print(f"✓ Loaded benchmark: {config['benchmark']['name']}")

        # Step 2: Create system
        print("Step 2: Creating system...")
        sys.stdout.flush()
        system = create_system_from_config(config["system"])
        print(
            f"✓ Created {config['system']['type']} system (dimension: {system.dimension})"
        )

        # Step 3: Sample sets
        print("Step 3: Sampling sets...")
        sys.stdout.flush()
        initial_sets = config["initial_sets"]
        target_sets = config["verification"]["target_sets"]

        initial_samples = sample_sets(initial_sets, num_samples_initial, random_seed=7)
        target_samples = sample_sets(target_sets, num_samples_target, random_seed=7)

        print(f"✓ Sampled {len(initial_samples)} initial points")
        print(f"✓ Sampled {len(target_samples)} target points")

        if len(initial_samples) == 0 or len(target_samples) == 0:
            print("✗ No valid samples found!")
            return

        # Step 4: Compute principal eigenfunction
        print("Step 4: Computing principal eigenfunction...")
        sys.stdout.flush()
        (eigenvalues, eigenvectors), g_symbolic = init_principle_eigenfunction_eval(
            system, equilibrium, False
        )
        print(f"✓ Principal eigenvalue: {eigenvalues}")

        # Step 5: Evaluate eigenfunction on sets via path integral
        print("Step 5: Evaluating eigenfunction on sets...")
        start_time_path_integral = time.time()
        T = 100  # Time steps for path integral

        print("  Evaluating on initial set...")
        ef_initial = path_integral_eigeval(
            system,
            equilibrium,
            T,
            initial_samples,
            eigenvalues,
            eigenvectors,
            g_symbolic,
        )

        print("  Evaluating on target set...")
        ef_target = path_integral_eigeval(
            system,
            equilibrium,
            T,
            target_samples,
            eigenvalues,
            eigenvectors,
            g_symbolic,
        )
        time_path_integral = time.time() - start_time_path_integral
        print(f"✓ Eigenfunction evaluated in {time_path_integral:.4f} seconds")

        # Step 6: Compute reach-time bounds
        print("Step 6: Computing reach-time bounds...")
        start_time_bounds = time.time()
        # Extract final time values (last time step)
        ef_initial_final = ef_initial[:, :, -1].T
        ef_target_final = ef_target[:, :, -1].T

        time_intervals, status = compute_reach_time_bounds(
            ef_initial_final, ef_target_final, eigenvalues
        )
        time_bounds = time.time() - start_time_bounds
        print(f"✓ Reach-time bounds computed in {time_bounds:.4f} seconds")
        print(f"Status: {status}")
        print()

        # Step 7: Display results
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Principal eigenvalue: {eigenvalues}")
        print(f"Number of computed intervals: {len(time_intervals)}")
        print()

        if len(time_intervals) > 0:
            print(
                "Reach-time bounds (time intervals when trajectory is in target set):"
            )
            for i, interval in enumerate(time_intervals):
                if isinstance(interval, (tuple, list)) and len(interval) == 2:
                    t_lower, t_upper = interval
                    print(f"  Interval {i+1}: [{t_lower:.6f}, {t_upper:.6f}]")
                else:
                    print(f"  Interval {i+1}: {interval}")
        else:
            print("No reachable time intervals found.")

        print("=" * 60)

        # Step 7: Verify reachability with simulation
        verification_result = verify_reachability_with_simulation(
            system=system,
            initial_sets=initial_sets,
            target_sets=target_sets,
            reach_time_bounds=time_intervals,
            num_sim_samples=NUM_SIM_SAMPLES,
        )

        # Step 8: Save results
        timings = {
            "path_integral": time_path_integral,
            "compute_bounds": time_bounds,
        }
        output_dir_root = os.path.dirname(__file__)
        save_krtb_results(
            system=system,
            config=config,
            initial_sets=initial_sets,
            target_sets=target_sets,
            reach_time_bounds=time_intervals,
            pts_x0=initial_samples,
            output_dir_root=output_dir_root,
            timings=timings,
            verification_result=verification_result,
            dt=0.01,
            max_trajectories=100,
        )

        print("=" * 60)
        print("SIMULATION-BASED VERIFICATION")
        print("=" * 60)
        print(f"Final Verification Result: {verification_result.upper()}")
        print("=" * 60)

    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
