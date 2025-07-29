#!/usr/bin/env python3
"""
HJ_Reachability analysis script for benchmark_DUFF_FIN_REA_BOX.
This script runs reachability analysis using HJ_Reachability methods.
"""

import sys
from pathlib import Path

# Add the tools directory to Python path
tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

from hj import (
    check_reachability_from_hj_values,
    create_hj_system,
    create_initial_set,
    create_target_set,
    load_benchmark_config,
    run_hj_analysis,
    save_hj_results,
)


def main():
    """Main function to run HJ_Reachability analysis for benchmark_DUFF_FIN_REA_BOX."""
    # Set parameters directly
    config_file = "../../configs/benchmark_DUFF_FIN_REA_BOX.json"
    setting_name = "setting_0"  # Extract from filename or set explicitly
    output_dir = f"results/hj_reachability/{setting_name}"
    grid_size = 500
    accuracy = "medium"  # HJ solver accuracy: "low", "medium", "high", "very_high"
    reachability_type = "backward"  # "forward" or "backward"

    # Get absolute paths
    script_dir = Path(__file__).parent
    config_path = (script_dir / config_file).resolve()
    output_dir = (script_dir / output_dir).resolve()

    try:
        print("=" * 60)
        print("HJ_Reachability Analysis - benchmark_DUFF_FIN_REA_BOX")
        print("=" * 60)

        # Load configuration
        print(f"Loading configuration from: {config_path}")
        config = load_benchmark_config(str(config_path))

        benchmark_name = config["benchmark"]["name"]
        print(f"Benchmark: {benchmark_name}")
        print(f"System type: {config['system']['type']}")

        # Create system dynamics
        print("Creating system dynamics...")
        dynamics_func = create_hj_system(config["system"])

        # Create initial set
        print("Setting up initial set...")
        initial_set_config = config["initial_sets"][0]  # Use first initial set
        initial_set = create_initial_set(initial_set_config)
        print(f"Initial set bounds: {initial_set}")

        # Create target set
        print("Setting up target set...")
        target_set_config = config["verification"]["target_sets"][
            0
        ]  # Use first target set
        target_set = create_target_set(target_set_config)
        print(f"Target set bounds: {target_set}")

        # Get domain bounds and time horizon
        domain_bounds = config["domain_bounds"]
        time_horizon = config["verification"]["time_horizon"]

        print(f"Domain bounds: {domain_bounds}")
        print(f"Time horizon: {time_horizon}")
        print(f"Grid size: {grid_size}")
        print(f"Solver accuracy: {accuracy}")
        print(f"Reachability type: {reachability_type}")

        # Run analysis
        print("\nRunning HJ reachability analysis...")
        results = run_hj_analysis(
            dynamics_func=dynamics_func,
            initial_set=initial_set,
            target_set=target_set,
            domain_bounds=domain_bounds,
            time_horizon=time_horizon,
            grid_size=grid_size,
            accuracy=accuracy,
            reachability_type=reachability_type,
        )

        # Check reachability based on reachability type
        if reachability_type == "backward":
            # For backward reachability: check if initial set can reach target set
            query_set = initial_set
            print("Checking if initial set can reach target set...")
        else:  # forward reachability
            # For forward reachability: check if target set can be reached from initial set
            query_set = target_set
            print("Checking if target set can be reached from initial set...")

        reachable = check_reachability_from_hj_values(
            query_set=query_set,
            grid=results["hj_grid"],
            hj_values=results["hj_values"],
            reachability_type=reachability_type,
        )

        # Display results
        print("\nAnalysis Results:")
        print(f"Reachable: {reachable}")
        print(f"Number of trajectories: {results['n_trajectories']}")
        expected_result = config.get("verification", {}).get(
            "expected_result", "unknown"
        )
        print(f"Expected result: {expected_result}")

        verification_passed = reachable == (expected_result == "reachable")
        print(f"Verification passed: {verification_passed}")

        # Display HJ computation time if available
        if "hj_computation_time" in results:
            hj_time = results["hj_computation_time"]
            print(f"\nHJ Computation Time: {hj_time:.3f}s")

        # Add reachability result to results for saving
        results["reachable"] = reachable

        # Save results
        print(f"\nSaving results to: {output_dir}")
        save_hj_results(results, benchmark_name, str(output_dir), config)

        print("\nAnalysis completed successfully!")

        # Return appropriate exit code
        return 0 if verification_passed else 1

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
