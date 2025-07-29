import argparse
from pathlib import Path

import numpy as np

from util_global import *

RANDOM_SEED = 0  # default random seed for reproducibility
NUM_SAMPLES = 10  # default number of samples for initial points


def check_single_benchmark_config(config_path):
    """Check a single benchmark configuration"""
    import time

    print(f"  📁 Loading config from: {config_path}")
    start_time = time.time()

    # load config
    config = load_config(config_path)
    load_time = time.time() - start_time
    print(f"  ✓ Config loaded in {load_time:.3f}s")

    # parse the dynamic
    print(f"  ⚙️  Parsing dynamics: {config['system']['type']}")
    start_time = time.time()
    dynamic = parse_dynamic(config["system"])
    parse_time = time.time() - start_time
    print(f"  ✓ Dynamics parsed in {parse_time:.3f}s")

    # get checking settings
    checking_settings = config.get("checking", {})
    num_samples = checking_settings.get("num_samples", NUM_SAMPLES)
    random_seed = checking_settings.get("random_seed", RANDOM_SEED)
    print(f"  🎯 Using {num_samples} samples with seed {random_seed}")

    # sample initial points from all initial sets
    print(f"  🎲 Sampling from {len(config['initial_sets'])} initial set(s)")
    start_time = time.time()
    all_initial_points = []
    for i, initial_set in enumerate(config["initial_sets"]):
        set_type = initial_set["type"]
        print(f"    - Set {i+1}: {set_type}")
        initial_points = sample_from_set(
            initial_set, num_samples=num_samples, random_seed=random_seed
        )
        print(f"      ✓ Sampled {len(initial_points)} points")
        all_initial_points.extend(initial_points)

    all_initial_points = np.array(all_initial_points)
    sample_time = time.time() - start_time
    print(
        f"  ✓ Total sampling completed in {sample_time:.3f}s ({len(all_initial_points)} total points)"
    )

    # simulate trajectories from the sampled initial points
    print(f"  🚀 Simulating {len(all_initial_points)} trajectories")
    time_horizon = config["verification"]["time_horizon"]
    print(f"     Time horizon: {time_horizon[0]} → {time_horizon[1]}")
    start_time = time.time()
    trajectories = simulate_trajectories(
        dynamic,
        initial_points=all_initial_points,
        time_horizon=time_horizon,
    )
    sim_time = time.time() - start_time
    print(
        f"  ✓ Trajectory simulation completed in {sim_time:.3f}s ({len(trajectories)} successful)"
    )

    # check the reachability of the trajectories
    target_sets = config["verification"]["target_sets"]
    expected_result = config["verification"]["expected_result"]
    print(f"  🎯 Checking reachability against {len(target_sets)} target set(s)")
    print(f"     Expected result: {expected_result}")
    start_time = time.time()
    check_result = check_reachability(
        trajectories,
        target_sets=target_sets,
        expected_result=expected_result,
    )
    check_time = time.time() - start_time
    print(f"  ✓ Reachability check completed in {check_time:.3f}s")
    print(
        f"     Result: {check_result['actual']} (reaching: {check_result['num_reaching']}/{check_result['total_trajectories']})"
    )

    # save a plot for this benchmark config
    print("  📊 Generating visualization...")
    start_time = time.time()
    save_plot(config_path, config, dynamic, trajectories, check_result)
    plot_time = time.time() - start_time
    print(f"  ✓ Visualization saved in {plot_time:.3f}s")

    # Summary timing
    total_time = (
        load_time + parse_time + sample_time + sim_time + check_time + plot_time
    )
    print(f"  ⏱️  Total processing time: {total_time:.3f}s")
    print(
        f"     Breakdown: Load={load_time:.3f}s, Parse={parse_time:.3f}s, Sample={sample_time:.3f}s, Sim={sim_time:.3f}s, Check={check_time:.3f}s, Plot={plot_time:.3f}s"
    )

    return check_result


def check_all_benchmark_configs(config_dir="configs"):
    """Check all benchmark configurations in the specified directory"""

    config_path = Path(config_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Config directory {config_dir} not found")

    results = {}
    config_files = list(config_path.glob("*.json"))

    print(f"Found {len(config_files)} benchmark configurations")

    for config_file in sorted(config_files):
        print(f"\nProcessing: {config_file.name}")
        try:
            result = check_single_benchmark_config(str(config_file))
            results[config_file.stem] = result
            print(
                f"✓ {config_file.stem}: {'PASSED' if result['correct'] else 'FAILED'}"
            )
        except Exception as e:
            print(f"✗ {config_file.stem}: ERROR - {str(e)}")
            results[config_file.stem] = {"error": str(e), "correct": False}

    # Summary
    total = len(results)
    passed = sum(1 for r in results.values() if r.get("correct", False))
    print(f"\n{'='*50}")
    print(f"SUMMARY: {passed}/{total} benchmarks passed")
    print(f"Success rate: {passed/total*100:.1f}%" if total > 0 else "N/A")

    return results


if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(
        description="Check benchmark configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_check_configs.py                                    # Check all configs
  python run_check_configs.py --all                             # Check all configs
  python run_check_configs.py --config benchmark_DUFF_FIN_REA_NONCONVEX_LEVEL_SET.json
  python run_check_configs.py -c benchmark_DUFF_FIN_REA_NONCONVEX_LEVEL_SET
  python run_check_configs.py --config-dir configs_old          # Check configs in different directory
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all benchmark configurations (default behavior)",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Check a specific configuration file (with or without .json extension)",
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing configuration files (default: configs)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Determine what to check
    if args.config:
        # Check single configuration
        config_file = args.config
        if not config_file.endswith(".json"):
            config_file += ".json"

        config_path = Path(args.config_dir) / config_file

        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            exit(1)

        print(f"🔍 Checking single configuration: {config_path.name}")
        print(f"📁 Using random seed: {args.seed}")
        print("=" * 50)

        try:
            result = check_single_benchmark_config(str(config_path))
            print("=" * 50)
            print(f"Result: {'PASSED' if result['correct'] else 'FAILED'}")
            if result["correct"]:
                print(f"✓ Expected: {result['expected']}, Actual: {result['actual']}")
            else:
                print(f"✗ Expected: {result['expected']}, Actual: {result['actual']}")
            print(
                f"Reaching trajectories: {result['num_reaching']}/{result['total_trajectories']}"
            )
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            exit(1)

    else:
        # Check all configurations (default behavior)
        print(f"🔍 Checking all configurations in directory: {args.config_dir}")
        print(f"📁 Using random seed: {args.seed}")
        print("=" * 50)

        try:
            results = check_all_benchmark_configs(args.config_dir)
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            exit(1)
