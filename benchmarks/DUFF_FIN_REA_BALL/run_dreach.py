#!/usr/bin/env python3
"""
dReach verification script for benchmarks
"""

# =============================================================================
# CONFIGURATION - Modify these values for different benchmarks
# =============================================================================
DRH_FILENAME = "duffing_fin_rea_ball.drh"  # Change this for different benchmarks
BENCHMARK_NAME = "DUFF_FIN_REA_BALL"  # Change this for different benchmarks
SETTING_NAME = "setting_0"  # Change this for different settings
DOCKER_IMAGE = "dreach:latest"  # Docker image name
TIMEOUT_SECONDS = 1200  # Timeout for verification (20 minutes)
# =============================================================================

import json
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path


def detect_container_runtime():
    """Detect available container runtime (podman or docker)"""
    system = platform.system()

    # Check for podman first (preferred on Linux)
    if system == "Linux":
        try:
            subprocess.run(["podman", "--version"], capture_output=True, check=True)
            return "podman"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Check for docker
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        return "docker"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # If neither found, default to docker and let it fail with a clear message
    return "docker"


def parse_time_output(output):
    """Parse timing output from bash time command"""
    if not output:
        return None

    # Look for real time in seconds
    real_time_pattern = r"real\s+(\d+)m([\d.]+)s"
    match = re.search(real_time_pattern, output)

    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        total_seconds = minutes * 60 + seconds
        return total_seconds

    # Alternative pattern (seconds only)
    seconds_pattern = r"real\s+([\d.]+)s"
    match = re.search(seconds_pattern, output)

    if match:
        return float(match.group(1))

    return None


def run_dreach_verification():
    """Run dreach verification and return results"""

    # Detect container runtime
    container_runtime = detect_container_runtime()
    print(f"Using container runtime: {container_runtime}")

    # Setup paths - follow the same pattern as other tools
    benchmark_dir = Path(__file__).parent
    drh_file = benchmark_dir / DRH_FILENAME

    # Create results directory structure: results/dreach/setting_0/
    results_dir = benchmark_dir / "results" / "dreach" / SETTING_NAME
    results_dir.mkdir(parents=True, exist_ok=True)

    # Container configuration
    container_name = f"dreach-{BENCHMARK_NAME}"

    print(f"Running dreach verification on {drh_file}")
    print(f"Results will be saved to: {results_dir}")

    # Check if drh file exists
    if not drh_file.exists():
        print(f"ERROR: DRH file not found - {drh_file}")
        return False

    # Clean up old container
    subprocess.run(
        [container_runtime, "rm", "-f", container_name], capture_output=True, text=True
    )

    # Run dreach with timing inside container using bash
    container_cmd = [
        container_runtime,  # Container runtime (podman or docker)
        "run",  # Run a new container
        "--name",  # Specify container name
        container_name,  # The container name variable
        "-v",  # Mount a host directory into the container
        f"{benchmark_dir}:/workspace",  # Mount path: host_dir:container_dir
        "-w",  # Set working directory inside the container
        "/workspace",  # The working directory in the container
        DOCKER_IMAGE,  # The Docker image to use
        "bash",  # Use bash to run the command
        "-c",  # Pass the following string as a command to bash
        f"time /usr/app/dreal/bin/dReach {DRH_FILENAME}",  # Time and run dReach
    ]

    try:
        # Execute dreach
        result = subprocess.run(
            container_cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
        )

        # Parse dreach execution time from timing output
        # bash time output goes to stderr
        dreach_time = parse_time_output(result.stderr)

        # If parsing failed, try stdout
        if dreach_time is None:
            dreach_time = parse_time_output(result.stdout)

        # Prepare results
        timestamp = datetime.now()
        results = {
            "timestamp": timestamp.isoformat(),
            "dreach_execution_time_seconds": dreach_time,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
            "tool": "dreach",
            "setting": SETTING_NAME,
            "benchmark": BENCHMARK_NAME,
            "drh_file": DRH_FILENAME,
            "container_runtime": container_runtime,
        }

        # Save results with consistent naming
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"summary_{timestamp_str}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        # Display results
        print("\n" + "=" * 60)
        print("dReach Verification Results")
        print("=" * 60)
        print(f"Return code: {result.returncode}")
        print(f"Success: {results['success']}")

        if dreach_time is not None:
            print(f"Execution time: {dreach_time:.3f} seconds")
        else:
            print("Execution time: Could not parse")

        print("Output preview:")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[:500] + ("..." if len(result.stdout) > 500 else ""))

        if result.stderr:
            print("STDERR:")
            print(result.stderr[:500] + ("..." if len(result.stderr) > 500 else ""))

        return results["success"]

    except subprocess.TimeoutExpired:
        print(f"ERROR: dreach verification timed out after {TIMEOUT_SECONDS} seconds")
        return False
    except Exception as e:
        print(f"ERROR: Failed to run dreach verification: {e}")
        return False
    finally:
        # Clean up container
        subprocess.run(
            [container_runtime, "rm", "-f", container_name], capture_output=True
        )


def main():
    """Main function"""
    print("=" * 60)
    print(f"dReach Verification - {BENCHMARK_NAME}")
    print("=" * 60)

    success = run_dreach_verification()

    if success:
        print("\n✓ Verification completed successfully")
        exit(0)
    else:
        print("\n✗ Verification failed")
        exit(1)


if __name__ == "__main__":
    main()
