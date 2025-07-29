#!/usr/bin/env python3
"""
dReach verification script for benchmarks
"""

# =============================================================================
# CONFIGURATION - Modify these values for different benchmarks
# =============================================================================
DRH_FILENAME = "nl_eig_fin_backward_rea_box.drh"  # Change this for different benchmarks
BENCHMARK_NAME = "NL_EIG_FIN_BACKWARD_REA_BOX"  # Change this for different benchmarks
SETTING_NAME = "setting_0"  # Change this for different settings
DOCKER_IMAGE = "dreach:latest"  # Docker image name
TIMEOUT_SECONDS = 300  # Timeout for verification (5 minutes)
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
    """Extract dreach execution time from bash timing output"""
    # Look for patterns like "real 0m1.234s" or "real 1.234s"
    time_patterns = [
        r"real\s+(?:(\d+)m)?([\d.]+)s",  # bash time format: real 0m1.234s
        r"(\d+\.\d+)s",  # simple seconds format
        r"(\d+\.\d+)\s*seconds",  # explicit seconds
    ]

    for pattern in time_patterns:
        match = re.search(pattern, output)
        if match:
            if len(match.groups()) == 2:  # minutes and seconds
                minutes = int(match.group(1)) if match.group(1) else 0
                seconds = float(match.group(2))
                return minutes * 60 + seconds
            else:  # just seconds
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
        result_file = (
            results_dir / f"dreach_result_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        # Display results
        print("\nResults:")
        print("  Tool: dreach")
        print(f"  Container runtime: {container_runtime}")
        print(f"  Benchmark: {BENCHMARK_NAME}")
        print(f"  Setting: {SETTING_NAME}")
        print(f"  DRH file: {DRH_FILENAME}")
        if dreach_time is not None:
            print(f"  Execution time: {dreach_time:.3f} seconds")
        else:
            print("  Execution time: Could not parse from output")

        print(f"  Return code: {result.returncode}")
        print(f"  Success: {result.returncode == 0}")
        print(f"  Results saved to: {result_file}")

        if result.stdout:
            print("\nDreach output:")
            print(result.stdout.strip())

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"dreach verification timed out after {TIMEOUT_SECONDS} seconds")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Clean up container
        subprocess.run(
            [container_runtime, "rm", "-f", container_name],
            capture_output=True,
            text=True,
        )


def main():
    """Main function"""
    print("=" * 60)
    print(f"dReach Verification - {BENCHMARK_NAME}")
    print("=" * 60)

    success = run_dreach_verification()

    print("=" * 60)
    if success:
        print("✓ dReach verification completed successfully")
    else:
        print("✗ dReach verification failed")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
