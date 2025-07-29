#!/usr/bin/env python3
"""
dreach verification script for HEAT3D125 system
Simplified 5-cell heat diffusion model verification
"""

import json
import subprocess
import time
from pathlib import Path

# ==================== CONFIGURATION ====================
# Easy-to-modify section for different benchmarks

# File paths
DRH_FILE = "heat3d125.drh"
DOCKER_IMAGE = "dreach:latest"

# Results configuration
RESULTS_DIR = "results/dreach/setting_0"
RESULTS_FILE = "verification_results.json"
LOG_FILE = "dreach_output.log"
ERROR_FILE = "dreach_error.log"

# Docker configuration
DOCKER_TIMEOUT = 300  # 5 minutes timeout
DOCKER_MEMORY = "2g"  # Memory limit

# =========================================================


def ensure_results_dir():
    """Create results directory if it doesn't exist"""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def check_docker_image():
    """Check if Docker image exists"""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def run_dreach_verification():
    """Run dreach verification using Docker"""

    # Get current directory (benchmark directory)
    current_dir = Path.cwd()

    print("🔍 Starting dreach verification for HEAT3D125 system")
    print(f"📁 Working directory: {current_dir}")
    print(f"🐳 Docker image: {DOCKER_IMAGE}")
    print(f"📄 DRH file: {DRH_FILE}")

    # Check if DRH file exists
    if not Path(DRH_FILE).exists():
        raise FileNotFoundError(f"DRH file not found: {DRH_FILE}")

    # Check if Docker image exists
    if not check_docker_image():
        raise RuntimeError(
            f"Docker image '{DOCKER_IMAGE}' not found. Please build it first."
        )

    # Prepare Docker command
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        f"--memory={DOCKER_MEMORY}",
        "--volume",
        f"{current_dir}:/workspace",
        "--workdir",
        "/workspace",
        DOCKER_IMAGE,
        "bash",
        "-c",
        f"time /usr/app/dreal/bin/dReach {DRH_FILE}",
    ]

    print(f"🚀 Running: {' '.join(docker_cmd)}")

    # Run the command with timing
    start_time = time.time()

    try:
        result = subprocess.run(
            docker_cmd, capture_output=True, text=True, timeout=DOCKER_TIMEOUT
        )

        end_time = time.time()
        execution_time = end_time - start_time

        return {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": execution_time,
            "timeout": False,
        }

    except subprocess.TimeoutExpired:
        end_time = time.time()
        execution_time = end_time - start_time

        return {
            "return_code": -1,
            "stdout": "",
            "stderr": f"Process timed out after {DOCKER_TIMEOUT} seconds",
            "execution_time": execution_time,
            "timeout": True,
        }


def parse_dreach_output(stdout, stderr):
    """Parse dreach output to extract verification results"""

    # Initialize results
    verification_result = {
        "property_satisfied": None,
        "delta_sat": None,
        "counterexample_found": False,
        "raw_output": stdout,
        "raw_error": stderr,
    }

    # Parse stdout for results
    lines = stdout.split("\n")
    for line in lines:
        line = line.strip()

        # Check for delta-sat result
        if "delta-sat with delta" in line.lower():
            verification_result["delta_sat"] = True
            verification_result["property_satisfied"] = True

        # Check for unsat result
        elif "unsat" in line.lower():
            verification_result["delta_sat"] = False
            verification_result["property_satisfied"] = False

        # Check for sat result
        elif "sat" in line.lower() and "delta" not in line.lower():
            verification_result["delta_sat"] = False
            verification_result["property_satisfied"] = True
            verification_result["counterexample_found"] = True

    return verification_result


def save_results(execution_result, verification_result):
    """Save all results to JSON file"""

    # Combine results
    full_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "drh_file": DRH_FILE,
        "execution": execution_result,
        "verification": verification_result,
        "system_info": {
            "problem": "HEAT3D125 - Simplified 2-cell heat diffusion model",
            "description": "Verification of heat diffusion system safety property",
            "dimensions": 2,
            "time_bound": 5.0,
            "property": "All cells reaching 0.7°C simultaneously (unsafe condition)",
        },
    }

    # Save JSON results
    results_path = Path(RESULTS_DIR) / RESULTS_FILE
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)

    # Save raw output logs
    log_path = Path(RESULTS_DIR) / LOG_FILE
    with open(log_path, "w") as f:
        f.write(execution_result["stdout"])

    error_path = Path(RESULTS_DIR) / ERROR_FILE
    with open(error_path, "w") as f:
        f.write(execution_result["stderr"])

    print(f"📊 Results saved to: {results_path}")
    print(f"📄 Logs saved to: {log_path}")

    return full_results


def print_results_summary(results):
    """Print a summary of verification results"""

    exec_result = results["execution"]
    verif_result = results["verification"]

    print("\n" + "=" * 60)
    print("🔍 DREACH VERIFICATION SUMMARY - HEAT3D125")
    print("=" * 60)

    # Execution info
    print(f"⏱️  Execution time: {exec_result['execution_time']:.3f} seconds")
    print(f"🔢 Return code: {exec_result['return_code']}")
    print(f"⏰ Timeout: {'Yes' if exec_result['timeout'] else 'No'}")

    # Verification results
    if verif_result["property_satisfied"] is not None:
        if verif_result["property_satisfied"]:
            print("🔴 Result: UNSAFE - Target condition is REACHABLE")
            print("   ⚠️  All cells can reach 0.7°C simultaneously")
            if verif_result["delta_sat"]:
                print("   📊 Delta-satisfiable (within numerical precision)")
            else:
                print("   ✅ Exactly satisfiable")
        else:
            print("🟢 Result: SAFE - Target condition is UNREACHABLE")
            print("   ✅ All cells cannot reach 0.7°C simultaneously")
    else:
        print("❓ Result: UNKNOWN - Could not determine verification result")

    # Status indicator
    if exec_result["return_code"] == 0:
        print("✅ Verification completed successfully")
    else:
        print("❌ Verification failed or encountered errors")

    print("=" * 60)


def main():
    """Main execution function"""

    try:
        # Setup
        ensure_results_dir()

        # Run verification
        print("🏃 Starting dreach verification...")
        execution_result = run_dreach_verification()

        # Parse results
        verification_result = parse_dreach_output(
            execution_result["stdout"], execution_result["stderr"]
        )

        # Save results
        full_results = save_results(execution_result, verification_result)

        # Print summary
        print_results_summary(full_results)

        return 0

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
