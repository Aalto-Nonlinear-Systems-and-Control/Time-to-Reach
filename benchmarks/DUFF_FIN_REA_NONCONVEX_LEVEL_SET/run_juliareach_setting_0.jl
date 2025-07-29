# JuliaReach Reachability Analysis - Duffing Oscillator System
# This script runs JuliaReach reachability analysis on the Duffing oscillator system.
# System: dx1/dt = x2, dx2/dt = -0.5*x2 - x1*(x1^2-1)
# Only supports box sets (no level sets).
# The nonlinear system is defined manually in this script.

using Pkg, Dates

# ================== SCRIPT CONFIGURATION ==================
# 1. Define the JuliaReach packages and settings
juliareach_packages = ["ReachabilityAnalysis", "LazySets", "Plots", "JSON3", "TaylorModels"]

# 2. Define the specific reachability setting for this script
setting = Dict(
    "name" => "juliareach_duffing",   # Tool-specific name for Duffing oscillator
    "tool" => "juliareach",           # Tool identifier
    "alg" => "TMJets",                # Nonlinear algorithm for Taylor models
    "δ" => 0.01,                      # Time step
    "T" => 6.0,                       # Time horizon (will be overridden by config)
    "vars" => (1, 2)                  # Variables to plot (1-based indexing)
)

# 3. Define the configuration file for the benchmark problem
config_file = "../../configs/benchmark_DUFF_FIN_REA_NONCONVEX_LEVEL_SET.json"
# ==========================================================

println("=== JuliaReach Reachability Analysis ===")
println("System: Duffing Oscillator")
println("Config: $config_file")
println("Setting: $(setting["name"])")
println("⚠️  Note: Only box sets are supported (level sets will be skipped)")
println("📍 Nonlinear system defined manually in script")

# Prepare Environment
println("\n--- Initializing Environment for JuliaReach ---")
juliareach_root_dir = joinpath(dirname(@__FILE__), "../../tools/juliareach")
include(joinpath(juliareach_root_dir, "utils.jl"))

try
    init_juliareach_environment(juliareach_packages)
catch e
    println("❌ Failed to initialize JuliaReach environment: $e")
    println("Make sure Julia packages are available")
    exit(1)
end

# Load Benchmark Configuration (for sets and parameters only)
println("\n--- Loading Benchmark Configuration ---")

try
    global config = load_benchmark_config(config_file)

    # Check compatibility (boxes only)
    if !check_sets_compatibility(config)
        println("❌ Configuration contains incompatible sets")
        exit(1)
    end

    # Define the Duffing oscillator system manually
    println("\n--- Defining Duffing Oscillator System ---")
    println("System equations:")
    println("  dx1/dt = x2")
    println("  dx2/dt = -0.5*x2 - x1*(x1^2-1)")

    # Import required packages for system definition
    using ReachabilityAnalysis

    # Define the Duffing oscillator dynamics function
    function duffing_oscillator!(dx, x, p, t)
        dx[1] = x[2]                                    # Position derivative = velocity
        dx[2] = -0.5 * x[2] - x[1] * (x[1]^2 - 1)     # Velocity derivative (Duffing equation)
    end

    # Create the nonlinear continuous system
    global sys = BlackBoxContinuousSystem(duffing_oscillator!, 2)

    println("✓ Duffing oscillator system defined successfully")

    # Create sets and parameters from config
    global X0, target = create_juliareach_sets(config)
    global params, options = setup_juliareach_params(config, setting)

    println("✓ Sets and parameters loaded from config")

catch e
    println("❌ Failed to load benchmark: $e")
    exit(1)
end

# Run Reachability Analysis
println("\n--- Starting Reachability Analysis ($(setting["name"])) ---")
start_time = time()

result = Dict()  # Initialize result dictionary

try
    # Import ReachabilityAnalysis functions
    using ReachabilityAnalysis, TaylorModels

    # Create initial value problem
    prob = InitialValueProblem(sys, X0)

    # Solve reachability problem using TMJets for nonlinear systems
    global reachable_set = solve(prob, T=params["T"], alg=TMJets(abstol=1e-12, orderT=7, orderQ=1))

    result["success"] = true
    result["error_message"] = ""
    result["num_sets"] = length(reachable_set.F)

    println("✓ Reachability analysis completed")
    println("  - Number of reachable sets: $(result["num_sets"])")

catch e
    global reachable_set = nothing
    result["success"] = false
    result["error_message"] = string(e)
    result["num_sets"] = 0
    println("❌ Reachability analysis failed: $e")
end

result["computation_time"] = time() - start_time
result["reachable_set"] = reachable_set

# Perform verification check
if result["success"]
    println("\n--- Checking Target Reachability ---")
    result["target_reached"] = check_reachability(reachable_set, target)
else
    result["target_reached"] = false
end

println("\n--- Analysis Complete ---")
println("Success: $(result["success"])")
println("Target Reached: $(result["target_reached"])")
println("Computation Time: $(round(result["computation_time"], digits=2)) s")
if result["success"]
    println("Number of Sets: $(result["num_sets"])")
end

# Save Results
println("\n--- Saving Results ---")
output_dir = dirname(@__FILE__)  # Just pass the benchmark directory

save_analysis_results(result, config, setting, X0, target, output_dir, config_file)

println("\n=== JuliaReach Analysis Complete ===")
