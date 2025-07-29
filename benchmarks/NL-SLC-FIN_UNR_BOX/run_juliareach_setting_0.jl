# JuliaReach Reachability Analysis - Nonlinear System (NL-SLC-FIN-REA-BOX)
# This script runs JuliaReach reachability analysis on a nonlinear system from the NL-SLC-FIN-REA-BOX benchmark.
# The nonlinear system is defined manually in this script, aligned with the JSON config.

using Pkg, Dates

# ================== SCRIPT CONFIGURATION ==================
# 1. Define the JuliaReach packages and settings
juliareach_packages = ["ReachabilityAnalysis", "LazySets", "Plots", "JSON3", "TaylorModels"]

# 2. Define the specific reachability setting for this script
setting = Dict(
    "name" => "juliareach_nl_slc",    # Tool-specific name for this benchmark
    "tool" => "juliareach",           # Tool identifier
    "alg" => "TMJets",                # Nonlinear algorithm for Taylor models
    "δ" => 0.01,                      # Time step
    "T" => 6.0,                       # Time horizon (will be overridden by config)
    "vars" => (1, 2)                  # Variables to plot (1-based indexing)
)

# 3. Define the configuration file for the benchmark problem
config_file = "../../configs/benchmark_NL_SLC_FIN_UNR_BOX.json"
# ==========================================================

println("=== JuliaReach Reachability Analysis ===")
println("System: Nonlinear System (NL-SLC-FIN-UNR-BOX)")
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

    # Define the nonlinear system manually
    println("\n--- Defining Nonlinear System ---")
    println("System equations (from config):")
    println("  dx1/dt = " * config["system"]["dynamics"]["equations"][1])
    println("  dx2/dt = " * config["system"]["dynamics"]["equations"][2])

    # Import required packages for system definition
    using ReachabilityAnalysis

    # Define the system dynamics function
    function nonlinear_system!(dx, x, p, t)
        x1, x2 = x[1], x[2]
        # Common denominator for efficiency
        den = 4 * (x1^2 + x2^2)
        if den == 0.0
            # Handle singularity at the origin, though states should not be exactly zero.
            # Behavior at the origin is undefined in the model.
            # Depending on analysis needs, could be zero, Inf, or NaN.
            dx[1] = 0.0
            dx[2] = 0.0
            return
        end
        dx[1] = (-x1^3 + 4 * x1^2 * x2 - x1 * x2^2 + x1 + 4 * x2^3) / den
        dx[2] = (-4 * x1^3 - x1^2 * x2 - 4 * x1 * x2^2 - x2^3 + x2) / den
    end

    # Create the nonlinear continuous system
    global sys = BlackBoxContinuousSystem(nonlinear_system!, 2)

    println("✓ Nonlinear system defined successfully")

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
