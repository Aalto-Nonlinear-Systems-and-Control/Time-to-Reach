# JuliaReach Reachability Analysis - CP-LQR System
# This script runs JuliaReach reachability analysis on the CP-LQR 4D nonlinear control system.
# System: 4D nonlinear control system with trigonometric coupling
# Only supports box sets (no level sets).
# The nonlinear system is defined manually in this script.

using Pkg, Dates

# ================== SCRIPT CONFIGURATION ==================
# 1. Define the JuliaReach packages and settings
juliareach_packages = ["ReachabilityAnalysis", "LazySets", "Plots", "JSON3", "TaylorModels"]

# 2. Define the specific reachability setting for this script
setting = Dict(
    "name" => "juliareach_cp_lqr",   # Tool-specific name for CP-LQR
    "tool" => "juliareach",          # Tool identifier
    "alg" => "TMJets",               # Nonlinear algorithm for Taylor models
    "δ" => 0.05,                     # Time step
    "T" => 8.0,                      # Time horizon (will be overridden by config)
    "vars" => (1, 2)                 # Variables to plot (1-based indexing)
)

# 3. Define the configuration file for the benchmark problem
config_file = "../../configs/benchmark_CP_LQR_REA_UNSAFE.json"
# ==========================================================

println("=== JuliaReach Reachability Analysis ===")
println("System: CP-LQR 4D Nonlinear Control System")
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

    # Define the CP-LQR system manually
    println("\n--- Defining CP-LQR 4D Nonlinear Control System ---")
    println("System equations:")
    println("  dx1/dt = x2")
    println("  dx2/dt = (-3.16227766016837*x1 - 6.69810360685677*x2 - 5.02657663490373*x3 - 1.6165296244068*x4 + 2*(x4^2 - 9.8065*cos(x3))*sin(x3))/(2*sin(x3)^2 + 3)")
    println("  dx3/dt = x4")
    println("  dx4/dt = (2*x4^2*sin(x3)*cos(x3) + (-3.16227766016837*x1 - 6.69810360685677*x2 - 5.02657663490373*x3 - 1.6165296244068*x4)*cos(x3) - 49.0325*sin(x3))/(2*sin(x3)^2 + 3)")

    # Import required packages for system definition
    using ReachabilityAnalysis

    # Define the CP-LQR dynamics function
    function cp_lqr_dynamics!(dx, x, p, t)
        # System parameters
        a11 = -3.16227766016837
        a12 = -6.69810360685677
        a13 = -5.02657663490373
        a14 = -1.6165296244068
        g = 9.8065

        # Extract state variables
        x1, x2, x3, x4 = x[1], x[2], x[3], x[4]

        # Compute common terms
        sin_x3 = sin(x3)
        cos_x3 = cos(x3)
        sin_x3_squared = sin_x3^2
        denominator = 2 * sin_x3_squared + 3

        # Control input terms
        u_linear = a11 * x1 + a12 * x2 + a13 * x3 + a14 * x4

        # System dynamics
        dx[1] = x2
        dx[2] = (u_linear + 2 * (x4^2 - g * cos_x3) * sin_x3) / denominator
        dx[3] = x4
        dx[4] = (2 * x4^2 * sin_x3 * cos_x3 + u_linear * cos_x3 - 2 * g * sin_x3) / denominator
    end

    # Create the nonlinear continuous system
    global sys = BlackBoxContinuousSystem(cp_lqr_dynamics!, 4)

    println("✓ CP-LQR system defined successfully")

    # Create sets and parameters from config
    global X0, target = create_juliareach_sets(config)
    global params, options = setup_juliareach_params(config, setting)

    println("✓ Sets and parameters loaded from config")
    println("  - Initial set: $(X0)")
    println("  - Target set: $(target)")
    println("  - Time horizon: $(params["T"]) seconds")

catch e
    println("❌ Failed to load benchmark: $e")
    exit(1)
end

# Run Reachability Analysis
println("\n--- Starting Reachability Analysis ($(setting["name"])) ---")
println("Using TMJets algorithm for nonlinear systems...")
start_time = time()

result = Dict()  # Initialize result dictionary

try
    # Import ReachabilityAnalysis functions
    using ReachabilityAnalysis, TaylorModels

    # Create initial value problem
    prob = InitialValueProblem(sys, X0)

    # Solve reachability problem using TMJets for nonlinear systems
    # Use conservative settings for better accuracy
    global reachable_set = solve(prob, T=params["T"], alg=TMJets(
        abstol=1e-10,    # Absolute tolerance
        orderT=5,        # Taylor order for time
        orderQ=1,        # Taylor order for spatial variables
        δ=0.05          # Time step
    ))

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

save_analysis_results(result, config, setting, X0, target, output_dir)

println("\n=== JuliaReach Analysis Complete ===")
