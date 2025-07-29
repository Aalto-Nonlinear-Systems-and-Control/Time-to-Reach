# JuliaReach Reachability Analysis - MAS-CON (16D Multi-Agent System)
# This script runs JuliaReach reachability analysis on the 16-dimensional multi-agent system.
# System: Complex nonlinear 16D dynamics with synchronization behavior
# Only supports box sets (no level sets).
# The nonlinear system is defined manually in this script.

using Pkg, Dates

# ================== SCRIPT CONFIGURATION ==================
# 1. Define the JuliaReach packages and settings
juliareach_packages = ["ReachabilityAnalysis", "LazySets", "Plots", "JSON3", "TaylorModels"]

# 2. Define the specific reachability setting for this script
setting = Dict(
    "name" => "juliareach_mas_con",     # Tool-specific name for MAS-CON
    "tool" => "juliareach",             # Tool identifier
    "alg" => "TMJets",                  # Nonlinear algorithm for Taylor models
    "δ" => 0.05,                        # Time step (larger for 16D system)
    "T" => 10.0,                        # Time horizon (will be overridden by config)
    "vars" => (1, 2),                   # Variables to plot (1-based indexing)
    "abstol" => 1e-10,                  # Absolute tolerance for TMJets
    "orderT" => 5,                      # Taylor order in time
    "orderQ" => 1                       # Taylor order in space
)

# 3. Define the configuration file for the benchmark problem
config_file = "../../configs/benchmark_MAS_CON_FIN_UNR_SAFE_0.json"
# ==========================================================

println("=== JuliaReach Reachability Analysis ===")
println("System: MAS-CON (16D Multi-Agent System)")
println("Config: $config_file")
println("Setting: $(setting["name"])")
println("⚠️  Note: Only box sets are supported (level sets will be skipped)")
println("📍 16D nonlinear system defined manually in script")
println("⚠️  Warning: This is a high-dimensional system - computation may take time")

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

    # Define the 16D Multi-Agent System manually
    println("\n--- Defining 16D Multi-Agent System ---")
    println("System: 16 agents with complex nonlinear coupling")
    println("Equations: Each agent has dynamics with sin/tanh coupling terms")
    println("Target: Synchronization to [-0.1, 0.1] for all agents")

    # Import required packages for system definition
    using ReachabilityAnalysis

    # Define the 16D Multi-Agent System dynamics function
    function mas_con_dynamics!(dx, x, p, t)
        # Extract individual agent states
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = x

        # Agent 1 dynamics
        dx[1] = -0.441936617444625 * x1 - 0.1 * (0.779918792240115 * sin(x1 - x2) + 0.438409231440893 * sin(x1 - x3) + 0.723465177830941 * sin(x1 - x4) + 0.538495870410434 * sin(x1 - x6) + 0.935902550688138 * tanh(x1 - x2) + 0.526091077729072 * tanh(x1 - x3) + 0.868158213397129 * tanh(x1 - x4) + 0.64619504449252 * tanh(x1 - x6)) / (1 + exp(-x1))

        # Agent 2 dynamics
        dx[2] = -0.441936617444625 * x2 - 0.1 * (-0.268438980101871 * sin(x1 - x2) + 0.803739036104375 * sin(x2 - x4) + 0.0659363469059051 * sin(x2 - x6) + 0.288145599307994 * sin(x2 - x7) + 0.909593527719614 * sin(x2 - x8) - 0.322126776122245 * tanh(x1 - x2) + 0.96448684332525 * tanh(x2 - x4) + 0.0791236162870861 * tanh(x2 - x6) + 0.345774719169592 * tanh(x2 - x7) + 1.09151223326354 * tanh(x2 - x8)) / (1 + exp(-x2))

        # Agent 3 dynamics
        dx[3] = -0.441936617444625 * x3 - 0.1 * (-0.213385353579916 * sin(x1 - x3) + 0.024899227550348 * sin(x3 - x4) + 0.230302879020965 * sin(x3 - x7) - 0.256062424295899 * tanh(x1 - x3) + 0.0298790730604176 * tanh(x3 - x4) + 0.276363454825158 * tanh(x3 - x7)) / (1 + exp(-x3))

        # Agent 4 dynamics
        dx[4] = -0.441936617444625 * x4 - 0.1 * (-0.909128374886731 * sin(x1 - x4) - 0.13316944575925 * sin(x2 - x4) - 0.523412580673766 * sin(x3 - x4) + 0.669013240883914 * sin(x4 - x5) + 0.49076588909107 * sin(x4 - x8) - 1.09095404986408 * tanh(x1 - x4) - 0.1598033349111 * tanh(x2 - x4) - 0.628095096808519 * tanh(x3 - x4) + 0.802815889060697 * tanh(x4 - x5) + 0.588919066909285 * tanh(x4 - x8)) / (1 + exp(-x4))

        # Agent 5 dynamics
        dx[5] = -0.441936617444625 * x5 - 0.1 * (-0.837917994309261 * sin(x4 - x5) + 0.313994677212662 * sin(x5 - x6) + 0.572625332643954 * sin(x5 - x7) - 1.00550159317111 * tanh(x4 - x5) + 0.376793612655195 * tanh(x5 - x6) + 0.687150399172745 * tanh(x5 - x7)) / (1 + exp(-x5))

        # Agent 6 dynamics
        dx[6] = -0.441936617444625 * x6 - 0.1 * (-0.4528429325464 * sin(x1 - x6) - 0.352978365944395 * sin(x2 - x6) - 0.459092977891431 * sin(x5 - x6) + 0.412991829113835 * sin(x6 - x7) - 0.54341151905568 * tanh(x1 - x6) - 0.423574039133274 * tanh(x2 - x6) - 0.550911573469718 * tanh(x5 - x6) + 0.495590194936602 * tanh(x6 - x7)) / (1 + exp(-x6))

        # Agent 7 dynamics
        dx[7] = -0.441936617444625 * x7 - 0.1 * (-0.741118872913264 * sin(x2 - x7) - 0.42237404364314 * sin(x3 - x7) - 0.634379868633839 * sin(x5 - x7) - 0.522906201028345 * sin(x6 - x7) + 0.00142688056275819 * sin(x7 - x8) - 0.889342647495917 * tanh(x2 - x7) - 0.506848852371768 * tanh(x3 - x7) - 0.761255842360607 * tanh(x5 - x7) - 0.627487441234014 * tanh(x6 - x7) + 0.00171225667530983 * tanh(x7 - x8)) / (1 + exp(-x7))

        # Agent 8 dynamics
        dx[8] = -0.441936617444625 * x8 - 0.1 * (-0.709394393725213 * sin(x2 - x8) - 0.696160463516969 * sin(x4 - x8) - 0.0531286906729576 * sin(x7 - x8) - 0.851273272470255 * tanh(x2 - x8) - 0.835392556220363 * tanh(x4 - x8) - 0.0637544288075491 * tanh(x7 - x8)) / (1 + exp(-x8))

        # Agent 9 dynamics
        dx[9] = -0.441936617444625 * x9 - 0.1 * (-0.964970999536127 * sin(x11 - x9) - 0.945048223792794 * sin(x12 - x9) - 0.472323996288402 * sin(x14 - x9) + 0.235120407257464 * sin(x9 - x10) - 1.15796519944335 * tanh(x11 - x9) - 1.13405786855135 * tanh(x12 - x9) - 0.566788795546083 * tanh(x14 - x9) + 0.282144488708957 * tanh(x9 - x10)) / (1 + exp(-x9))

        # Agent 10 dynamics
        dx[10] = -0.441936617444625 * x10 - 0.1 * (-0.485825228708897 * sin(x12 - x10) - 0.343536529704358 * sin(x14 - x10) - 0.324426169672443 * sin(x15 - x10) - 0.30041890431804 * sin(x16 - x10) - 0.308733657297835 * sin(x9 - x10) - 0.582990274450676 * tanh(x12 - x10) - 0.41224383564523 * tanh(x14 - x10) - 0.389311403606932 * tanh(x15 - x10) - 0.360502685181648 * tanh(x16 - x10) - 0.370480388757402 * tanh(x9 - x10)) / (1 + exp(-x10))

        # Agent 11 dynamics
        dx[11] = -0.441936617444625 * x11 - 0.1 * (0.774900375814017 * sin(x11 - x12) + 0.460630296163277 * sin(x11 - x15) + 0.165501400465787 * sin(x11 - x9) + 0.929880450976821 * tanh(x11 - x12) + 0.552756355395932 * tanh(x11 - x15) + 0.198601680558945 * tanh(x11 - x9)) / (1 + exp(-x11))

        # Agent 12 dynamics
        dx[12] = -0.441936617444625 * x12 - 0.1 * (-0.800479048998464 * sin(x11 - x12) + 0.0406558094366858 * sin(x12 - x13) + 0.475764499424244 * sin(x12 - x16) + 0.887288951852735 * sin(x12 - x9) + 0.674918769866788 * sin(x12 - x10) - 0.960574858798156 * tanh(x11 - x12) + 0.0487869713240229 * tanh(x12 - x13) + 0.570917399309093 * tanh(x12 - x16) + 1.06474674222328 * tanh(x12 - x9) + 0.809902523840145 * tanh(x12 - x10)) / (1 + exp(-x12))

        # Agent 13 dynamics
        dx[13] = -0.441936617444625 * x13 - 0.1 * (-0.658748258954145 * sin(x12 - x13) + 0.357070628352879 * sin(x13 - x14) + 0.812829550163706 * sin(x13 - x15) - 0.790497910744975 * tanh(x12 - x13) + 0.428484754023455 * tanh(x13 - x14) + 0.975395460196448 * tanh(x13 - x15)) / (1 + exp(-x13))

        # Agent 14 dynamics
        dx[14] = -0.441936617444625 * x14 - 0.1 * (-0.0071432752789985 * sin(x13 - x14) + 0.463136218038147 * sin(x14 - x15) + 0.599854437543446 * sin(x14 - x9) + 0.728161283147134 * sin(x14 - x10) - 0.00857193033479819 * tanh(x13 - x14) + 0.555763461645777 * tanh(x14 - x15) + 0.719825325052136 * tanh(x14 - x9) + 0.87379353977656 * tanh(x14 - x10)) / (1 + exp(-x14))

        # Agent 15 dynamics
        dx[15] = -0.441936617444625 * x15 - 0.1 * (-0.828453194996125 * sin(x11 - x15) - 0.128147854002733 * sin(x13 - x15) - 0.230430674410219 * sin(x14 - x15) + 0.132473988816749 * sin(x15 - x16) + 0.607770751443867 * sin(x15 - x10) - 0.99414383399535 * tanh(x11 - x15) - 0.153777424803279 * tanh(x13 - x15) - 0.276516809292263 * tanh(x14 - x15) + 0.158968786580098 * tanh(x15 - x16) + 0.729324901732641 * tanh(x15 - x10)) / (1 + exp(-x15))

        # Agent 16 dynamics
        dx[16] = -0.441936617444625 * x16 - 0.1 * (-0.782230151323778 * sin(x12 - x16) - 0.532644804943674 * sin(x15 - x16) + 0.574862587589149 * sin(x16 - x10) - 0.938676181588534 * tanh(x12 - x16) - 0.639173765932408 * tanh(x15 - x16) + 0.689835105106979 * tanh(x16 - x10)) / (1 + exp(-x16))
    end

    # Create the nonlinear continuous system
    global sys = BlackBoxContinuousSystem(mas_con_dynamics!, 16)

    println("✓ 16D Multi-Agent System defined successfully")

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
println("⏰ This may take several minutes for a 16D system...")
start_time = time()

result = Dict()  # Initialize result dictionary

try
    # Import ReachabilityAnalysis functions
    using ReachabilityAnalysis, TaylorModels

    # Create initial value problem
    prob = InitialValueProblem(sys, X0)

    # Solve reachability problem using TMJets for nonlinear systems
    # Use conservative settings for high-dimensional system
    global reachable_set = solve(prob, T=params["T"],
        alg=TMJets(abstol=setting["abstol"],
            orderT=setting["orderT"],
            orderQ=setting["orderQ"]))

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
