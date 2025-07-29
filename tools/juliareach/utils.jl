# JuliaReach Utilities for Reachability Analysis
# 
# This file contains utility functions for:
# - Loading benchmark configurations
# - Creating linear systems from config matrices
# - Creating initial and target sets (box sets only)
# - Setting up analysis parameters
# - Running reachability analysis
# - Saving results
#
# Note: Only linear systems are supported via utility functions.
# For nonlinear systems, define them manually in the Julia script.
# Only box sets are supported - other set types will cause errors.

using Pkg, JSON3, Dates

# Try to import the main packages at the top level
try
    using ReachabilityAnalysis, LazySets, Plots, TaylorModels
    global packages_loaded = true
catch e
    global packages_loaded = false
    println("Some packages not available yet - will install as needed")
end

"""
Initialize JuliaReach environment by installing and importing required packages
"""
function init_juliareach_environment(packages::Vector{String})
    println("Installing/updating JuliaReach packages...")

    for pkg in packages
        try
            # Try to import the package
            @eval using $(Symbol(pkg))
            println("✓ $pkg already available")
        catch
            println("Installing $pkg...")
            Pkg.add(pkg)
            @eval using $(Symbol(pkg))
            println("✓ $pkg installed and loaded")
        end
    end

    println("JuliaReach environment initialized successfully!")
end

"""
Load benchmark configuration from JSON file
"""
function load_benchmark_config(config_file::String)
    println("Loading configuration from: $config_file")

    # Read JSON file
    config_content = read(config_file, String)
    config = JSON3.read(config_content)

    println("✓ Configuration loaded successfully")
    return config
end

"""
Check if all sets in the configuration are compatible with JuliaReach (boxes only)
"""
function check_sets_compatibility(config)
    incompatible_sets = []

    # Check initial sets
    for (i, set_config) in enumerate(config.initial_sets)
        if set_config.type != "box"
            push!(incompatible_sets, "Initial set $i: $(set_config.type)")
        end
    end

    # Check target sets  
    for (i, set_config) in enumerate(config.verification.target_sets)
        if set_config.type != "box"
            push!(incompatible_sets, "Target set $i: $(set_config.type)")
        end
    end

    if length(incompatible_sets) > 0
        println("⚠️  Incompatible sets found (JuliaReach only supports box sets):")
        for set_info in incompatible_sets
            println("    - $set_info")
        end
        return false
    end

    return true
end

"""
Create JuliaReach system from benchmark configuration
Note: Only linear systems are supported via this function.
For nonlinear systems, define them manually in the script.
"""
function create_juliareach_system(config)
    println("Creating JuliaReach system...")

    system_config = config.system

    if system_config.type == "linear"
        return create_linear_system(system_config)
    elseif system_config.type == "nonlinear"
        error("Nonlinear systems must be defined manually in the script. Use create_linear_system() for linear systems only.")
    else
        error("Unsupported system type: $(system_config.type)")
    end
end

"""
Create linear system from dynamics configuration
Supports both autonomous (x' = Ax) and non-autonomous (x' = Ax + Bu) systems
"""
function create_linear_system(system_config)
    println("Creating linear system...")

    if !haskey(system_config, :A)
        error("Linear system must have matrix A")
    end

    # Convert A matrix from nested array to Julia matrix
    A_nested = system_config.A
    n_rows = length(A_nested)
    n_cols = length(A_nested[1])

    A = zeros(n_rows, n_cols)
    for i in 1:n_rows
        for j in 1:n_cols
            A[i, j] = A_nested[i][j]
        end
    end

    println("  A matrix: $(size(A))")

    # Check if there's a B matrix (for control input systems)
    if haskey(system_config, :B)
        B_nested = system_config.B
        B = zeros(n_rows, length(B_nested[1]))
        for i in 1:n_rows
            for j in 1:length(B_nested[1])
                B[i, j] = B_nested[i][j]
            end
        end
        println("  B matrix: $(size(B))")

        # Linear system with control input: x' = Ax + Bu
        sys = LinearControlSystem(A, B)
        println("✓ Linear control system created")
    else
        # Autonomous linear system: x' = Ax
        sys = LinearContinuousSystem(A)
        println("✓ Autonomous linear system created")
    end

    return sys
end



"""
Create JuliaReach initial and target sets from benchmark configuration
Only supports box sets - converts them to zonotopes for computation
"""
function create_juliareach_sets(config)
    println("Creating JuliaReach sets (boxes only)...")

    # Create initial sets (only boxes)
    X0 = create_initial_sets(config.initial_sets)

    # Create target sets (only boxes)
    target = create_target_sets(config.verification.target_sets)

    println("✓ Sets created successfully")
    return X0, target
end

"""
Create initial sets from configuration
"""
function create_initial_sets(initial_sets_config)
    println("Creating initial sets...")

    # Filter only box sets
    box_sets = filter(set -> set.type == "box", initial_sets_config)

    if length(box_sets) == 0
        error("No box sets found in initial sets. JuliaReach only supports box sets.")
    end

    if length(box_sets) == 1
        return create_box_set(box_sets[1])
    else
        # Union of multiple box sets
        sets = [create_box_set(set_config) for set_config in box_sets]
        return UnionSet(sets)
    end
end

"""
Create target sets from configuration
"""
function create_target_sets(target_sets_config)
    println("Creating target sets...")

    # Filter only box sets
    box_sets = filter(set -> set.type == "box", target_sets_config)

    if length(box_sets) == 0
        error("No box sets found in target sets. JuliaReach only supports box sets.")
    end

    if length(box_sets) == 1
        return create_box_set(box_sets[1])
    else
        # Union of multiple box sets
        sets = [create_box_set(set_config) for set_config in box_sets]
        return UnionSet(sets)
    end
end

"""
Create a box set and convert to zonotope for JuliaReach computation
"""
function create_box_set(set_config)
    bounds = set_config.bounds

    # Convert bounds to low and high vectors
    low = [bound[1] for bound in bounds]
    high = [bound[2] for bound in bounds]

    # Calculate center and radius for Hyperrectangle
    center = (low + high) / 2
    radius = (high - low) / 2

    # Create hyperrectangle (box) with center and radius
    box = Hyperrectangle(center, radius)

    # Convert to zonotope for better computation
    zonotope_set = convert(Zonotope, box)

    println("  ✓ Box set created: $(length(bounds))D box")
    println("    Center: $center, Radius: $radius")
    return zonotope_set
end

"""
Setup JuliaReach parameters and options from benchmark configuration
"""
function setup_juliareach_params(config, setting)
    println("Setting up JuliaReach parameters...")

    # Extract time horizon from config
    time_horizon = config.verification.time_horizon
    T = time_horizon[2] - time_horizon[1]

    # Setup parameters
    params = Dict(
        "T" => T,
        "t0" => time_horizon[1]
    )

    # Setup options based on algorithm
    options = Dict(
        "δ" => setting["δ"],
        "vars" => setting["vars"],
        "plot_vars" => (1, 2)  # Default to first two variables
    )

    # Set plotting variables if specified in config
    if haskey(config, :checking) && haskey(config.checking, :vis_dims)
        vis_dims = config.checking.vis_dims[1]  # Take first vis_dims pair
        options["plot_vars"] = (vis_dims[1] + 1, vis_dims[2] + 1)  # Convert to 1-based indexing
    end

    println("✓ Parameters and options configured")
    return params, options
end

"""
Check if reachable set intersects with target set
"""
function check_reachability(reachable_set, target_set)
    println("Checking reachability...")

    try
        # For TMJets results, we need to project the reachable sets
        intersection_found = false
        total_sets = length(reachable_set.F)

        println("  Checking intersection with $total_sets reachable sets...")

        # Check all reachable sets for intersection
        for i in 1:total_sets
            reach_set = reachable_set.F[i]

            # For Taylor model sets, we need to overapproximate
            try
                # Get a box overapproximation of the reach set
                reach_box = overapproximate(reach_set, Hyperrectangle)

                # Check intersection with target
                if !isempty(intersection(reach_box, target_set))
                    println("✓ Intersection found at step $i")
                    intersection_found = true
                    break
                end
            catch e2
                # If overapproximation fails, skip this set
                continue
            end

            # Print progress for large sets
            if i % 50 == 0 || i == total_sets
                println("  Checked $i/$total_sets sets...")
            end
        end

        if intersection_found
            println("✓ Target is REACHABLE!")
            return true
        else
            println("✓ Target is NOT reachable")
            return false
        end

    catch e
        println("⚠️  Error in reachability check: $e")
        println("⚠️  Assuming target is not reachable")
        return false
    end
end

"""
Save analysis results (summary and plot) to output directory
"""
function save_analysis_results(result, config, setting, X0, target, output_dir, config_file="")
    println("Saving analysis results...")

    # Create tool-specific subdirectory inside the results folder
    tool_name = haskey(setting, "tool") ? setting["tool"] : "juliareach"
    tool_output_dir = joinpath(output_dir, "results", tool_name, setting["name"])

    if !isdir(tool_output_dir)
        mkpath(tool_output_dir)
    end

    # Generate timestamp
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

    # Save summary as JSON
    summary_file = joinpath(tool_output_dir, "summary_$timestamp.json")
    save_summary(result, config, setting, summary_file, config_file)

    # Save plot if analysis was successful
    if result["success"] && result["reachable_set"] !== nothing
        plot_file = joinpath(tool_output_dir, "plot_$timestamp.png")
        save_plot(result["reachable_set"], X0, target, plot_file, setting)
    end

    println("✓ Results saved to: $tool_output_dir")
end

"""
Save summary to JSON file
"""
function save_summary(result, config, setting, summary_file, config_file="")
    # Extract just the filename from the full path
    config_filename = config_file != "" ? basename(config_file) : "unknown.json"

    summary = Dict(
        "timestamp" => string(now()),
        "config_file" => config_filename,
        "setting" => setting,
        "success" => result["success"],
        "target_reached" => result["target_reached"],
        "computation_time" => result["computation_time"],
        "error_message" => result["error_message"]
    )

    if haskey(result, "num_sets")
        summary["num_sets"] = result["num_sets"]
    end

    open(summary_file, "w") do f
        JSON3.pretty(f, summary)
    end

    println("✓ Summary saved to: $summary_file")
end

"""
Save plot to PNG file - uses JuliaReach's built-in plot functionality
"""
function save_plot(reachable_set, X0, target, plot_file, setting)
    try
        # Determine the dimension of the system
        dim = 2  # Default value

        try
            # Try to get dimension from initial set
            if isa(X0, Zonotope) && hasfield(typeof(X0), :center)
                dim = length(X0.center)
            elseif hasfield(typeof(X0), :center)
                dim = length(X0.center)
            else
                # Try to get dimension from the first reachable set
                if !isempty(reachable_set.F)
                    first_set = reachable_set.F[1]
                    if isa(first_set, Zonotope) && hasfield(typeof(first_set), :center)
                        dim = length(first_set.center)
                    elseif hasfield(typeof(first_set), :center)
                        dim = length(first_set.center)
                    end
                end
            end
        catch e
            println("⚠️  Error detecting dimension: $e")
            dim = 2
        end

        println("Detected system dimension: $dim")

        if dim == 2
            # Handle 2D system - single plot
            save_2d_plot(reachable_set, X0, target, plot_file, setting)
        else
            # Handle multi-dimensional system - dimension pairs plots
            save_dimension_pairs_plots(reachable_set, X0, target, plot_file, setting, dim)
        end

        println("✓ Plot saved to: $plot_file")
    catch e
        println("⚠️  Error saving plot: $e")
    end
end

"""
Save 2D plot - state space visualization
"""
function save_2d_plot(reachable_set, X0, target, plot_file, setting)
    # Create plot with specified variables
    p = plot(reachable_set, vars=(1, 2), alpha=0.3, color=:blue, label="Reachable Set")

    # Plot initial set
    plot!(p, X0, alpha=0.5, color=:green, label="Initial Set")

    # Plot target set
    plot!(p, target, alpha=0.5, color=:red, label="Target Set")

    # Set plot properties
    xlabel!(p, "x1")
    ylabel!(p, "x2")
    title!(p, "JuliaReach Analysis - $(setting["name"])")

    # Save plot
    savefig(p, plot_file)
end

"""
Save dimension pairs plots for multi-dimensional systems
Generate plots like CORA: [1,2], [3,4], etc. with initial, target, and reachable sets
"""
function save_dimension_pairs_plots(reachable_set, X0, target, plot_file, setting, dim)
    # Get base filename without extension
    base_name = replace(plot_file, ".png" => "")

    println("Creating dimension pairs plots for $dim-dimensional system...")

    # Generate dimension pairs: [1,2], [3,4], [5,6], etc.
    dimension_pairs = []
    for i in 1:2:dim
        if i + 1 <= dim
            push!(dimension_pairs, (i, i + 1))
        elseif i <= dim
            # For odd dimensions, pair the last dimension with the first
            push!(dimension_pairs, (i, 1))
        end
    end

    println("Dimension pairs to plot: $dimension_pairs")

    # Create individual plots for each dimension pair
    for (i, (dim1, dim2)) in enumerate(dimension_pairs)
        try
            println("Creating plot for dimensions [$dim1, $dim2]...")

            # Create plot for this dimension pair - plot the solution directly
            p = plot(reachable_set; vars=(dim1, dim2),
                linecolor=:blue, color=:blue, alpha=0.3,
                label="Reachable Set", lw=0.5)

            # For initial and target sets, we need to project to 2D if they are higher dimensional
            try
                if dim > 2
                    # Project initial set to 2D
                    X0_proj = project(X0, [dim1, dim2])
                    plot!(p, X0_proj, alpha=0.8, color=:green, linewidth=2,
                        label="Initial Set", fillalpha=0.6)
                else
                    plot!(p, X0, alpha=0.8, color=:green, linewidth=2,
                        label="Initial Set", fillalpha=0.6)
                end
            catch e
                println("⚠️  Error plotting initial set: $e")
            end

            try
                if dim > 2
                    # Project target set to 2D
                    target_proj = project(target, [dim1, dim2])
                    plot!(p, target_proj, alpha=0.8, color=:red, linewidth=2,
                        label="Target Set", fillalpha=0.6)
                else
                    plot!(p, target, alpha=0.8, color=:red, linewidth=2,
                        label="Target Set", fillalpha=0.6)
                end
            catch e
                println("⚠️  Error plotting target set: $e")
            end

            # Set plot properties
            xlabel!(p, "x$dim1")
            ylabel!(p, "x$dim2")
            title!(p, "JuliaReach Analysis - $(setting["name"]) - x$dim1 vs x$dim2")

            # Save individual plot
            individual_file = "$(base_name)_x$(dim1)_x$(dim2).png"
            savefig(p, individual_file)
            println("✓ Saved dimension pair plot: $individual_file")

        catch e
            println("⚠️  Error creating plot for dimensions [$dim1, $dim2]: $e")
        end
    end

    # Create combined plot with all dimension pairs
    if length(dimension_pairs) > 1
        try
            println("Creating combined dimension pairs plot...")

            # Determine layout
            num_pairs = length(dimension_pairs)
            if num_pairs == 2
                layout = (1, 2)
            elseif num_pairs <= 4
                layout = (2, 2)
            elseif num_pairs <= 6
                layout = (2, 3)
            else
                layout = (3, 3)  # Max 9 plots
            end

            combined_plot = plot(layout=layout, size=(600 * layout[2], 400 * layout[1]))

            for (i, (dim1, dim2)) in enumerate(dimension_pairs)
                if i > 9  # Max 9 subplots
                    break
                end

                try
                    # Create subplot for this dimension pair
                    plot!(combined_plot, reachable_set; vars=(dim1, dim2),
                        linecolor=:blue, color=:blue, alpha=0.3, lw=0.5,
                        label=(i == 1 ? "Reachable Set" : ""), subplot=i)

                    # Plot initial set
                    try
                        if dim > 2
                            X0_proj = project(X0, [dim1, dim2])
                            plot!(combined_plot, X0_proj, alpha=0.8, color=:green, linewidth=2,
                                label=(i == 1 ? "Initial Set" : ""), subplot=i, fillalpha=0.6)
                        else
                            plot!(combined_plot, X0, alpha=0.8, color=:green, linewidth=2,
                                label=(i == 1 ? "Initial Set" : ""), subplot=i, fillalpha=0.6)
                        end
                    catch e
                        # Skip if projection fails
                    end

                    # Plot target set
                    try
                        if dim > 2
                            target_proj = project(target, [dim1, dim2])
                            plot!(combined_plot, target_proj, alpha=0.8, color=:red, linewidth=2,
                                label=(i == 1 ? "Target Set" : ""), subplot=i, fillalpha=0.6)
                        else
                            plot!(combined_plot, target, alpha=0.8, color=:red, linewidth=2,
                                label=(i == 1 ? "Target Set" : ""), subplot=i, fillalpha=0.6)
                        end
                    catch e
                        # Skip if projection fails
                    end

                    # Set subplot properties
                    xlabel!(combined_plot, "x$dim1", subplot=i)
                    ylabel!(combined_plot, "x$dim2", subplot=i)
                    title!(combined_plot, "x$dim1 vs x$dim2", subplot=i)

                catch e2
                    println("⚠️  Error creating subplot for dimensions [$dim1, $dim2]: $e2")
                end
            end

            # Set main title
            plot!(combined_plot, plot_title="JuliaReach Analysis - $(setting["name"]) - Dimension Pairs")

            # Save combined plot
            combined_file = "$(base_name)_dimension_pairs.png"
            savefig(combined_plot, combined_file)
            println("✓ Saved combined dimension pairs plot: $combined_file")

        catch e
            println("⚠️  Error creating combined dimension pairs plot: $e")
        end
    end
end
