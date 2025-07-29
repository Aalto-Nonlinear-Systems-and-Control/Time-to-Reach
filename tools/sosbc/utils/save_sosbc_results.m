function save_sosbc_results(result, output_dir, create_plots)
    % SAVE_SOSBC_RESULTS - Save SOSBC results and create visualizations
    %
    % Inputs:
    %   result - result structure from sosbc_pipeline
    %   output_dir - directory to save results
    %   create_plots - boolean, whether to create plots

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Generate timestamp
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');

    %% Save numerical results
    results_file = fullfile(output_dir, sprintf('sosbc_results_%s.mat', timestamp));

    % Prepare data structure
    analysis_results.success = result.success;
    analysis_results.computation_time = result.computation_time;
    analysis_results.timestamp = timestamp;
    analysis_results.config = result.config;

    if result.success
        analysis_results.barrier_coeffs = result.barrier_coeffs;
        analysis_results.barrier_degmat = result.barrier_degmat;
        analysis_results.solver_info = result.solver_info;
    end

    save(results_file, 'analysis_results');
    fprintf('✓ Results saved: %s\n', results_file);

    %% Create summary JSON file
    summary_file = fullfile(output_dir, sprintf('summary_%s.json', timestamp));
    create_summary_file(result, summary_file, timestamp);
    fprintf('✓ Summary saved: %s\n', summary_file);

    %% Create visualization if successful and requested
    if result.success && create_plots
        plot_file = fullfile(output_dir, sprintf('sosbc_plot_%s.png', timestamp));
        create_sosbc_plot(result, plot_file);
        fprintf('✓ Plot saved: %s\n', plot_file);
    end

end

function create_sosbc_plot(result, plot_file)
    % CREATE_SOSBC_PLOT - Create visualization of barrier certificate

    system = result.system;
    config = result.config;
    barrier = result.barrier;

    % Get domain bounds for plotting
    if isfield(config, 'checking') && isfield(config.checking, 'domain')
        domain = config.checking.domain;

        if iscell(domain)
            % Handle cell array format
            xlim_range = domain{1};
            ylim_range = domain{2};
        else
            % Handle numeric array format (from JSON)
            xlim_range = domain(1, :); % First row: [x1_min, x1_max]
            ylim_range = domain(2, :); % Second row: [x2_min, x2_max]
        end

    else
        % Default domain
        xlim_range = [-3, 3];
        ylim_range = [-3, 3];
    end

    % Create plotting grid
    x1_range = linspace(xlim_range(1), xlim_range(2), 150);
    x2_range = linspace(ylim_range(1), ylim_range(2), 120);
    [X1, X2] = meshgrid(x1_range, x2_range);

    % Evaluate barrier certificate on grid
    B_vals = evaluate_barrier_on_grid(barrier, system.vars, X1, X2);

    % Create plot
    figure('Position', [100, 100, 800, 600]);
    hold on;

    % Track what elements are actually plotted for legend
    legend_entries = {};
    legend_handles = [];

    % Plot barrier function
    if ~all(isnan(B_vals(:)))
        min_B = min(B_vals(:));
        max_B = max(B_vals(:));

        % Plot barrier boundary (B = 0) if it exists
        if min_B <= 0 && max_B >= 0
            % First, fill the interior of the 0-level set (B > 0 region)
            % Create a binary mask for the safe region
            safe_region = B_vals > 0;

            if any(safe_region(:))
                % Plot only the interior of the 0-level set with light blue fill
                contourf(X1, X2, B_vals, [0, max_B], 'FaceColor', [0.7, 0.9, 1], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
                legend_handles(end + 1) = fill(NaN, NaN, [0.7, 0.9, 1], 'FaceAlpha', 0.4);
                legend_entries{end + 1} = 'Safe Region (B > 0)';
            end

            % Then plot the barrier boundary (B = 0) prominently
            [C, h2] = contour(X1, X2, B_vals, [0, 0], 'k-', 'LineWidth', 3);

            if ~isempty(C) && ~isempty(h2)
                legend_handles(end + 1) = h2;
                legend_entries{end + 1} = 'Barrier Boundary (B = 0)';
            end

        else
            % If no zero level set exists, just indicate this
            fprintf('Warning: No zero-level set found for barrier certificate\n');
        end

    end

    % Plot initial sets (green with filled interior)
    if ~isempty(config.initial_sets)
        plot_sets(config.initial_sets, 'green', 'Initial Set');
        % Add a dummy handle for initial set legend
        h3 = fill(NaN, NaN, 'green', 'FaceAlpha', 0.5);
        legend_handles(end + 1) = h3;
        legend_entries{end + 1} = 'Initial Set';
    end

    % Plot target sets (red with filled interior)
    if ~isempty(config.verification.target_sets)
        plot_sets(config.verification.target_sets, 'red', 'Target Set');
        % Add a dummy handle for target set legend
        h4 = fill(NaN, NaN, 'red', 'FaceAlpha', 0.5);
        legend_handles(end + 1) = h4;
        legend_entries{end + 1} = 'Target Set';
    end

    % Formatting
    xlim(xlim_range);
    ylim(ylim_range);
    xlabel('x_1', 'FontSize', 14);
    ylabel('x_2', 'FontSize', 14);
    title(sprintf('SOSBC: %s', config.benchmark.description), 'FontSize', 16);
    grid on;

    % Add legend only if we have elements to show
    if ~isempty(legend_handles)
        legend(legend_handles, legend_entries, 'Location', 'best');
    end

    % Save plot
    saveas(gcf, plot_file);
    close(gcf);
end

function B_vals = evaluate_barrier_on_grid(barrier, vars, X1, X2)
    % EVALUATE_BARRIER_ON_GRID - Evaluate barrier certificate on grid
    % Uses SOSTOOLS polynomial evaluation approach

    try
        % Convert polynomial to string, then to symbolic expression
        barrier_str = poly2str(barrier);
        syms x1 x2;
        barrier_sym = str2sym(barrier_str);

        % Create function handle from symbolic expression
        B_func = matlabFunction(barrier_sym, 'Vars', [x1, x2]);
        B_vals = B_func(X1, X2);

    catch ME
        fprintf('Warning: Could not evaluate barrier certificate: %s\n', ME.message);
        B_vals = NaN(size(X1));
    end

end

function s = poly2str(p)
    % Convert polynomial to string (helper function)
    c = char(p);
    s = c{1};
end

function plot_sets(set_definitions, color, label)
    % PLOT_SETS - Plot sets on current figure

    for k = 1:length(set_definitions)
        set_def = set_definitions(k);

        switch set_def.type
            case 'box'
                % Plot box as rectangle
                bounds = set_def.bounds;

                if iscell(bounds)
                    % Handle cell array format
                    x_min = bounds{1}(1);
                    x_max = bounds{1}(2);
                    y_min = bounds{2}(1);
                    y_max = bounds{2}(2);
                else
                    % Handle numeric array format (from JSON)
                    x_min = bounds(1, 1);
                    x_max = bounds(1, 2);
                    y_min = bounds(2, 1);
                    y_max = bounds(2, 2);
                end

                width = x_max - x_min;
                height = y_max - y_min;

                rectangle('Position', [x_min, y_min, width, height], ...
                    'EdgeColor', color, 'LineWidth', 2, ...
                    'FaceColor', color, 'FaceAlpha', 0.5);

            case 'level_set'
                % Plot level set defined by f(x) <= 0
                plot_level_set(set_def, color);
        end

    end

end

function plot_level_set(set_def, color)
    % PLOT_LEVEL_SET - Plot a level set defined by f(x) <= 0

    % Determine plotting domain
    if isfield(set_def, 'domain') && ~isempty(set_def.domain)
        % Use domain from config file
        domain = set_def.domain;

        % Handle different domain formats
        if iscell(domain)
            % Domain is a cell array of bounds
            x1_bounds = domain{1};
            x2_bounds = domain{2};
        else
            % Domain is a numeric array
            x1_bounds = domain(1, :);
            x2_bounds = domain(2, :);
        end

        x1_range = linspace(x1_bounds(1), x1_bounds(2), 200);
        x2_range = linspace(x2_bounds(1), x2_bounds(2), 200);
    else
        % Fall back to current axis limits
        ax = gca;
        xlim_range = xlim(ax);
        ylim_range = ylim(ax);
        x1_range = linspace(xlim_range(1), xlim_range(2), 200);
        x2_range = linspace(ylim_range(1), ylim_range(2), 200);
    end

    % Create grid for level set plotting
    [X1, X2] = meshgrid(x1_range, x2_range);

    % Get the function string
    func_str = set_def.function;

    try
        % Create symbolic variables
        syms x1 x2 real;

        % Evaluate the function symbolically
        f_sym = eval(func_str);

        % Convert to function handle for numerical evaluation
        f_func = matlabFunction(f_sym, 'Vars', [x1, x2]);

        % Evaluate function on grid
        F_vals = f_func(X1, X2);

        % Plot the level set boundary (f = 0)
        contour(X1, X2, F_vals, [0, 0], 'Color', color, 'LineWidth', 2);

        % Fill the interior region (f <= 0)
        F_vals_fill = F_vals;
        F_vals_fill(F_vals > 0) = NaN; % Mask out exterior
        F_vals_fill(F_vals <= 0) = 1; % Set interior to constant

        % Create filled contour for the interior
        % Suppress contour warnings for constant functions
        warning('off', 'MATLAB:contour:ConstantData');
        contourf(X1, X2, F_vals_fill, [0.5, 1.5], ...
            'FaceColor', color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        warning('on', 'MATLAB:contour:ConstantData');

    catch ME
        fprintf('Warning: Could not plot level set: %s\n', ME.message);
        fprintf('Function: %s\n', func_str);
    end

end

function create_summary_file(result, summary_file, timestamp)
    % CREATE_SUMMARY_FILE - Create summary JSON file for SOSBC results
    %
    % Inputs:
    %   result - SOSBC pipeline result structure
    %   summary_file - path to save summary JSON file
    %   timestamp - timestamp string

    % Extract benchmark information
    config = result.config;

    % Create summary structure following the format of other tools
    summary = struct();
    summary.benchmark = config.benchmark.name;
    summary.timestamp = timestamp;
    summary.method = 'sosbc';
    summary.barrier_found = result.success;
    summary.computation_time = result.computation_time;

    % Add system information
    summary.system_type = result.system.type;
    summary.system_dimension = length(result.system.vars);

    % Add domain bounds from config
    if isfield(config, 'checking') && isfield(config.checking, 'domain')
        summary.domain_bounds = config.checking.domain;
    end

    % Add verification result interpretation
    % For barrier certificates: if barrier found, the initial set is safe (cannot reach target)
    summary.verification_result = result.success;

    if result.success
        summary.interpretation = 'Safe: Initial set cannot reach target set (barrier certificate found)';
    else
        summary.interpretation = 'Unknown: No barrier certificate found (does not prove reachability)';
    end

    % Add solver information if available
    if result.success && isfield(result, 'solver_info')
        summary.solver_status = 'optimal';

        if isfield(result.solver_info, 'info')
            summary.solver_details.pinf = result.solver_info.info.pinf;
            summary.solver_details.dinf = result.solver_info.info.dinf;
        end

    else
        summary.solver_status = 'failed';
    end

    % Add barrier certificate information if successful
    if result.success
        summary.barrier_degree = size(result.barrier_degmat, 1);
        summary.barrier_terms = size(result.barrier_degmat, 2);
    end

    % Convert to JSON and save
    json_text = jsonencode(summary, 'PrettyPrint', true);

    % Write to file
    fid = fopen(summary_file, 'w');

    if fid == -1
        warning('Could not create summary file: %s', summary_file);
        return;
    end

    fprintf(fid, '%s', json_text);
    fclose(fid);
end
