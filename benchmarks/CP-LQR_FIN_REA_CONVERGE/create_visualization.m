%% CP-LQR Visualization Script
% Creates visualization for CP-LQR reachability analysis results

clear; clc; close all;

%% Configuration
cora_version = 'CORA-v2025.2.0';
config_file = '../../configs/benchmark_CP_LQR_REA_CONVERGE.json';
setting.name = 'default_nonlinear';
setting.alg = 'lin';
setting.taylorTerms = 4;
setting.zonotopeOrder = 50;
setting.timeStep = 0.05;

%% Initialize Environment
fprintf('--- Initializing CORA Environment ---\n');
cora_root_dir = fullfile(fileparts(mfilename('fullpath')), '../../tools/cora');
addpath(fullfile(cora_root_dir, 'utils'));
init_cora_environment(cora_version);

%% Load Configuration and Create Sets
fprintf('--- Loading Configuration ---\n');
config = load_benchmark_config(config_file);
sys = create_cora_system(config);
[X0, target] = create_cora_sets(config);
[params, options] = setup_cora_params(config, setting);
params.R0 = X0;

%% Run Quick Analysis for Visualization
fprintf('--- Running Analysis for Visualization ---\n');

try
    reachable_set = reach(sys, params, options);
    success = true;

    % Debug: Check the structure of reachable_set
    fprintf('Reachable set structure:\n');

    if isstruct(reachable_set)
        fprintf('  Fields: %s\n', strjoin(fieldnames(reachable_set), ', '));

        if isfield(reachable_set, 'timePoint') && isfield(reachable_set, 'timeInterval')
            fprintf('  timePoint length: %d\n', length(reachable_set.timePoint));
            fprintf('  timeInterval length: %d\n', length(reachable_set.timeInterval));
        end

    else
        fprintf('  Type: %s, Length: %d\n', class(reachable_set), length(reachable_set));
    end

catch ME
    fprintf('Error: %s\n', ME.message);
    success = false;
    reachable_set = [];
end

%% Create Visualizations
if success && ~isempty(reachable_set)
    fprintf('--- Creating Visualizations ---\n');

    % Define dimension pairs for 4D system
    dim_pairs = [1, 2; 1, 3; 1, 4; 2, 3; 2, 4; 3, 4];
    pair_names = {'x1-x2', 'x1-x3', 'x1-x4', 'x2-x3', 'x2-x4', 'x3-x4'};

    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'results', 'cora', setting.name);

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Create individual plots for each dimension pair (ALL REACHABLE SETS)
    for i = 1:size(dim_pairs, 1)
        dims = dim_pairs(i, :);

        try
            h = figure('Visible', 'off');

            % Plot reachable set (avoid FaceAlpha - use older syntax)
            plot(reachable_set, dims, 'FaceColor', [0.8, 0.8, 1], 'EdgeColor', 'blue', 'LineWidth', 1);
            hold on;

            % Plot initial set
            plot(X0, dims, 'FaceColor', 'green', 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);

            % Plot target set
            plot(target, dims, 'FaceColor', 'red', 'EdgeColor', [0.8, 0, 0], 'LineWidth', 2);

            % Set labels and title
            xlabel(sprintf('x_%d', dims(1)));
            ylabel(sprintf('x_%d', dims(2)));
            title(sprintf('CP-LQR System - %s Projection (All Sets)', pair_names{i}));
            legend('Reachable Set', 'Initial Set', 'Target Set', 'Location', 'best');
            grid on;

            % Save plot
            fig_file = fullfile(output_dir, sprintf('plot_%s_all_%s.png', pair_names{i}, timestamp));
            saveas(h, fig_file);
            fprintf('Saved: %s\n', fig_file);

            close(h);

        catch ME
            fprintf('Warning: Could not create plot for %s: %s\n', pair_names{i}, ME.message);
        end

    end

    % Create plots for LAST 5 REACHABLE SETS (enhanced approach)
    fprintf('--- Creating Last 5 Sets Visualizations ---\n');

    % Handle different types of reachable set structures
    if isstruct(reachable_set) && isfield(reachable_set, 'timePoint') && isfield(reachable_set, 'timeInterval')
        % CORA structure with timePoint and timeInterval
        time_points = reachable_set.timePoint;
        time_intervals = reachable_set.timeInterval;

        num_time_points = length(time_points);
        num_time_intervals = length(time_intervals);

        fprintf('Found %d time points and %d time intervals\n', num_time_points, num_time_intervals);

        % Use time intervals for visualization (they show the evolution over time)
        if num_time_intervals > 1
            last_n = min(5, num_time_intervals);
            start_idx = max(1, num_time_intervals - last_n + 1);

            fprintf('Plotting last %d time intervals (steps %d to %d)\n', last_n, start_idx, num_time_intervals);

            for i = 1:size(dim_pairs, 1)
                dims = dim_pairs(i, :);

                try
                    h = figure('Visible', 'off');

                    % Plot only the last n time intervals
                    for step = start_idx:num_time_intervals

                        if step <= length(time_intervals)
                            % Use different colors for different time steps
                            alpha_val = 0.3 + 0.7 * (step - start_idx + 1) / last_n;
                            color_intensity = 0.4 + 0.6 * (step - start_idx + 1) / last_n;

                            plot(time_intervals{step}, dims, 'FaceColor', [0.8 * color_intensity, 0.8 * color_intensity, 1], ...
                                'EdgeColor', [0, 0, color_intensity], 'LineWidth', 1);
                            hold on;
                        end

                    end

                    % Plot initial set
                    plot(X0, dims, 'FaceColor', 'green', 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);

                    % Plot target set (make it more prominent)
                    plot(target, dims, 'FaceColor', 'red', 'EdgeColor', [0.8, 0, 0], 'LineWidth', 3);

                    % Set labels and title
                    xlabel(sprintf('x_%d', dims(1)));
                    ylabel(sprintf('x_%d', dims(2)));
                    title(sprintf('CP-LQR System - %s Projection (Last %d Sets)', pair_names{i}, last_n));
                    legend('Last Sets', 'Initial Set', 'Target Set', 'Location', 'best');
                    grid on;

                    % Save plot
                    fig_file = fullfile(output_dir, sprintf('plot_%s_last%d_%s.png', pair_names{i}, last_n, timestamp));
                    saveas(h, fig_file);
                    fprintf('Saved: %s\n', fig_file);

                    close(h);

                catch ME
                    fprintf('Warning: Could not create last %d plot for %s: %s\n', last_n, pair_names{i}, ME.message);
                end

            end

            % Create combined plot for LAST 5 SETS
            try
                h_combined_last = figure('Position', [100, 100, 1200, 800], 'Visible', 'off');

                for i = 1:6
                    dims = dim_pairs(i, :);

                    subplot(2, 3, i);

                    % Plot only the last n time intervals
                    for step = start_idx:num_time_intervals

                        if step <= length(time_intervals)
                            alpha_val = 0.3 + 0.7 * (step - start_idx + 1) / last_n;
                            color_intensity = 0.4 + 0.6 * (step - start_idx + 1) / last_n;

                            plot(time_intervals{step}, dims, 'FaceColor', [0.8 * color_intensity, 0.8 * color_intensity, 1], ...
                                'EdgeColor', [0, 0, color_intensity], 'LineWidth', 1);
                            hold on;
                        end

                    end

                    % Plot initial and target sets
                    plot(X0, dims, 'FaceColor', 'green', 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
                    plot(target, dims, 'FaceColor', 'red', 'EdgeColor', [0.8, 0, 0], 'LineWidth', 3);

                    xlabel(sprintf('x_%d', dims(1)));
                    ylabel(sprintf('x_%d', dims(2)));
                    title(sprintf('%s (Last %d)', pair_names{i}, last_n));
                    grid on;

                    if i == 1
                        legend('Last Sets', 'Initial Set', 'Target Set', 'Location', 'best', 'FontSize', 8);
                    end

                end

                sgtitle(sprintf('CP-LQR System - All Projections (Last %d Sets)', last_n));

                combined_last_file = fullfile(output_dir, sprintf('plot_all_projections_last%d_%s.png', last_n, timestamp));
                saveas(h_combined_last, combined_last_file);
                fprintf('Saved combined last %d plot: %s\n', last_n, combined_last_file);

                close(h_combined_last);

            catch ME
                fprintf('Warning: Could not create combined last %d plot: %s\n', last_n, ME.message);
            end

        else
            fprintf('Only %d time interval(s) available, skipping last 5 sets plots.\n', num_time_intervals);
        end

    else
        % Try to handle as array or other structure
        num_steps = length(reachable_set);

        fprintf('Treating as array/cell with %d elements\n', num_steps);

        if num_steps > 1
            last_n = min(5, num_steps);
            start_idx = max(1, num_steps - last_n + 1);

            fprintf('Plotting last %d sets (steps %d to %d)\n', last_n, start_idx, num_steps);

            for i = 1:size(dim_pairs, 1)
                dims = dim_pairs(i, :);

                try
                    h = figure('Visible', 'off');

                    % Plot only the last n reachable sets
                    for step = start_idx:num_steps

                        if step <= length(reachable_set)
                            % Use different colors for different time steps
                            alpha_val = 0.3 + 0.7 * (step - start_idx + 1) / last_n;
                            color_intensity = 0.4 + 0.6 * (step - start_idx + 1) / last_n;

                            if iscell(reachable_set)
                                plot(reachable_set{step}, dims, 'FaceColor', [0.8 * color_intensity, 0.8 * color_intensity, 1], ...
                                    'EdgeColor', [0, 0, color_intensity], 'LineWidth', 1);
                            else
                                plot(reachable_set(step), dims, 'FaceColor', [0.8 * color_intensity, 0.8 * color_intensity, 1], ...
                                    'EdgeColor', [0, 0, color_intensity], 'LineWidth', 1);
                            end

                            hold on;
                        end

                    end

                    % Plot initial set
                    plot(X0, dims, 'FaceColor', 'green', 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);

                    % Plot target set (make it more prominent)
                    plot(target, dims, 'FaceColor', 'red', 'EdgeColor', [0.8, 0, 0], 'LineWidth', 3);

                    % Set labels and title
                    xlabel(sprintf('x_%d', dims(1)));
                    ylabel(sprintf('x_%d', dims(2)));
                    title(sprintf('CP-LQR System - %s Projection (Last %d Sets)', pair_names{i}, last_n));
                    legend('Last Sets', 'Initial Set', 'Target Set', 'Location', 'best');
                    grid on;

                    % Save plot
                    fig_file = fullfile(output_dir, sprintf('plot_%s_last%d_%s.png', pair_names{i}, last_n, timestamp));
                    saveas(h, fig_file);
                    fprintf('Saved: %s\n', fig_file);

                    close(h);

                catch ME
                    fprintf('Warning: Could not create last %d plot for %s: %s\n', last_n, pair_names{i}, ME.message);
                end

            end

        else
            fprintf('Only %d step(s) available, skipping last 5 sets plots.\n', num_steps);
        end

    end

    % Create combined plot (ALL SETS)
    try
        h_combined = figure('Position', [100, 100, 1200, 800], 'Visible', 'off');

        for i = 1:6
            dims = dim_pairs(i, :);

            subplot(2, 3, i);

            % Plot reachable set
            plot(reachable_set, dims, 'FaceColor', [0.8, 0.8, 1], 'EdgeColor', 'blue', 'LineWidth', 1);
            hold on;

            % Plot initial and target sets
            plot(X0, dims, 'FaceColor', 'green', 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
            plot(target, dims, 'FaceColor', 'red', 'EdgeColor', [0.8, 0, 0], 'LineWidth', 2);

            xlabel(sprintf('x_%d', dims(1)));
            ylabel(sprintf('x_%d', dims(2)));
            title(pair_names{i});
            grid on;

            if i == 1
                legend('Reachable Set', 'Initial Set', 'Target Set', 'Location', 'best', 'FontSize', 8);
            end

        end

        sgtitle('CP-LQR System - All Projections (All Sets)');

        combined_file = fullfile(output_dir, sprintf('plot_all_projections_all_%s.png', timestamp));
        saveas(h_combined, combined_file);
        fprintf('Saved combined plot: %s\n', combined_file);

        close(h_combined);

    catch ME
        fprintf('Warning: Could not create combined plot: %s\n', ME.message);
    end

    fprintf('--- Visualization Complete ---\n');

else
    fprintf('--- Cannot Create Visualizations ---\n');
    fprintf('No reachable set data available.\n');
end

fprintf('\n=== Visualization Script Complete ===\n');
