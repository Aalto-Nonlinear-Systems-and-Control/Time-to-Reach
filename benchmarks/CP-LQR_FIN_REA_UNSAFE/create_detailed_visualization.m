%% CP-LQR Detailed Visualization Script for Small Target Sets
% Creates focused visualization around the target set to better observe intersection

clear; clc; close all;

%% Configuration
cora_version = 'CORA-v2025.2.0';
config_file = '../../configs/benchmark_CP_LQR_REA_UNSAFE.json';
setting.name = 'default_nonlinear';
setting.alg = 'lin';
setting.taylorTerms = 3;
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

%% Run Analysis for Visualization
fprintf('--- Running Analysis for Detailed Visualization ---\n');

try
    reachable_set = reach(sys, params, options);
    success = true;

    fprintf('Reachable set type: %s\n', class(reachable_set));

catch ME
    fprintf('Error: %s\n', ME.message);
    success = false;
    reachable_set = [];
end

%% Create Detailed Visualizations
if success && ~isempty(reachable_set)
    fprintf('--- Creating Detailed Visualizations ---\n');

    % Define dimension pairs for 4D system
    dim_pairs = [1, 2; 1, 3; 1, 4; 2, 3; 2, 4; 3, 4];
    pair_names = {'x1-x2', 'x1-x3', 'x1-x4', 'x2-x3', 'x2-x4', 'x3-x4'};

    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'results', 'cora', setting.name);

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Create detailed plots with focus on target area
    for i = 1:size(dim_pairs, 1)
        dims = dim_pairs(i, :);

        try
            % Create two plots: normal view and zoomed view

            % 1. Normal view
            h1 = figure('Visible', 'off');
            plot(reachable_set, dims, 'FaceColor', [0.8, 0.8, 1], 'EdgeColor', 'blue', 'LineWidth', 1);
            hold on;
            plot(X0, dims, 'FaceColor', 'green', 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
            plot(target, dims, 'FaceColor', 'red', 'EdgeColor', [0.8, 0, 0], 'LineWidth', 3);

            xlabel(sprintf('x_%d', dims(1)));
            ylabel(sprintf('x_%d', dims(2)));
            title(sprintf('CP-LQR System - %s Projection (Normal View)', pair_names{i}));
            legend('Reachable Set', 'Initial Set', 'Target Set', 'Location', 'best');
            grid on;

            fig_file = fullfile(output_dir, sprintf('plot_%s_normal_%s.png', pair_names{i}, timestamp));
            saveas(h1, fig_file);
            fprintf('Saved normal view: %s\n', fig_file);
            close(h1);

            % 2. Zoomed view around target set
            h2 = figure('Visible', 'off');
            plot(reachable_set, dims, 'FaceColor', [0.8, 0.8, 1], 'EdgeColor', 'blue', 'LineWidth', 1);
            hold on;
            plot(X0, dims, 'FaceColor', 'green', 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
            plot(target, dims, 'FaceColor', 'red', 'EdgeColor', [0.8, 0, 0], 'LineWidth', 4);

            % Set zoom limits around target (±0.1 from target center)
            target_center = [0, 0]; % Target is centered at origin
            zoom_range = 0.1;
            xlim([target_center(1) - zoom_range, target_center(1) + zoom_range]);
            ylim([target_center(2) - zoom_range, target_center(2) + zoom_range]);

            xlabel(sprintf('x_%d', dims(1)));
            ylabel(sprintf('x_%d', dims(2)));
            title(sprintf('CP-LQR System - %s Projection (Zoomed around Target)', pair_names{i}));
            legend('Reachable Set', 'Initial Set', 'Target Set', 'Location', 'best');
            grid on;

            fig_file = fullfile(output_dir, sprintf('plot_%s_zoomed_%s.png', pair_names{i}, timestamp));
            saveas(h2, fig_file);
            fprintf('Saved zoomed view: %s\n', fig_file);
            close(h2);

            % 3. Ultra-zoomed view (±0.02 from target center)
            h3 = figure('Visible', 'off');
            plot(reachable_set, dims, 'FaceColor', [0.8, 0.8, 1], 'EdgeColor', 'blue', 'LineWidth', 1);
            hold on;
            plot(X0, dims, 'FaceColor', 'green', 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
            plot(target, dims, 'FaceColor', 'red', 'EdgeColor', [0.8, 0, 0], 'LineWidth', 5);

            % Set ultra-zoom limits
            ultra_zoom_range = 0.02;
            xlim([target_center(1) - ultra_zoom_range, target_center(1) + ultra_zoom_range]);
            ylim([target_center(2) - ultra_zoom_range, target_center(2) + ultra_zoom_range]);

            xlabel(sprintf('x_%d', dims(1)));
            ylabel(sprintf('x_%d', dims(2)));
            title(sprintf('CP-LQR System - %s Projection (Ultra-Zoomed)', pair_names{i}));
            legend('Reachable Set', 'Initial Set', 'Target Set', 'Location', 'best');
            grid on;

            fig_file = fullfile(output_dir, sprintf('plot_%s_ultra_%s.png', pair_names{i}, timestamp));
            saveas(h3, fig_file);
            fprintf('Saved ultra-zoomed view: %s\n', fig_file);
            close(h3);

        catch ME
            fprintf('Warning: Could not create detailed plots for %s: %s\n', pair_names{i}, ME.message);
        end

    end

    % Create combined normal view
    try
        h_combined = figure('Position', [100, 100, 1200, 800], 'Visible', 'off');

        for i = 1:6
            dims = dim_pairs(i, :);
            subplot(2, 3, i);

            plot(reachable_set, dims, 'FaceColor', [0.8, 0.8, 1], 'EdgeColor', 'blue', 'LineWidth', 1);
            hold on;
            plot(X0, dims, 'FaceColor', 'green', 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
            plot(target, dims, 'FaceColor', 'red', 'EdgeColor', [0.8, 0, 0], 'LineWidth', 3);

            xlabel(sprintf('x_%d', dims(1)));
            ylabel(sprintf('x_%d', dims(2)));
            title(sprintf('%s (Normal)', pair_names{i}));
            grid on;

            if i == 1
                legend('Reachable Set', 'Initial Set', 'Target Set', 'Location', 'best', 'FontSize', 8);
            end

        end

        sgtitle('CP-LQR System - All Projections (Normal View)');
        combined_file = fullfile(output_dir, sprintf('plot_all_normal_%s.png', timestamp));
        saveas(h_combined, combined_file);
        fprintf('Saved combined normal view: %s\n', combined_file);
        close(h_combined);

    catch ME
        fprintf('Warning: Could not create combined normal plot: %s\n', ME.message);
    end

    % Create combined zoomed view
    try
        h_combined_zoom = figure('Position', [100, 100, 1200, 800], 'Visible', 'off');

        for i = 1:6
            dims = dim_pairs(i, :);
            subplot(2, 3, i);

            plot(reachable_set, dims, 'FaceColor', [0.8, 0.8, 1], 'EdgeColor', 'blue', 'LineWidth', 1);
            hold on;
            plot(X0, dims, 'FaceColor', 'green', 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
            plot(target, dims, 'FaceColor', 'red', 'EdgeColor', [0.8, 0, 0], 'LineWidth', 4);

            % Set zoom limits
            xlim([-0.1, 0.1]);
            ylim([-0.1, 0.1]);

            xlabel(sprintf('x_%d', dims(1)));
            ylabel(sprintf('x_%d', dims(2)));
            title(sprintf('%s (Zoomed)', pair_names{i}));
            grid on;

            if i == 1
                legend('Reachable Set', 'Initial Set', 'Target Set', 'Location', 'best', 'FontSize', 8);
            end

        end

        sgtitle('CP-LQR System - All Projections (Zoomed around Target)');
        combined_zoom_file = fullfile(output_dir, sprintf('plot_all_zoomed_%s.png', timestamp));
        saveas(h_combined_zoom, combined_zoom_file);
        fprintf('Saved combined zoomed view: %s\n', combined_zoom_file);
        close(h_combined_zoom);

    catch ME
        fprintf('Warning: Could not create combined zoomed plot: %s\n', ME.message);
    end

    % Create combined ultra-zoomed view
    try
        h_combined_ultra = figure('Position', [100, 100, 1200, 800], 'Visible', 'off');

        for i = 1:6
            dims = dim_pairs(i, :);
            subplot(2, 3, i);

            plot(reachable_set, dims, 'FaceColor', [0.8, 0.8, 1], 'EdgeColor', 'blue', 'LineWidth', 1);
            hold on;
            plot(X0, dims, 'FaceColor', 'green', 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
            plot(target, dims, 'FaceColor', 'red', 'EdgeColor', [0.8, 0, 0], 'LineWidth', 5);

            % Set ultra-zoom limits
            xlim([-0.02, 0.02]);
            ylim([-0.02, 0.02]);

            xlabel(sprintf('x_%d', dims(1)));
            ylabel(sprintf('x_%d', dims(2)));
            title(sprintf('%s (Ultra-Zoom)', pair_names{i}));
            grid on;

            if i == 1
                legend('Reachable Set', 'Initial Set', 'Target Set', 'Location', 'best', 'FontSize', 8);
            end

        end

        sgtitle('CP-LQR System - All Projections (Ultra-Zoomed around Target)');
        combined_ultra_file = fullfile(output_dir, sprintf('plot_all_ultra_%s.png', timestamp));
        saveas(h_combined_ultra, combined_ultra_file);
        fprintf('Saved combined ultra-zoomed view: %s\n', combined_ultra_file);
        close(h_combined_ultra);

    catch ME
        fprintf('Warning: Could not create combined ultra-zoomed plot: %s\n', ME.message);
    end

    fprintf('--- Detailed Visualization Complete ---\n');
    fprintf('Generated 3 zoom levels for each dimension pair:\n');
    fprintf('1. Normal view - Full range\n');
    fprintf('2. Zoomed view - Target ±0.1\n');
    fprintf('3. Ultra-zoomed view - Target ±0.02\n');

else
    fprintf('--- Cannot Create Detailed Visualizations ---\n');
    fprintf('No reachable set data available.\n');
end

fprintf('\n=== Detailed Visualization Script Complete ===\n');
