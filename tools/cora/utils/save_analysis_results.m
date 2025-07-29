function save_analysis_results(result, config, setting, X0, target, output_dir)
    % SAVE_ANALYSIS_RESULTS - Saves analysis results in three formats:
    % 1. A visualization plot (.png)
    % 2. A data file with reachable sets (.mat)
    % 3. A summary file with all metadata (.json)
    % Results are saved in tool-specific directories

    % Create tool-specific subdirectory inside the results folder
    tool_name = 'cora';
    tool_output_dir = fullfile(output_dir, 'results', tool_name, setting.name);

    if ~exist(tool_output_dir, 'dir')
        mkdir(tool_output_dir);
    end

    timestamp = datestr(now, 'yyyymmdd_HHMMSS');

    % 1. Save Visualization
    try
        h = figure('Visible', 'off');

        if result.success
            plot(result.reachable_set, [1, 2], 'FaceColor', [0.8, 0.8, 1], 'FaceAlpha', 0.6, 'EdgeColor', 'blue', 'LineWidth', 1);
        end

        hold on;
        plot(X0, [1, 2], 'FaceColor', 'green', 'FaceAlpha', 0.8, 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
        plot(target, [1, 2], 'FaceColor', 'red', 'FaceAlpha', 0.8, 'EdgeColor', [0.8, 0, 0], 'LineWidth', 2);
        title(sprintf('%s - %s', config.benchmark.name, setting.name));
        legend('Reachable Set', 'Initial Set', 'Target Set');
        xlabel('x_1');
        ylabel('x_2');
        grid on;

        fig_file = fullfile(tool_output_dir, sprintf('plot_%s.png', timestamp));
        saveas(h, fig_file);
        fprintf('Saved plot to %s\n', fig_file);
        close(h);
    catch ME
        fprintf('Warning: Could not save plot. Reason: %s\n', ME.message);
    end

    % 2. Save .mat file with CORA objects converted to avoid deprecation warnings
    mat_file = fullfile(tool_output_dir, sprintf('data_%s.mat', timestamp));

    % Convert CORA objects to basic representations to avoid deprecation warnings
    data_to_save = struct();
    data_to_save.config = config;
    data_to_save.setting = setting;

    % Convert result and reachable sets
    data_to_save.result = struct();
    data_to_save.result.success = result.success;
    data_to_save.result.target_reached = result.target_reached;
    data_to_save.result.computation_time = result.computation_time;
    data_to_save.result.error_message = result.error_message;

    % Convert reachable sets (avoid saving CORA objects directly)
    if result.success && ~isempty(result.reachable_set)

        try
            % Extract basic information from reachable set
            data_to_save.result.reachable_set_type = class(result.reachable_set);

            % Save time interval sets if they exist
            if isfield(result.reachable_set, 'timeInterval') && isfield(result.reachable_set.timeInterval, 'set')
                data_to_save.result.num_time_intervals = length(result.reachable_set.timeInterval.set);
                % Convert each zonotope in the time interval to basic representation
                for i = 1:min(length(result.reachable_set.timeInterval.set), 10) % Limit to first 10 to avoid huge files
                    zono = result.reachable_set.timeInterval.set{i};

                    if isa(zono, 'zonotope')
                        data_to_save.result.timeInterval_centers{i} = center(zono);
                        data_to_save.result.timeInterval_generators{i} = generators(zono);
                    end

                end

            end

            % Save time point sets if they exist
            if isfield(result.reachable_set, 'timePoint') && isfield(result.reachable_set.timePoint, 'set')
                data_to_save.result.num_time_points = length(result.reachable_set.timePoint.set);
            end

        catch ME
            % Fallback: save basic info only
            data_to_save.result.reachable_set_conversion_error = ME.message;
        end

    else
        data_to_save.result.reachable_set = [];
    end

    % Convert zonotopes to basic matrix representation
    try

        if isa(X0, 'zonotope')
            data_to_save.X0_center = center(X0);
            data_to_save.X0_generators = generators(X0);
            data_to_save.X0_type = 'zonotope';
        else
            data_to_save.X0 = X0;
        end

        if isa(target, 'zonotope')
            data_to_save.target_center = center(target);
            data_to_save.target_generators = generators(target);
            data_to_save.target_type = 'zonotope';
        else
            data_to_save.target = target;
        end

    catch
        % Fallback: save as-is if conversion fails
        data_to_save.X0 = X0;
        data_to_save.target = target;
    end

    save(mat_file, '-struct', 'data_to_save');
    fprintf('Saved data to %s\n', mat_file);

    % 3. Save JSON summary
    summary.benchmark_name = config.benchmark.name;
    summary.setting_name = setting.name;
    summary.cora_version = evalin('base', 'cora_version'); % Get from base workspace
    summary.analysis_timestamp = timestamp;
    summary.success = result.success;
    summary.target_reached = result.target_reached;
    summary.computation_time_seconds = result.computation_time;
    summary.error_message = result.error_message;
    summary.setting_details = setting;

    json_file = fullfile(tool_output_dir, sprintf('summary_%s.json', timestamp));
    fid = fopen(json_file, 'w');
    fprintf(fid, '%s', jsonencode(summary, 'PrettyPrint', true));
    fclose(fid);
    fprintf('Saved summary to %s\n', json_file);

    fprintf('All results saved to tool-specific directory: %s\n', tool_output_dir);
end
