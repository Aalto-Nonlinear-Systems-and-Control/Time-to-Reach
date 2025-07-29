function visualize_cora_results(result, X0, target, config, output_dir)
    % VISUALIZE_CORA_RESULTS - Plots and saves the results of a CORA analysis.
    %
    % Inputs:
    %   result     - The successful result struct from the analysis.
    %   X0         - The initial set (CORA object).
    %   target     - The target set (CORA object).
    %   config     - The benchmark configuration struct.
    %   output_dir - The directory where the plot file will be saved.

    try
        figure('Name', 'CORA Reachability Analysis', 'Position', [100, 100, 800, 600], 'Visible', 'off');

        % Plot reachable sets from the successful result
        plot(result.reachable_set, [1, 2], 'FaceColor', [0.8, 0.8, 1], 'EdgeColor', 'b');
        hold on;

        % Plot initial and target sets
        plot(X0, [1, 2], 'FaceColor', 'g', 'EdgeColor', 'g', 'LineWidth', 2);
        plot(target, [1, 2], 'FaceColor', 'r', 'EdgeColor', 'r', 'LineWidth', 2);

        % Configure plot aesthetics
        xlabel('x_1');
        ylabel('x_2');
        title_str = sprintf('CORA Analysis - %s (Setting: %s)', config.benchmark.name, result.setting.name);
        title(title_str);
        legend('Reachable Set', 'Initial Set', 'Target Set', 'Location', 'best');
        grid on;
        axis equal;

        % Save the plot to a file
        plot_file = fullfile(output_dir, sprintf('cora_plot_%s.png', datestr(now, 'yyyymmdd_HHMMSS')));
        saveas(gcf, plot_file);
        fprintf('Plot saved to: %s\n', plot_file);
        close(gcf); % Close the figure window after saving

    catch ME
        fprintf('Warning: Could not create visualization: %s\n', ME.message);
    end
end