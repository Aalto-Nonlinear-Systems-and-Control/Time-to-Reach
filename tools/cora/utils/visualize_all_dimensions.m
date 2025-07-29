function visualize_all_dimensions(result, X0, target, config, setting, output_dir)
    % VISUALIZE_ALL_DIMENSIONS - 绘制16维系统的所有维度对的可达集
    % 特别适用于MAS-CON基准测试，绘制[1,2], [3,4], ..., [15,16]维度对
    %
    % Inputs:
    %   result     - 成功的分析结果结构体
    %   X0         - 初始集合 (CORA对象)
    %   target     - 目标集合 (CORA对象)
    %   config     - 基准配置结构体
    %   setting    - 分析设置结构体
    %   output_dir - 输出目录

    if ~result.success
        fprintf('Warning: Analysis was not successful. Cannot visualize results.\n');
        return;
    end

    % 定义维度对
    dimension_pairs = [1, 2; 3, 4; 5, 6; 7, 8; 9, 10; 11, 12; 13, 14; 15, 16];

    % 创建时间戳
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');

    % 创建输出目录
    tool_output_dir = fullfile(output_dir, 'results', 'cora', setting.name);

    if ~exist(tool_output_dir, 'dir')
        mkdir(tool_output_dir);
    end

    fprintf('Creating visualizations for all dimension pairs...\n');

    % 为每个维度对创建单独的图
    for i = 1:size(dimension_pairs, 1)
        dims = dimension_pairs(i, :);

        try
            % 创建图形
            h = figure('Name', sprintf('Dimensions %d-%d', dims(1), dims(2)), ...
                'Position', [100 + (i - 1) * 50, 100 + (i - 1) * 50, 800, 600], ...
                'Visible', 'off');

            % 绘制可达集
            if ~isempty(result.reachable_set)
                plot(result.reachable_set, dims, ...
                    'FaceColor', [0.8, 0.8, 1], 'FaceAlpha', 0.6, ...
                    'EdgeColor', 'blue', 'LineWidth', 1);
                hold on;
            end

            % 绘制初始集合
            plot(X0, dims, 'FaceColor', 'green', 'FaceAlpha', 0.8, ...
                'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);

            % 绘制目标集合
            plot(target, dims, 'FaceColor', 'red', 'FaceAlpha', 0.8, ...
                'EdgeColor', [0.8, 0, 0], 'LineWidth', 2);

            % 设置图形属性
            xlabel(sprintf('x_{%d}', dims(1)));
            ylabel(sprintf('x_{%d}', dims(2)));
            title(sprintf('%s - Dimensions %d-%d (%s)', ...
                config.benchmark.name, dims(1), dims(2), setting.name));
            legend('Reachable Set', 'Initial Set', 'Target Set', 'Location', 'best');
            grid on;
            axis equal;

            % 保存图形
            fig_file = fullfile(tool_output_dir, ...
                sprintf('plot_dims_%d_%d_%s.png', dims(1), dims(2), timestamp));
            saveas(h, fig_file);
            fprintf('Saved plot for dimensions %d-%d to %s\n', dims(1), dims(2), fig_file);

            close(h);

        catch ME
            fprintf('Warning: Could not create plot for dimensions %d-%d. Error: %s\n', ...
                dims(1), dims(2), ME.message);
        end

    end

    % 创建综合图：所有维度对在一个图中
    try
        h_combined = figure('Name', 'All Dimension Pairs', ...
            'Position', [100, 100, 1200, 900], ...
            'Visible', 'off');

        for i = 1:size(dimension_pairs, 1)
            dims = dimension_pairs(i, :);

            subplot(2, 4, i);

            % 绘制可达集
            if ~isempty(result.reachable_set)
                plot(result.reachable_set, dims, ...
                    'FaceColor', [0.8, 0.8, 1], 'FaceAlpha', 0.6, ...
                    'EdgeColor', 'blue', 'LineWidth', 1);
                hold on;
            end

            % 绘制初始集合
            plot(X0, dims, 'FaceColor', 'green', 'FaceAlpha', 0.8, ...
                'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);

            % 绘制目标集合
            plot(target, dims, 'FaceColor', 'red', 'FaceAlpha', 0.8, ...
                'EdgeColor', [0.8, 0, 0], 'LineWidth', 2);

            % 设置子图属性
            xlabel(sprintf('x_{%d}', dims(1)));
            ylabel(sprintf('x_{%d}', dims(2)));
            title(sprintf('Dims %d-%d', dims(1), dims(2)));
            grid on;
            axis equal;

            % 仅在第一个子图中添加图例
            if i == 1
                legend('Reachable Set', 'Initial Set', 'Target Set', ...
                    'Location', 'best', 'FontSize', 8);
            end

        end

        % 设置综合图标题
        sgtitle(sprintf('%s - All Dimension Pairs (%s)', ...
            config.benchmark.name, setting.name));

        % 保存综合图
        combined_file = fullfile(tool_output_dir, ...
            sprintf('plot_all_dimensions_%s.png', timestamp));
        saveas(h_combined, combined_file);
        fprintf('Saved combined plot to %s\n', combined_file);

        close(h_combined);

    catch ME
        fprintf('Warning: Could not create combined plot. Error: %s\n', ME.message);
    end

    fprintf('Visualization complete. All plots saved to: %s\n', tool_output_dir);
end
