%% 创建MAS-CON基准测试所有维度对的可达集可视化
% 此脚本重新运行CORA分析并创建所有维度对的图形

clear; clc; close all;

fprintf('=== Creating All Dimension Visualizations for MAS-CON ===\n');

%% 设置环境
% 1. 定义CORA版本
cora_version = 'CORA-v2025.2.0';

% 2. 配置文件
config_file = '../../configs/benchmark_MAS_CON_FIN_REA_BOX.json';

% 3. 设置信息（使用较快的设置进行可视化）
setting.name = 'visualization';
setting.alg = 'lin';
setting.taylorTerms = 3; % 减少Taylor项以加快速度
setting.zonotopeOrder = 30; % 减少zonotope阶数
setting.timeStep = 0.05; % 增大时间步长

%% 初始化环境
fprintf('--- Initializing CORA Environment ---\n');
cora_root_dir = fullfile(fileparts(mfilename('fullpath')), '../../tools/cora');
addpath(fullfile(cora_root_dir, 'utils'));
init_cora_environment(cora_version);

%% 加载配置和创建系统对象
fprintf('--- Loading Configuration ---\n');
config = load_benchmark_config(config_file);
sys = create_cora_system(config);
[X0, target] = create_cora_sets(config);
[params, options] = setup_cora_params(config, setting);

% 设置初始集合
params.R0 = X0;

%% 运行可达性分析
fprintf('--- Running Reachability Analysis for Visualization ---\n');
fprintf('Using faster settings: timeStep=%.3f, taylorTerms=%d, zonotopeOrder=%d\n', ...
    setting.timeStep, setting.taylorTerms, setting.zonotopeOrder);

tic;

try
    reachable_set = reach(sys, params, options);
    result.success = true;
    result.reachable_set = reachable_set;
    result.error_message = '';
    fprintf('Analysis completed successfully.\n');
catch ME
    result.success = false;
    result.reachable_set = [];
    result.error_message = ME.message;
    fprintf('Analysis failed: %s\n', ME.message);
end

result.computation_time = toc;
fprintf('Computation time: %.2f seconds\n', result.computation_time);

%% 生成所有维度对的可视化
if result.success && ~isempty(result.reachable_set)
    fprintf('--- Creating All Dimension Visualizations ---\n');
    output_dir = fileparts(mfilename('fullpath'));

    % 调用可视化函数
    visualize_all_dimensions(result, X0, target, config, setting, output_dir);

    fprintf('--- Visualization Complete ---\n');
    fprintf('All plots have been saved to the results directory.\n');

    % 列出生成的文件
    results_dir = fullfile(output_dir, 'results', 'cora', setting.name);

    if exist(results_dir, 'dir')
        files = dir(fullfile(results_dir, '*.png'));
        fprintf('\nGenerated files:\n');

        for i = 1:length(files)
            fprintf('  %s\n', files(i).name);
        end

    end

else
    fprintf('--- Cannot Create Full Visualizations ---\n');
    fprintf('Analysis was not successful. Creating basic plots with initial and target sets only.\n');

    % 创建仅显示初始集合和目标集合的图形
    dimension_pairs = [1, 2; 3, 4; 5, 6; 7, 8; 9, 10; 11, 12; 13, 14; 15, 16];
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');

    output_dir = fileparts(mfilename('fullpath'));
    tool_output_dir = fullfile(output_dir, 'results', 'cora', setting.name);

    if ~exist(tool_output_dir, 'dir')
        mkdir(tool_output_dir);
    end

    % 创建综合图
    h_combined = figure('Name', 'All Dimension Pairs (Initial & Target)', ...
        'Position', [100, 100, 1200, 900], ...
        'Visible', 'off');

    for i = 1:size(dimension_pairs, 1)
        dims = dimension_pairs(i, :);

        subplot(2, 4, i);

        % 绘制初始集合
        plot(X0, dims, 'FaceColor', 'green', 'FaceAlpha', 0.8, ...
            'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
        hold on;

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
            legend('Initial Set', 'Target Set', 'Location', 'best', 'FontSize', 8);
        end

    end

    % 设置综合图标题
    sgtitle(sprintf('%s - All Dimension Pairs (Initial & Target Sets)', config.benchmark.name));

    % 保存综合图
    combined_file = fullfile(tool_output_dir, ...
        sprintf('plot_initial_target_all_dims_%s.png', timestamp));
    saveas(h_combined, combined_file);
    fprintf('Saved initial/target plot to %s\n', combined_file);

    close(h_combined);
end

fprintf('\n=== Script Complete ===\n');
