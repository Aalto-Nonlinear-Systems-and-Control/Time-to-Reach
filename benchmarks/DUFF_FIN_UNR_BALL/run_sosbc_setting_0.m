clear; clc; close all;

% SOSBC Benchmark: benchmark_DUFF_FIN_UNR_BALL - Using new pipeline
fprintf('=== SOSBC Benchmark: benchmark_DUFF_FIN_UNR_BALL ===\n');

% Add utilities
sosbc_root_dir = fullfile(fileparts(mfilename('fullpath')), '../../tools/sosbc');
addpath(fullfile(sosbc_root_dir, 'utils'));

% Configuration
config_file = '../../configs/benchmark_DUFF_FIN_UNR_BALL.json';
fprintf('Loading config: %s\n', config_file);

% Solver options
options.deg_B = 8; % Barrier certificate degree
options.deg_s = 6; % SOS multiplier degree
options.epsilon = 1e-6; % Small positive constant
options.solver = 'mosek';

fprintf('Solver options: deg_B=%d, deg_s=%d, epsilon=%.1e, solver=%s\n', ...
    options.deg_B, options.deg_s, options.epsilon, options.solver);

% Run SOSBC pipeline
fprintf('\nRunning SOSBC pipeline...\n');
result = sosbc_pipeline(config_file, options);

% Create results directory
output_dir = fileparts(mfilename('fullpath'));
results_dir = fullfile(output_dir, 'results', 'sosbc');

% Save results
save_sosbc_results(result, results_dir, true);

% Display summary
if result.success
    fprintf('\n✓ SOSBC completed successfully!\n');
    fprintf('  Computation time: %.2f seconds\n', result.computation_time);
    fprintf('  Barrier certificate found\n');
else
    fprintf('\n✗ SOSBC failed to find barrier certificate\n');
    fprintf('  Computation time: %.2f seconds\n', result.computation_time);
end

fprintf('\n=== Benchmark Complete ===\n');
