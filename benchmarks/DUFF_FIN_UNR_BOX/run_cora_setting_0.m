%% CORA Reachability Analysis - Setting: default_lin
% This script runs a single CORA reachability analysis on the Lotka-Volterra system.

clear; clc; close all;

%% ================== SCRIPT CONFIGURATION ==================
% 1. Define the CORA version to use
cora_version = 'CORA-v2025.2.0'; % Or 'CORA-v2024.4.1'

% 2. Define the specific reachability setting for this script
setting.name = 'default_lin';
setting.alg = 'lin';
setting.taylorTerms = 4;
setting.zonotopeOrder = 50;
setting.timeStep = 0.01;

% 3. Define the configuration file for the benchmark problem
config_file = '../../configs/benchmark_DUFF_FIN_UNR_BOX.json';
%% ==========================================================

%% Prepare Environment
fprintf('--- Initializing Environment for CORA %s ---\n', cora_version);
cora_root_dir = fullfile(fileparts(mfilename('fullpath')), '../../tools/cora');
addpath(fullfile(cora_root_dir, 'utils'));
init_cora_environment(cora_version);

%% Load Benchmark and Setup System
fprintf('--- Loading Benchmark: %s ---\n', config_file);
config = load_benchmark_config(config_file);
sys = create_cora_system(config);
[X0, target] = create_cora_sets(config);
[params, options] = setup_cora_params(config, setting);

% Set initial set in parameters
params.R0 = X0;

%% Run Reachability Analysis
fprintf('--- Starting Reachability Analysis (%s) ---\n', setting.name);
tic;

try
    reachable_set = reach(sys, params, options);
    result.success = true;
    result.error_message = '';
catch ME
    reachable_set = [];
    result.success = false;
    result.error_message = ME.message;
end

result.computation_time = toc;
result.reachable_set = reachable_set;

% Perform verification check
if result.success
    result.target_reached = check_reachability(result.reachable_set, target);
else
    result.target_reached = false;
end

fprintf('--- Analysis Complete ---\n');
fprintf('Success: %d | Target Reached: %d | Time: %.2f s\n', ...
    result.success, result.target_reached, result.computation_time);

%% Save Results
fprintf('--- Saving Results ---\n');
output_dir = fileparts(mfilename('fullpath')); % Just pass the benchmark directory

save_analysis_results(result, config, setting, X0, target, output_dir);

fprintf('\n=== Script Finished ===\n');
