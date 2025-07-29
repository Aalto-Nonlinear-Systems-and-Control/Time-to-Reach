%% CORA Reachability Analysis - Setting: default_lin
% This script runs a single CORA reachability analysis on the Lotka-Volterra system.

clear; clc; close all;

%% ================== SCRIPT CONFIGURATION ==================
% 1. Define the CORA version to use
cora_version = 'CORA-v2025.2.0'; % Or 'CORA-v2024.4.1'

% 2. Define the specific reachability setting for this script
setting.name = 'setting_0';
setting.alg = 'krylov';
setting.taylorTerms = 4;
setting.tensorOrder = 2;
setting.zonotopeOrder = 50;
setting.timeStep = 0.01;
setting.tFinal = 5;

% 3. Configuration for heat3d benchmark
% (No external config file needed for this benchmark)
%% ==========================================================

%% Prepare Environment
fprintf('--- Initializing Environment for CORA %s ---\n', cora_version);
cora_root_dir = fullfile(fileparts(mfilename('fullpath')), '../../tools/cora');
addpath(fullfile(cora_root_dir, 'utils'));
init_cora_environment(cora_version);

%% Setup

% load the heat5.mat file
load('../../data/heat5.mat');
sys = linearSys('heat125', A, []);

% initial a zonotope with a center of 1 and a radius of 0.1
X0c = [100; zeros(124, 1)];
X0 = zonotope(X0c, 0.01 * eye(125));
% X0 = zonotope(X0c, zeros(125));

% Set initial set in parameters
params.R0 = X0;
params.U = zonotope(0);
params.tFinal = setting.tFinal;
options.timeStep = setting.timeStep;
options.taylorTerms = setting.taylorTerms;
options.zonotopeOrder = setting.zonotopeOrder;
options.alg = setting.alg;
options.tensorOrder = setting.tensorOrder;

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

fprintf('--- Analysis Complete ---\n');
% fprintf('Success: %d | Target Reached: %d | Time: %.2f s\n', ...
%     result.success, result.target_reached, result.computation_time);
fprintf('Time: %.2f s\n', result.computation_time);

%% Save Results
fprintf('--- Saving Results ---\n');
output_dir = fileparts(mfilename('fullpath')); % Just pass the benchmark directory

% save_analysis_results(result, setting, X0, output_dir);

%% Verification: Check if any cell exceeds maximum allowed temperature
% Define verification problem: verify that all cells stay below 0.7°C
max_allowed_temperature = 0.7;
n_dimensions = size(A, 1); % Number of cells (dimensions)

% Create target as a single point representing unsafe boundary (temperatures = 0.7°C)
% This is simply a point in n-dimensional space where all cells are at 0.7°C
target_point = max_allowed_temperature * ones(n_dimensions, 1);

fprintf('--- Starting Verification ---\n');
fprintf('Property: All cells must stay below %.2f°C\n', max_allowed_temperature);
fprintf('System has %d cells (dimensions)\n', n_dimensions);
fprintf('Target point: [%.2f, %.2f, ..., %.2f] (all %d cells at %.2f°C)\n', ...
    max_allowed_temperature, max_allowed_temperature, max_allowed_temperature, n_dimensions, max_allowed_temperature);

% Check reachability: is the target point contained in any reachable set?
is_safe = true;
target_reached = false;

% Check if target point is contained in any reachable set
for i = 1:length(result.reachable_set.timeInterval.set)
    % Get the current reachable set (zonotope)
    this_reachable_set = result.reachable_set.timeInterval.set{i};

    % Check if target point is inside this reachable set
    if contains(this_reachable_set, target_point)
        target_reached = true;
        is_safe = false;
        fprintf('Target point contained in reachable set at time step %d!\n', i);
        fprintf('This means all cells can reach %.2f°C simultaneously\n', max_allowed_temperature);
        break;
    end

end

% Store verification results
result.target_reached = target_reached;
result.is_safe = is_safe;

% Display verification results
fprintf('--- Verification Complete ---\n');
fprintf('Target point reachable: %d\n', target_reached);

if is_safe
    fprintf('Result: ✓ SAFE - Target point (all cells at %.2f°C) is NOT reachable\n', max_allowed_temperature);
else
    fprintf('Result: ✗ UNSAFE - Target point (all cells at %.2f°C) IS reachable\n', max_allowed_temperature);
end

% visualize the reachable set for the dimensions [1,2] in a 2D plot, using the CORA's plot function
% specify the dimensions to check and plot
dim_to_check = [1, 65];

% project the reachable set to the dimensions to check
reachable_set_projected = project(result.reachable_set, dim_to_check);

%

% plot the projected reachable set
figure;
plot(reachable_set_projected);
title('Reachable Set Projection');
xlabel('Dimension 1');
ylabel('Dimension 65');

% save the figure
saveas(gcf, 'reachable_set_projection.png');

fprintf('\n=== Script Finished ===\n');
