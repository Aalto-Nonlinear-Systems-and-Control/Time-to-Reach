function run_cora(configPath, settingsPath, outPath)
% RUN_CORA - CORA reachability on one benchmark; called by tools/cora/run.py.
%
% The benchmark is a forward problem from one initial box to one target box;
% tools/cora/run.py reports every other benchmark as unsupported. The vector
% field is generated from the config's equation strings. Every time-interval
% reachable set is compared with the target box:
%   disjoint  no intersection (CORA's isIntersecting)
%   inside    the interval hull of the set lies in the target (CORA's contains)
%   possible  otherwise
% A set inside the target proves that the target is reached: the reachable
% sets are over-approximations, so every trajectory is then in the target. No
% intersection at all proves that the target is not reached. An intersection
% alone proves nothing. A reachable set that does not cover the horizon (CORA
% stops on set explosion) is reported as such.

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, 'CORA')));
cfg = jsondecode(fileread(configPath));
st = jsondecode(fileread(settingsPath));
if isfield(st, 'threads') && ~isempty(st.threads)
    maxNumCompThreads(st.threads);
end

eqs = cellstr(cfg.system.equations);
n = numel(eqs);
body = strjoin(cellfun(@(e) ['(' aux_matlab(e) ')'], eqs, 'UniformOutput', false), '; ');
f = str2func(['@(x,u) [' body ']']);
sys = nonlinearSys(['ttr_' lower(cfg.id)], f, n, 1);

options = struct('timeStep', st.timeStep, 'taylorTerms', st.taylorTerms, ...
    'zonotopeOrder', st.zonotopeOrder, 'alg', st.alg, 'tensorOrder', st.tensorOrder);
t0 = cfg.time_horizon(1);
t1 = cfg.time_horizon(2);
R0 = zonotope(aux_box(cfg.initial_sets));
T = aux_box(cfg.target_sets);

% warm-up: derivative generation and first calls are not part of the timed reach
params = struct('tStart', t0, 'tFinal', t0 + 2 * st.timeStep, 'R0', R0, 'U', zonotope(0));
tw = tic;
reach(sys, params, options);
warmup = toc(tw);

params.tFinal = t1;
tr = tic;
R = reach(sys, params, options);
reachTime = toc(tr);

tc = tic;
segments = zeros(0, 2);
flow = zeros(1024, 2 + 2 * n + 1);  % per segment: t0, t1, box lower bounds, upper bounds, status (0 disjoint, 1 possible, 2 inside)
m = 0;
numPossible = 0;
numInside = 0;
tEnd = -Inf;
for r = 1:numel(R)
    if isempty(R(r).timeInterval)
        continue
    end
    for i = 1:numel(R(r).timeInterval.set)
        S = R(r).timeInterval.set{i};
        I = R(r).timeInterval.time{i};
        a = infimum(I);
        e = supremum(I);
        tEnd = max(tEnd, e);
        H = interval(S);
        if ~isIntersecting(S, T)
            status = 0;
        elseif contains(T, H)
            status = 2;
        else
            status = 1;
        end
        m = m + 1;
        if m > size(flow, 1)
            flow = [flow; zeros(size(flow))]; %#ok<AGROW>
        end
        flow(m, :) = [a, e, infimum(H)', supremum(H)', status];
        numPossible = numPossible + (status == 1);
        numInside = numInside + (status == 2);
        if status > 0
            segments(end + 1, :) = [max(a, t0), min(e, t1)]; %#ok<AGROW>
        end
    end
end
checkTime = toc(tc);
covered = tEnd >= t1 - 1e-9;

if numInside > 0
    verdict = 'reachable_proven';
elseif ~covered
    verdict = 'incomplete';
elseif numPossible == 0
    verdict = 'unreachable_proven';
else
    verdict = 'reachable_possible';
end

out = struct();
out.verdict = verdict;
out.horizon_covered = covered;
out.reachable_set_end_time = tEnd;
out.num_segments = m;
out.num_possible = numPossible;
out.num_inside = numInside;
out.reach_time_bounds = aux_merge(segments);
out.times = struct('warmup', warmup, 'reach', reachTime, 'check', checkTime);
out.versions = struct('matlab', version, 'cora', CORAVERSION());
fid = fopen(outPath, 'w');
fprintf(fid, '%s', jsonencode(out));
fclose(fid);

% the interval hulls of the reachable sets behind the verdict, for plotting (scripts/report.py)
fp = struct('n', n, 'status_codes', {{'disjoint', 'possible', 'inside'}}, 'segments', round(flow(1:m, :), 6, 'significant'));
fid = fopen(fullfile(fileparts(outPath), 'flowpipe.json'), 'w');
fprintf(fid, '%s', jsonencode(fp));
fclose(fid);
end


% Auxiliary functions -----------------------------------------------------

function s = aux_matlab(e)
% 'x12^2' -> 'x(12)^2'
s = regexprep(strrep(char(e), '**', '^'), '\<x(\d+)\>', 'x($1)');
end

function I = aux_box(raw)
% the one box of a list of sets (jsondecode gives a struct for a list with one element)
if iscell(raw)
    if numel(raw) ~= 1
        error('run_cora:unsupported', 'one box is expected, got %d sets', numel(raw));
    end
    raw = raw{1};
end
if numel(raw) ~= 1 || ~strcmp(raw.type, 'box')
    error('run_cora:unsupported', 'one box is expected');
end
I = interval(raw.bounds(:, 1), raw.bounds(:, 2));
end

function m = aux_merge(seg)
m = zeros(0, 2);
if isempty(seg)
    return
end
seg = sortrows(seg);
m = seg(1, :);
for i = 2:size(seg, 1)
    if seg(i, 1) <= m(end, 2)
        m(end, 2) = max(m(end, 2), seg(i, 2));
    else
        m(end + 1, :) = seg(i, :); %#ok<AGROW>
    end
end
end
