function run_sosbc(problemPath, outPath)
% RUN_SOSBC - search a polynomial barrier certificate B with SOSTOOLS; called
% by tools/sosbc/run.py, which writes the problem (polynomial data only).
%
% With semialgebraic descriptions {g >= 0} of each initial set, each target
% set and the domain box D, the program asks for
%   -B - 1 - sum s g in SOS    on every initial set (B <= -1)
%    B - 1 - sum s g in SOS    on every target set  (B >= 1)
%   -L - sum s h     in SOS    on D, L = grad(B) . p  (non-increasing B)
%    B - 1 - lam*(x_i - b) - sum s h in SOS   on each face of D (B >= 1)
% where p is the polynomial vector field. A trajectory from an initial set then cannot leave
% {B <= 0} through the boundary of D nor reach a target: unreachable for all
% times. Infeasibility at a given degree proves nothing.

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, 'SOSTOOLS')));
P = jsondecode(fileread(problemPath));
n = P.n;
x = mpvar('x', [n 1]);
dB = P.deg_B;

t0 = tic;
prog = sosprogram(x);
[prog, B] = sospolyvar(prog, monomials(x, 0:dB));
field = aux_polys(P.vector_field, x);
L = 0;
for i = 1:n
    L = L + diff(B, x(i)) * field{i};
end
dL = dB - 1 + P.deg_field;
sets0 = aux_cell(P.initial);
setsF = aux_cell(P.target);
for k = 1:numel(sets0)
    prog = aux_on_set(prog, -B - 1, sets0{k}, x, dB);
end
for k = 1:numel(setsF)
    prog = aux_on_set(prog, B - 1, setsF{k}, x, dB);
end
D = struct('kind', 'box', 'bounds', P.domain);
prog = aux_on_set(prog, -L, D, x, dL);
if P.boundary_condition
    for i = 1:n
        for side = 1:2
            [prog, lam] = sospolyvar(prog, monomials(x, 0:dB - 1));
            face = D;
            face.bounds(i, :) = [-Inf, Inf];  % no box constraint in the fixed coordinate
            prog = aux_on_set(prog, B - 1 - lam * (x(i) - P.domain(i, side)), face, x, dB);
        end
    end
end
setup = toc(t0);

opts.solver = 'mosek';
t1 = tic;
prog = sossolve(prog, opts);
solve = toc(t1);
info = prog.solinfo.info;
Bsol = sosgetsol(prog, B);

out = struct();
out.pinf = info.pinf;
out.dinf = info.dinf;
out.numerr = info.numerr;
out.feasratio = NaN;
if isfield(info, 'feasratio')
    out.feasratio = info.feasratio;
end
out.barrier = aux_coefficients(Bsol, x);
out.times = struct('setup', setup, 'solve', solve);
out.versions = struct('matlab', version, 'sostools', 'v4.03');
fid = fopen(outPath, 'w');
fprintf(fid, '%s', jsonencode(out));
fclose(fid);
end


% Auxiliary functions -----------------------------------------------------

function c = aux_cell(raw)
if iscell(raw)
    c = raw;
else
    c = num2cell(raw);
end
end

function p = aux_polys(strs, x)
strs = cellstr(strs);
p = cell(numel(strs), 1);
for i = 1:numel(strs)
    f = str2func(['@(x) ' regexprep(strs{i}, '\<x(\d+)\>', 'x($1)')]);
    p{i} = f(x);
end
end

function prog = aux_on_set(prog, expr, s, x, d)
% expr >= 0 on the set s, via SOS multipliers of the set's constraints
g = {};
switch s.kind
    case 'box'
        for i = 1:size(s.bounds, 1)
            lo = s.bounds(i, 1);
            hi = s.bounds(i, 2);
            if isfinite(lo) && isfinite(hi) && hi > lo
                g{end + 1} = (x(i) - lo) * (hi - x(i)); %#ok<AGROW>
            elseif isfinite(lo) && hi == lo
                [prog, lam] = sospolyvar(prog, monomials(x, 0:max(d - 1, 0)));
                expr = expr - lam * (x(i) - lo);  % equality x_i = lo
            end
        end
    case 'ball'
        g{end + 1} = s.radius^2 - sum((x - s.center(:)).^2);
    case 'poly'
        q = aux_polys({s.poly}, x);  % s.poly >= 0 on the set
        g{end + 1} = q{1};
        for i = 1:size(s.bounding_box, 1)
            g{end + 1} = (x(i) - s.bounding_box(i, 1)) * (s.bounding_box(i, 2) - x(i)); %#ok<AGROW>
        end
end
sigma = 0;
md = max(2 * floor((d - 2) / 2), 0);  % multipliers times quadratic constraints reach degree d
for j = 1:numel(g)
    [prog, s_j] = sossosvar(prog, monomials(x, 0:md / 2));
    sigma = sigma + s_j * g{j};
end
prog = sosineq(prog, expr - sigma);
end

function c = aux_coefficients(p, ~)
% terms of a polynomial (coefficient, exponents per variable name) for Python
c = struct('coefficients', full(double(p.coefficient(:)')), 'degmat', full(p.degmat), ...
    'varname', {p.varname(:)'});
end
