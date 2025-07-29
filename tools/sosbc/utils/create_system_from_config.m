function system = create_system_from_config(config)
    % CREATE_SYSTEM_FROM_CONFIG - Create system from configuration
    %
    % Inputs:
    %   config - parsed JSON configuration
    %
    % Output:
    %   system - structure with vars, dynamics, and type

    system_type = config.system.type;

    % Determine system dimension
    if strcmp(system_type, 'linear')
        % For linear systems, get dimension from matrix
        A = load_matrix_from_config(config.system.dynamics);
        n = size(A, 1);
    else
        % For nonlinear systems, get dimension from number of equations
        n = length(config.system.dynamics.equations);
    end

    % Create symbolic variables dynamically
    vars = create_symbolic_variables(n);

    switch system_type
        case 'linear'
            % Linear system: ẋ = Ax
            system = create_linear_system(vars, A);

        case 'nonlinear'
            % Nonlinear polynomial system
            equations = config.system.dynamics.equations;
            system = create_nonlinear_system(vars, equations);

        otherwise
            error('Unsupported system type: %s', system_type);
    end

end

function vars = create_symbolic_variables(n)
    % Create n symbolic variables: x1, x2, ..., xn
    vars = [];

    for i = 1:n
        eval(sprintf('pvar x%d;', i));
        vars = [vars, eval(sprintf('x%d', i))];
    end

end

function A = load_matrix_from_config(dynamics_config)
    % Load matrix A from config (explicit or from .mat file)

    if isfield(dynamics_config, 'matrix')
        % Explicit matrix definition
        A = dynamics_config.matrix;

        if iscell(A)
            A = cell2mat(A);
        end

    elseif isfield(dynamics_config, 'matrix_file')
        % Load matrix from .mat file
        matrix_file = dynamics_config.matrix_file;

        % Construct full path (matrices stored in data folder)
        if ~startsWith(matrix_file, '/')
            % Relative path - prepend data folder
            matrix_file = fullfile('data', matrix_file);
        end

        if ~endsWith(matrix_file, '.mat')
            matrix_file = [matrix_file, '.mat'];
        end

        if ~exist(matrix_file, 'file')
            error('Matrix file not found: %s', matrix_file);
        end

        % Load .mat file
        data = load(matrix_file);
        field_names = fieldnames(data);
        A = data.(field_names{1}); % Take first field

    else
        error('Linear system must specify either "matrix" or "matrix_file"');
    end

end

function system = create_linear_system(vars, A)
    % Convert linear system ẋ = Ax to polynomial form
    n = length(vars);

    % Create polynomial dynamics: f_i(x) = Σ(A_ij * x_j)
    dynamics = cell(n, 1);

    for i = 1:n
        dynamics{i} = 0;

        for j = 1:n
            dynamics{i} = dynamics{i} + A(i, j) * vars(j);
        end

    end

    system.vars = vars;
    system.dynamics = dynamics;
    system.type = 'linear';
    system.matrix = A;
end

function system = create_nonlinear_system(vars, equations)
    % Convert string equations to symbolic polynomials
    n = length(vars);
    dynamics = cell(n, 1);

    for i = 1:n
        % Convert string equation to symbolic expression
        eq_str = equations{i};

        % Replace x1, x2, ... with actual symbolic variables
        for j = 1:n
            eq_str = strrep(eq_str, sprintf('x%d', j), sprintf('vars(%d)', j));
        end

        % Evaluate the string as symbolic expression
        dynamics{i} = eval(eq_str);
    end

    system.vars = vars;
    system.dynamics = dynamics;
    system.type = 'nonlinear';
end
