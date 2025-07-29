function sys = create_cora_system(config)
    % Creates a CORA system object from the configuration.
    % Supports:
    % - Linear systems: dx/dt = Ax + Bu
    % - Nonlinear systems: dx/dt = f(x,u)

    system_type = config.system.type;

    switch lower(system_type)
        case 'linear'
            % Linear system: dx/dt = Ax + Bu
            if isfield(config.system, 'A')
                A = config.system.A;
            elseif isfield(config.system, 'dynamics') && isfield(config.system.dynamics, 'A')
                A = config.system.dynamics.A;
            else
                error('Linear system requires matrix A in config.system.A or config.system.dynamics.A');
            end

            % 新增：如果A是字符串且以.mat结尾，则加载文件，只取A变量
            if ischar(A) && endsWith(A, '.mat')
                % 如果不是绝对路径，则补全为相对于主项目根目录的路径
                if ~isfile(A)
                    project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
                    A_full = fullfile(project_root, A);
                else
                    A_full = A;
                end

                matData = load(A_full, 'A'); % 只加载A变量

                if isfield(matData, 'A')
                    A = matData.A;
                else
                    error('The .mat file %s does not contain variable A.', A_full);
                end

            end

            sys = linearSys(A, []);

        case 'nonlinear'
            % Nonlinear system: dx/dt = f(x,u)
            if isfield(config.system.dynamics, 'equations')
                equations = config.system.dynamics.equations;
                dynamics_function = create_dynamics_function(equations);

                % Determine number of states from equations
                n_states = length(equations);
                % Autonomous system (no inputs)
                n_inputs = 0;

                sys = nonlinearSys(dynamics_function, n_states, n_inputs);
            else
                error('Nonlinear system requires equations in config.system.dynamics.equations');
            end

        otherwise
            error('Unsupported system type: %s. Use "linear" or "nonlinear"', system_type);
    end

end

function dynamics_func = create_dynamics_function(equations)
    % Convert string equations to MATLAB function for any nonlinear system
    % equations: cell array of strings like {"1.5*x1 - x1*x2", "-3*x2 + x1*x2"}
    % Returns a function handle that can be used with CORA's nonlinearSys

    dynamics_func = @(x, u) evaluate_equations(equations, x, u);
end

function dx = evaluate_equations(equations, x, u)
    % Evaluate the system of equations from configuration
    % equations: cell array of strings representing the dynamics
    % x: state vector
    % u: input vector (can be empty for autonomous systems)

    n_states = length(equations);

    % Handle both numeric and symbolic inputs
    if isa(x, 'sym')
        dx = sym(zeros(n_states, 1));
    else
        dx = zeros(n_states, 1);
    end

    % Evaluate each equation
    for i = 1:n_states
        equation_str = equations{i};

        % Replace variable names with actual values
        % Support variables: x1, x2, ..., xn and u1, u2, ..., um
        eval_str = equation_str;

        % Replace state variables (x1, x2, etc.) with element-wise operations
        % Use more robust pattern matching to avoid partial replacements
        for j = n_states:-1:1 % Replace in reverse order to avoid issues with x10 vs x1
            pattern = sprintf('x%d', j);
            replacement = sprintf('x(%d)', j);
            eval_str = regexprep(eval_str, ['\<' pattern '\>'], replacement);
        end

        % Replace input variables (u1, u2, etc.) if inputs exist
        if ~isempty(u)
            n_inputs = length(u);

            for j = n_inputs:-1:1
                pattern = sprintf('u%d', j);
                replacement = sprintf('u(%d)', j);
                eval_str = regexprep(eval_str, ['\<' pattern '\>'], replacement);
            end

        end

        % Ensure all mathematical operations are element-wise
        % This prevents MATLAB from trying to do matrix operations
        eval_str = strrep(eval_str, '*', '.*');
        eval_str = strrep(eval_str, '/', './');
        eval_str = strrep(eval_str, '^', '.^');

        % Fix cases where we accidentally made element-wise operations on scalars
        eval_str = strrep(eval_str, '..', '.');

        % Handle the case where we have .*( or ./( which should be *(  or /(
        eval_str = regexprep(eval_str, '\.\*\(', '*(');
        eval_str = regexprep(eval_str, '\./\(', '/(');

        % Fix leading operators
        eval_str = regexprep(eval_str, '^-\.', '-');
        eval_str = regexprep(eval_str, '^\+\.', '+');

        try
            % Evaluate the equation
            dx(i) = eval(eval_str);
        catch ME
            % If evaluation fails, provide detailed error information
            fprintf('Error evaluating equation %d:\n', i);
            fprintf('Original: %s\n', equation_str);
            fprintf('Processed: %s\n', eval_str);
            fprintf('Error: %s\n', ME.message);
            rethrow(ME);
        end

    end

end
