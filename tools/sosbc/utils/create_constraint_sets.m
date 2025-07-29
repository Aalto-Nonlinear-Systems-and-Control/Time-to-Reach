function constraints = create_constraint_sets(set_definitions, vars)
    % CREATE_CONSTRAINT_SETS - Create polynomial constraints from set definitions
    %
    % Inputs:
    %   set_definitions - array of set definitions from config
    %   vars - symbolic variables
    %
    % Output:
    %   constraints - cell array of polynomial constraints g_i(x) >= 0

    constraints = {};

    for k = 1:length(set_definitions)
        set_def = set_definitions(k);

        switch set_def.type
            case 'box'
                % Convert box to polynomial constraints
                box_constraints = create_box_constraints(set_def.bounds, vars);
                constraints = [constraints, box_constraints];

            case 'level_set'
                % Direct polynomial constraint
                poly_constraint = create_polynomial_constraint(set_def, vars);
                constraints{end + 1} = poly_constraint;

            otherwise
                error('Unsupported set type: %s', set_def.type);
        end

    end

end

function constraints = create_box_constraints(bounds, vars)
    % Convert n-dimensional box to polynomial constraints
    % bounds: numeric array [[a1,b1]; [a2,b2]; ...; [an,bn]] or cell array {[a1,b1], [a2,b2], ..., [an,bn]}
    % Returns: cell array of polynomial constraints g_i(x) >= 0

    if iscell(bounds)
        % Handle cell array format
        n = length(bounds);
        constraints = {};

        for i = 1:n
            % For each dimension: ai <= xi <= bi
            % Convert to: xi - ai >= 0 and bi - xi >= 0
            constraints{end + 1} = vars(i) - bounds{i}(1); % xi >= ai
            constraints{end + 1} = bounds{i}(2) - vars(i); % xi <= bi
        end

    else
        % Handle numeric array format (from JSON)
        n = size(bounds, 1);
        constraints = {};

        for i = 1:n
            % For each dimension: ai <= xi <= bi
            % Convert to: xi - ai >= 0 and bi - xi >= 0
            constraints{end + 1} = vars(i) - bounds(i, 1); % xi >= ai
            constraints{end + 1} = bounds(i, 2) - vars(i); % xi <= bi
        end

    end

end

function constraint = create_polynomial_constraint(poly_def, vars)
    % Create polynomial constraint from string definition
    % Config defines sets as f(x) <= 0, but SOS needs g(x) >= 0
    % So we convert by negating: g(x) = -f(x)

    poly_str = poly_def.function;

    % Check if the function contains non-polynomial elements like exp()
    if contains(poly_str, 'exp')
        warning('SOSBC:NonPolynomial', 'The function contains exp() which is not polynomial. SOS methods work best with polynomial constraints.');

        % For now, we'll try to approximate or handle this case
        % You may need to manually approximate the exponential with a polynomial
        error('SOSBC:UnsupportedFunction', ...
        'Exponential functions are not directly supported in SOS formulations. Please provide a polynomial approximation.');
    end

    % Replace x1, x2, ... with actual polynomial variable references
    n = length(vars);

    for i = 1:n
        poly_str = strrep(poly_str, sprintf('x%d', i), sprintf('vars(%d)', i));
    end

    % Evaluate the polynomial expression
    f_constraint = eval(poly_str);

    % Convert f(x) <= 0 to g(x) >= 0 by negating
    constraint = -f_constraint;
end
