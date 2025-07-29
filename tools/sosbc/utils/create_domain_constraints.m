function constraints = create_domain_constraints(domain, vars)
    % CREATE_DOMAIN_CONSTRAINTS - Create domain constraints from checking.domain in config
    %
    % Inputs:
    %   domain - domain bounds [[x1_min, x1_max], [x2_min, x2_max], ...]
    %   vars - symbolic variables
    %
    % Output:
    %   constraints - cell array of polynomial constraints g_i(x) >= 0

    constraints = create_box_constraints(domain, vars);
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
