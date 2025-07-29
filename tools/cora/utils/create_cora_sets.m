function [X0, target] = create_cora_sets(config)
    % Creates CORA zonotope objects for initial and target sets.
    % Supports box-type sets defined by bounds, converted to zonotopes

    % Parse initial set (take the first one if multiple)
    if iscell(config.initial_sets)
        initial_set = config.initial_sets{1};
    else
        initial_set = config.initial_sets(1);
    end

    if strcmp(initial_set.type, 'box')
        bounds = initial_set.bounds;
        init_lower = bounds(:, 1); % First column is lower bounds
        init_upper = bounds(:, 2); % Second column is upper bounds

        % Convert box to zonotope: center + generators for each dimension
        center = (init_upper + init_lower) / 2;
        generators = diag((init_upper - init_lower) / 2);
        X0 = zonotope(center, generators);
    else
        error('Unsupported initial set type: %s', initial_set.type);
    end

    % Parse target set (take the first one if multiple)
    if iscell(config.verification.target_sets)
        target_set = config.verification.target_sets{1};
    else
        target_set = config.verification.target_sets(1);
    end

    if strcmp(target_set.type, 'box')
        bounds = target_set.bounds;
        target_lower = bounds(:, 1); % First column is lower bounds
        target_upper = bounds(:, 2); % Second column is upper bounds

        % Convert box to zonotope: center + generators for each dimension
        center = (target_upper + target_lower) / 2;
        generators = diag((target_upper - target_lower) / 2);
        target = zonotope(center, generators);
    else
        error('Unsupported target set type: %s', target_set.type);
    end

end
