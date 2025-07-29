function result = check_reachability(reachable_set, target_set)
    % CHECK_REACHABILITY - Checks intersection between overapproximated reachable set and target set
    %
    % Inputs:
    %   reachable_set - CORA reachSet object containing time intervals
    %   target_set    - CORA set object (zonotope, interval, etc.)
    %
    % Outputs:
    %   result - string: 'unreachable' if no intersection, 'unknown' if intersection exists
    %           (since reachable sets are overapproximations)

    result = 'unreachable'; % Default: assume unreachable until intersection found

    if isempty(reachable_set)
        return;
    end

    try
        % Check intersection with time interval sets
        if isfield(reachable_set, 'timeInterval') && isfield(reachable_set.timeInterval, 'set')
            fprintf('Checking %d time interval sets for intersection...\n', length(reachable_set.timeInterval.set));

            for i = 1:length(reachable_set.timeInterval.set)
                reachable_i = reachable_set.timeInterval.set{i};

                try
                    % Simple intersection using & operator
                    intersection = reachable_i & target_set;

                    if ~representsa(intersection, 'emptySet')
                        fprintf('Intersection found at time step %d - Result: UNKNOWN (overapproximation)\n', i);
                        result = 'unknown';
                        return;
                    end

                catch intersectionError
                    fprintf('Intersection error at step %d: %s\n', i, intersectionError.message);
                end

            end

            fprintf('No intersection found in time interval sets - Result: UNREACHABLE\n');
        end

        % Alternative: check with time point sets
        if isfield(reachable_set, 'timePoint') && isfield(reachable_set.timePoint, 'set')
            fprintf('Checking %d time point sets for intersection...\n', length(reachable_set.timePoint.set));

            for i = 1:length(reachable_set.timePoint.set)
                reachable_i = reachable_set.timePoint.set{i};

                try
                    intersection = reachable_i & target_set;

                    if ~representsa(intersection, 'emptySet')
                        fprintf('Intersection found at time point %d - Result: UNKNOWN (overapproximation)\n', i);
                        result = 'unknown';
                        return;
                    end

                catch intersectionError
                    fprintf('Intersection error at time point %d: %s\n', i, intersectionError.message);
                end

            end

            fprintf('No intersection found in time point sets - Result: UNREACHABLE\n');
        end

    catch ME
        % Handle potential intersection errors
        warning('CORA:reachability', 'Error during reachability check: %s', ME.message);
        result = 'error';
    end

end
