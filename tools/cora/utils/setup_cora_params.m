function [params, options] = setup_cora_params(config, setting)
    % Sets up CORA reachability parameters and options from the config and current setting.

    % Extract final time from time_horizon array [t_start, t_final]
    if isfield(config.verification, 'time_horizon')

        if length(config.verification.time_horizon) == 2
            params.tFinal = config.verification.time_horizon(2); % Use final time
        else
            params.tFinal = config.verification.time_horizon; % Single value
        end

    else
        error('No time_horizon specified in config.verification');
    end

    % For autonomous systems, set U as zero zonotope
    params.U = zonotope(0); % Single zero input for autonomous systems

    % Algorithm options go in options struct
    options.timeStep = setting.timeStep;
    options.taylorTerms = setting.taylorTerms;
    options.zonotopeOrder = setting.zonotopeOrder;
    options.alg = setting.alg;
    options.tensorOrder = 2; % Default tensor order

    % Additional optional parameters
    if isfield(setting, 'reductionTechnique')
        options.reductionTechnique = setting.reductionTechnique;
    end

end
