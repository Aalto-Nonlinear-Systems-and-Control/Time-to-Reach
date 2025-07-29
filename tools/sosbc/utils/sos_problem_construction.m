function [prog, B, s_init, s_target, s_domain] = sos_problem_construction(prog, system, init_constraints, target_constraints, domain_constraints, options)
    % BUILD_SOS_PROBLEM - Main function to build the complete SOS problem
    %
    % Inputs:
    %   prog - SOS program
    %   system - system structure (vars, dynamics)
    %   init_constraints, target_constraints, domain_constraints - constraint sets
    %   options - solver options (deg_B, deg_s, epsilon)
    %
    % Outputs:
    %   prog - updated SOS program with all constraints
    %   B - barrier certificate variable
    %   s_init, s_target, s_domain - SOS multipliers

    % Create barrier certificate variable
    [prog, B] = sospolyvar(prog, monomials(system.vars, 0:options.deg_B));

    % Create SOS multipliers
    [prog, s_init] = create_sos_multipliers(prog, system.vars, length(init_constraints), options.deg_s);
    [prog, s_target] = create_sos_multipliers(prog, system.vars, length(target_constraints), options.deg_s);
    [prog, s_domain] = create_sos_multipliers(prog, system.vars, length(domain_constraints), options.deg_s);

    % Compute Lie derivative
    LfB = compute_lie_derivative(B, system.vars, system.dynamics);

    % Add SOS constraints
    prog = add_barrier_constraints(prog, B, init_constraints, target_constraints, domain_constraints, ...
        LfB, s_init, s_target, s_domain, options.epsilon);

end

function [prog, multipliers] = create_sos_multipliers(prog, vars, num_constraints, deg_s)
    % CREATE_SOS_MULTIPLIERS - Create SOS multipliers for constraints
    %
    % Inputs:
    %   prog - SOS program
    %   vars - symbolic variables
    %   num_constraints - number of constraints
    %   deg_s - degree of SOS multipliers
    %
    % Output:
    %   prog - updated SOS program
    %   multipliers - cell array of SOS multipliers

    multipliers = cell(num_constraints, 1);

    for i = 1:num_constraints
        [prog, s] = sospolyvar(prog, monomials(vars, 0:deg_s));
        multipliers{i} = s;
    end

end

function LfB = compute_lie_derivative(B, vars, dynamics)
    % COMPUTE_LIE_DERIVATIVE - Compute Lie derivative for n-dimensional system
    % LfB = ∇B · f(x) = Σ(∂B/∂xi * fi(x))

    n = length(vars);
    LfB = 0;

    for i = 1:n
        dB_dxi = diff(B, vars(i));
        LfB = LfB + dB_dxi * dynamics{i};
    end

end

function prog = add_barrier_constraints(prog, B, init_constraints, target_constraints, domain_constraints, ...
        LfB, s_init, s_target, s_domain, epsilon)
    % ADD_BARRIER_CONSTRAINTS - Add SOS constraints for barrier certificate
    %
    % Inputs:
    %   prog - SOS program
    %   B - barrier certificate variable
    %   init_constraints, target_constraints, domain_constraints - constraint sets
    %   LfB - Lie derivative of B
    %   s_init, s_target, s_domain - SOS multipliers
    %   epsilon - small positive constant

    % Constraint 1: B(x) > 0 on target/unsafe set
    % B(x) - ε - Σ(s_j * g_target_j) ∈ SOS
    target_sum = sum_constraints(target_constraints, s_target);
    prog = sosineq(prog, B - epsilon - target_sum);

    % Constraint 1: B(x) <= 0 on initial set
    % -B(x) - Σ(s_i * g_init_i) ∈ SOS
    init_sum = sum_constraints(init_constraints, s_init);
    prog = sosineq(prog, -B -epsilon - init_sum);

    % Constraint 3: dB/dt <= 0 on domain
    % -LfB - Σ(s_k * g_domain_k) ∈ SOS
    domain_sum = sum_constraints(domain_constraints, s_domain);
    prog = sosineq(prog, -LfB - domain_sum);

    % Constraint 4: All multipliers must be SOS
    all_multipliers = [s_init; s_target; s_domain];

    for i = 1:length(all_multipliers)
        prog = sosineq(prog, all_multipliers{i});
    end

end

function constraint_sum = sum_constraints(constraints, multipliers)
    % SUM_CONSTRAINTS - Sum over all constraints: Σ(si * gi)

    constraint_sum = 0;

    for i = 1:length(constraints)
        constraint_sum = constraint_sum + multipliers{i} * constraints{i};
    end

end
