function result = sosbc_pipeline(config_file, options)
    % SOSBC_PIPELINE - Complete pipeline for Sum of Squares Barrier Certificate
    %
    % Inputs:
    %   config_file - path to JSON configuration file
    %   options - structure with solver options (deg_B, deg_s, epsilon, solver)
    %
    % Output:
    %   result - structure with success flag, barrier certificate, and metadata

    fprintf('=== SOSBC Pipeline Started ===\n');
    tic;

    %% Step 1: Load Problem from Config File
    fprintf('Step 1: Loading problem configuration...\n');

    % Load configuration
    config_text = fileread(config_file);
    config = jsondecode(config_text);
    fprintf('✓ Loaded config: %s\n', config.benchmark.name);

    % Create system from config
    system = create_system_from_config(config);
    fprintf('✓ System created: %s, dimension %d\n', system.type, length(system.vars));

    % Create constraint sets
    init_constraints = create_constraint_sets(config.initial_sets, system.vars);
    target_constraints = create_constraint_sets(config.verification.target_sets, system.vars);
    domain_constraints = create_domain_constraints(config.checking.domain, system.vars);

    fprintf('✓ Constraint sets created\n');

    %% Step 2: SOS Problem Formulation
    fprintf('Step 2: Formulating SOS problem...\n');

    % Initialize SOS program
    prog = sosprogram(system.vars);

    % Build complete SOS problem
    [prog, B, s_init, s_target, s_domain] = sos_problem_construction(prog, system, init_constraints, target_constraints, domain_constraints, options);

    fprintf('✓ SOS constraints added\n');

    %% Step 3: Solve with MOSEK
    fprintf('Step 3: Solving SOS program...\n');

    % Set solver options
    % prog = sossetobj(prog, 0); % Feasibility problem

    % Solve
    solver_opt.solver = 'mosek';
    prog = sossolve(prog, solver_opt);

    % Check solution
    if prog.solinfo.info.pinf == 0 && prog.solinfo.info.dinf == 0
        success = true;
        barrier = sosgetsol(prog, B);
        fprintf('✓ Solution found successfully!\n');
    else
        success = false;
        barrier = [];
        fprintf('✗ No solution found\n');
    end

    computation_time = toc;

    %% Step 4: Package Results
    result.success = success;
    result.computation_time = computation_time;
    result.barrier = barrier;
    result.system = system;
    result.config = config;
    result.solver_info = prog.solinfo;

    if success
        result.barrier_coeffs = barrier.coefficient;
        result.barrier_degmat = barrier.degmat;
    end

    fprintf('=== Pipeline completed in %.2f seconds ===\n', computation_time);
end
