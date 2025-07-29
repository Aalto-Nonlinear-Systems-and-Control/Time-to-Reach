import cvxpy as cp
import numpy as np
import portion as P
from scipy.integrate import solve_ivp
from sympy import Matrix, lambdify

from .sampling import point_in_set, sample_sets
from .simulation import simulate_trajectories


def _get_fn(dyn, x):
    return dyn.f - _get_jacob(dyn, x) @ (Matrix(dyn.x) - Matrix(x))


def _get_jacob(dyn, x):
    assert dyn.dimension == len(x)
    jacob = Matrix(dyn.f.jacobian(dyn.x))
    jacob_f = lambdify(dyn.x, jacob, "numpy")

    return jacob_f(*x)


def init_principle_eigenfunction_eval(dyn, xe, skip_check=False):
    A = np.asarray(_get_jacob(dyn, xe), dtype=float)
    lam, w = np.linalg.eig(
        A.T
    )  # v is the left eigen vector of A, do not use scipy.linalg.eig to compute this!
    # to ensure the satisfaction of Theorem 2 & Corollary 1
    eig = np.real(lam)

    if not skip_check:
        if eig.max() < 0:
            assert 2 * eig.max() < eig.min()
        elif eig.min() > 0:
            assert eig.min() < 2 * eig.max()
        else:
            print(f"Eigenvalues: {eig}")
            raise ValueError("Eigenvalues are not all positive or negative")

    g = w.T @ _get_fn(dyn, xe)

    return (lam, w), g


def _path_integral_eig(dyn, xe, T, pts, lam, w, g, num_t_eval=150):
    def H(t, xh):
        xh = xh.reshape((pts.shape[0], -1))
        x = xh[:, :-1].T

        dx = dyn.ff(*x)
        # Handle different possible shapes of dx
        if dx.ndim == 3 and dx.shape[2] == 1:
            dx = dx.squeeze(axis=2)
        elif dx.ndim == 2 and dx.shape[1] == 1:
            dx = dx.squeeze(axis=1)
        elif dx.ndim == 1:
            dx = dx.reshape(-1, 1)

        # Ensure dx has shape (dim, num_points)
        if dx.ndim == 2 and dx.shape[0] != dyn.dimension:
            dx = dx.T

        gfx = gf(*x)
        if not isinstance(gfx, np.ndarray):
            gfx = np.ones((dx.shape[1], 1)) * gfx
        elif gfx.ndim == 1:
            gfx = gfx.reshape(1, -1)
        elif gfx.ndim == 2 and gfx.shape[0] != 1:
            gfx = gfx.reshape(1, -1)

        dh = np.exp(-lam * t) * gfx

        # Ensure dx.T is 2D (num_points, dim)
        dx_t = dx.T
        if dx_t.ndim == 3:
            dx_t = np.squeeze(dx_t, axis=2)
        if dx_t.ndim == 1:
            dx_t = dx_t.reshape(-1, 1)
        # Ensure dh.T is 2D (num_points, 1)
        dh_t = dh.T
        if dh_t.ndim == 3:
            dh_t = np.squeeze(dh_t, axis=2)
        if dh_t.ndim == 1:
            dh_t = dh_t.reshape(-1, 1)
        return np.concatenate([dx_t, dh_t], axis=1).flatten()

    gf = lambdify(dyn.x, g, "numpy")

    h0 = np.zeros((pts.shape[0], 1), dtype=np.complex64)

    xh0 = np.concatenate([pts, h0], axis=1, dtype=np.complex64).flatten()

    t_evals = np.linspace(0, T, num_t_eval)

    sol = solve_ivp(H, [0, T], xh0, t_eval=t_evals, method="RK45")

    hx = sol.y.reshape(pts.shape[0], pts.shape[1] + 1, -1)[:, -1, :]

    lin = (pts - xe) @ w

    return lin[:, None] + hx


def path_integral_eigeval(dyn, xe, T, pts, lam, w, g):
    ef_vals = [
        _path_integral_eig(dyn, xe, T, pts, lam[i], w[:, i], g[i])
        for i in range(dyn.dim)
    ]

    return np.vstack([ef_vals])


def linear_fractional_programming(A, b, c, d, lx, ux, e=0, f=0, minimize=True):
    """Linear fractional programming solver."""
    y = cp.Variable(c.shape[0])
    t = cp.Variable(pos=True)

    numerator = c.T @ y + e * t
    denominator = d.T @ y + f * t

    if np.count_nonzero(d >= 0) == 0 and f <= 0:
        numerator *= -1
        denominator *= -1

    # set constraints
    constraints = [denominator == 1, t >= 0]

    if A is not None and b is not None:
        constraints.append(A @ y <= b * t)
    if lx is not None:
        constraints.append(lx * t <= y)
    if ux is not None:
        constraints.append(ux * t >= y)

    objective = cp.Minimize(numerator) if minimize else cp.Maximize(numerator)

    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.MOSEK, verbose=False)

    return problem.status, problem.value, y.value, t.value


def reach_time_bounds_with_magnitude_sep(ef0_vals, efF_vals, lams):
    """Compute reach time bounds using magnitude separation."""
    mag0, magF = np.log(np.abs(ef0_vals)), np.log(np.abs(efF_vals))

    mag0_inf, mag0_sup = np.min(mag0, axis=0), np.max(mag0, axis=0)
    magF_inf, magF_sup = np.min(magF, axis=0), np.max(magF, axis=0)

    lams_real = np.real(lams)

    # determine the time bounds in the case of positive eigenfunctions
    ## If lambda > 0, unstable
    ### lower bound
    A = -lams_real
    b = np.zeros(1)

    c = magF_inf - mag0_sup
    d = lams_real

    _, bound_lower_pos_unstable, _, _ = linear_fractional_programming(
        A, b, c, d, 0, 1, 0, 0, minimize=False
    )

    ### upper bound
    c = magF_sup - mag0_inf

    _, bound_upper_pos_unstable, _, _ = linear_fractional_programming(
        A, b, c, d, 0, 1, 0, 0, minimize=True
    )

    bound_pos_unstable = P.closed(bound_lower_pos_unstable, bound_upper_pos_unstable)

    ## If lambda < 0, stable
    ### lower bound
    A = lams_real
    c = magF_sup - mag0_inf

    _, bound_lower_pos_stable, _, _ = linear_fractional_programming(
        A, b, c, d, 0, 1, 0, 0, minimize=False
    )

    ### upper bound
    c = magF_inf - mag0_sup

    _, bound_upper_pos_stable, _, _ = linear_fractional_programming(
        A, b, c, d, 0, 1, 0, 0, minimize=True
    )

    bound_pos_stable = P.closed(bound_lower_pos_stable, bound_upper_pos_stable)

    print(bound_pos_stable, bound_pos_unstable, ">>>>>>>")

    return bound_pos_unstable & bound_pos_stable


def reach_time_bounds_with_imaginary(ef0_vals, efF_vals, lams):
    """Compute reach time bounds using imaginary part."""

    def __get_bound(s_, y_, t_, lams_, diff_min, diff_max):
        bound_lower_, bound_upper_, img_opt = -np.inf, np.inf, None
        if s_ == "optimal":
            alpha = y_ / t_
            img_opt = np.dot(alpha, lams_)

            if img_opt != 0:
                bound_lower_ = np.dot(alpha, diff_min) / img_opt
                bound_upper_ = np.dot(alpha, diff_max) / img_opt

                bound_lower_, bound_upper_ = np.minimum(
                    bound_lower_, bound_upper_
                ), np.maximum(bound_lower_, bound_upper_)

        return P.closed(bound_lower_, bound_upper_), img_opt

    ang0, angF = np.angle(ef0_vals), np.angle(efF_vals)

    # correct the angle
    def correct_angle(a):
        a_new = np.zeros_like(a)
        diff = np.abs(np.max(a, axis=0) - np.min(a, axis=0))

        for i in range(diff.shape[0]):
            if diff[i] > 0.5 * np.pi:
                a_new[:, i] = a[:, i]
                a_new[a_new[:, i] < 0, i] += np.pi * 2
            else:
                a_new[:, i] = a[:, i]
        return a_new

    ang0 = correct_angle(ang0)
    angF = correct_angle(angF)

    ang0_inf, ang0_sup = np.min(ang0, axis=0), np.max(ang0, axis=0)
    angF_inf, angF_sup = np.min(angF, axis=0), np.max(angF, axis=0)

    ang_diff_min = angF_inf - ang0_sup
    ang_diff_max = angF_sup - ang0_inf

    lams_img = np.imag(lams)

    # determine the time bounds in the case of λ > 0
    A = -lams_img
    b = np.zeros(1)
    c = angF_sup - ang0_inf - angF_inf + ang0_sup
    d = lams_img

    status, _, y, t = linear_fractional_programming(
        A, b, c, d, 0, 1, 0, 0, minimize=True
    )

    bound_pos, img_pos = __get_bound(status, y, t, lams_img, ang_diff_min, ang_diff_max)

    # determine the time bounds in the case of λ < 0
    A = lams_img
    c = ang0_sup - angF_inf - ang0_inf + angF_sup
    d = -lams_img

    status, _, y, t = linear_fractional_programming(
        A, b, c, d, 0, 1, 0, 0, minimize=True
    )

    bound_neg, img_neg = __get_bound(status, y, t, lams_img, ang_diff_max, ang_diff_min)

    # return bound_pos & bound_neg, img_pos, img_neg
    assert np.allclose(img_pos, abs(img_neg))
    return bound_pos, img_pos, img_neg


def compute_reach_time_bounds(ef0_vals, efF_vals, lams, itol=1e-3):
    """Compute reach-time bounds using KRTB method."""

    bound_mag = reach_time_bounds_with_magnitude_sep(ef0_vals, efF_vals, lams)

    # if all eigenvalues are real, then we skip the imaginary part
    if np.all(np.isreal(lams)):
        if (
            bound_mag.empty
            or bound_mag.upper < 0
            or np.all(
                np.isinf(
                    np.array(
                        [
                            bound_mag.lower,
                            bound_mag.upper,
                        ]
                    )
                )
            )
        ):
            return [], ("UNREACHABLE", 0)
        else:

            return [(bound_mag.lower, bound_mag.upper)], ("PROBABLY REACHABLE", 1)

    bound_img, img_pos, img_neg = reach_time_bounds_with_imaginary(
        ef0_vals, efF_vals, lams
    )

    if (
        bound_mag.empty
        or bound_mag.upper < 0
        or np.all(
            np.isinf(
                np.array(
                    [bound_mag.lower, bound_mag.upper, bound_img.lower, bound_img.upper]
                )
            )
        )
    ):
        return [], ("UNREACHABLE", 0)

    if img_pos is None:
        assert img_neg is None
        return [(bound_mag.lower, bound_mag.upper)], ("PROBABLY REACHABLE", 1)

    period = (2 * np.pi) / abs(img_pos) if img_pos != 0 else np.inf

    if np.isposinf(float(bound_mag.upper)):
        if bound_img.lower < 0:
            min_shift = np.ceil(-float(bound_img.lower) / period) * period
            return [(bound_img.lower + min_shift, bound_img.upper + min_shift)], (
                "PROBABLY REACHABLE",
                period,
            )

    bounds = []

    if bound_mag.lower < 0:
        bound_mag = bound_mag.replace(lower=0)

    min_shift = (
        np.ceil((float(bound_mag.lower) - float(bound_img.upper)) / period) * period
    )
    assert min_shift >= 0

    bound_wind = P.closed(bound_img.lower + min_shift, bound_img.upper + min_shift)

    i = 0
    while True:
        this_bound = P.closed(
            bound_wind.lower + i * period, bound_wind.upper + i * period
        )

        this_intersection = this_bound & bound_mag

        if this_intersection.empty:
            break

        if len(bounds) >= 1:
            if abs(float(bounds[-1][1]) - float(this_intersection.lower)) <= itol:
                bounds[-1] = (bounds[-1][0], this_intersection.upper)
                i = i + 1
                continue

        bounds.append((this_intersection.lower, this_intersection.upper))
        i = i + 1

    if len(bounds) == 0:
        return [], ("UNREACHABLE", 0)

    return bounds, ("PROBABLY REACHABLE", len(bounds))


def verify_reachability_with_simulation(
    system,
    initial_sets,
    target_sets,
    reach_time_bounds,
    num_sim_samples=100,
    dt=0.01,
):
    """
    Verify reachability by running simulations based on computed time bounds.

    Args:
        system: The system object.
        initial_sets: A list of initial set definitions.
        target_sets: A list of target set definitions.
        reach_time_bounds: A list of (lower, upper) time bound tuples.
        num_sim_samples (int): Number of points to sample for simulation.
        dt (float): Time step for simulation.

    Returns:
        A string indicating the verification result: "reachable", "unreachable", or "inconclusive".
    """
    # Case 1: No reach-time bounds were found by KRTB.
    if not reach_time_bounds:
        return "unreachable"

    print("\n--- Starting Simulation-Based Verification ---")
    print(f"Sampling {num_sim_samples} points from initial set for simulation.")

    # Case 2: Bounds found, run simulations to confirm.
    # Sample points from the initial set(s).
    sim_initial_points = sample_sets(initial_sets, num_sim_samples)
    if sim_initial_points.size == 0:
        print(
            "Warning: Could not sample any points from the initial set for verification."
        )
        return "inconclusive"

    # Determine max simulation time from the bounds.
    T_max = max(b[1] for b in reach_time_bounds if b[1] != np.inf and b[1] is not None)
    if T_max is None or T_max == -np.inf:
        print("Warning: No finite upper time bound found for simulation.")
        return "inconclusive"

    print(f"Simulating trajectories up to T_max = {T_max:.4f}...")

    # Run simulations.
    trajectories, t_vec = simulate_trajectories(
        system, sim_initial_points, T=T_max, dt=dt
    )

    print("Checking if any trajectory hits the target set...")
    # Check if any point in any trajectory enters any target set.
    for i, traj in enumerate(trajectories):
        for point in traj:
            for target_set in target_sets:
                if point_in_set(point, target_set):
                    print(
                        f"✓ Reachable: Trajectory {i} hit the target set at point {point}."
                    )
                    return "reachable"

    # If no trajectory hit the target set after all simulations.
    print("✗ Inconclusive: No simulated trajectory intersected the target set.")
    return "inconclusive"
