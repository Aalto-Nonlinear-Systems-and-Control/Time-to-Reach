import numpy as np
from scipy.integrate import odeint


def simulate(dyn, pts, T, dt, round_up=True, forward=True):
    if not isinstance(pts, np.ndarray):
        pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    assert pts.ndim == 2  # N*n matrix, N n-dimensional points
    assert T > 0 and T >= dt
    T = np.ceil(T / dt) * dt if round_up else np.floor(T / dt) * dt
    steps = np.ceil(T / dt).astype(int)

    coeff = 1 if forward else -1

    def ode_fx(x, t):
        x = x.reshape(pts.shape)
        y = dyn.ff(*x.T)
        # Robustly handle output shape
        if isinstance(y, list):
            y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        elif y.ndim == 2 and y.shape[0] != x.shape[0]:
            y = y.T
        return y.flatten() * coeff

    traj = []

    x_cur = pts

    for i in range(steps):
        x_next = odeint(ode_fx, x_cur.flatten(), np.linspace(0, dt, 5))[-1].reshape(
            pts.shape
        )
        traj.append(x_next)
        x_cur = x_next

    traj = np.stack(traj)

    return traj


def simulate_trajectories(system, initial_points, T, dt=0.01, forward=True):
    """
    Simulate trajectories for a given system from multiple initial points.

    Args:
        system: Object with a .ff(*x) method returning the vector field (ODE RHS).
        initial_points: (N, dim) array of initial conditions.
        T: Final time (float).
        dt: Time step (float).
        forward: If True, integrate forward in time; else backward.

    Returns:
        trajectories: (N, num_steps+1, dim) array of simulated points.
        t: (num_steps+1,) array of time points.
    """
    initial_points = np.asarray(initial_points, dtype=float)
    if initial_points.ndim == 1:
        initial_points = initial_points[None, :]
    N, dim = initial_points.shape
    t = np.arange(0, T + dt, dt)
    if not forward:
        t = t[::-1]
    trajectories = np.zeros((N, len(t), dim))
    for i, x0 in enumerate(initial_points):

        def ode_rhs(x, _t):
            x = np.asarray(x)
            return np.ravel(system.ff(*x))

        traj = odeint(ode_rhs, x0, t)
        trajectories[i] = traj
    return trajectories, t
