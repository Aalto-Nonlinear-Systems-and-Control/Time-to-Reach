"""Principal Koopman eigenfunctions: analytic expressions or the path-integral formula.

Path integral (for an asymptotically stable equilibrium xe with Jacobian A):
    psi_i(x) = w_i^T (x - xe) + int_0^inf exp(-lam_i t) w_i^T r(x(t)) dt,
where A^T w_i = lam_i w_i and r(x) = f(x) - A (x - xe) is the nonlinear residual.
The integral converges when 2 max Re(lam) < min Re(lam); it is truncated at T.
"""

from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import cumulative_simpson, solve_ivp
from scipy.ndimage import maximum_filter1d

from tools.common import expr


@dataclass
class Eigenfunctions:
    eigenvalues: np.ndarray  # complex, shape (m,), eigenvalues of f
    evaluate: object  # X (N, n) -> complex (N, m)
    info: dict = field(default_factory=dict)


def _complex_vectorized(exprs, n):
    symbols = expr.state_symbols(n)
    f = sp.lambdify(symbols, list(exprs), modules="numpy")

    def F(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty((X.shape[0], len(exprs)), dtype=complex)
        for j, col in enumerate(f(*X.T)):
            out[:, j] = col
        return out

    return F


def analytic(functions, eigenvalues, system, check_points=None):
    """Eigenfunctions given in closed form, e.g. 'x1^2 + 2*x2 + x2^3'.

    If check_points are given, the eigen-relation grad(psi) . f = lam psi is
    checked there and the largest relative residual is reported.
    """
    n = system.dim
    exprs = [expr.parse(f, n) for f in functions]
    lams = np.array([complex(v) for v in eigenvalues])
    if len(exprs) != len(lams):
        raise ValueError("need one eigenvalue per eigenfunction")
    info = {"method": "analytic", "functions": list(functions)}
    if check_points is not None:
        xs = expr.state_symbols(n)
        lie = [sum(sp.diff(e, x) * fx for x, fx in zip(xs, system.exprs)) for e in exprs]
        L = _complex_vectorized(lie, n)(check_points)
        P = _complex_vectorized(exprs, n)(check_points)
        resid = np.abs(L - P * lams) / np.maximum(np.abs(P * lams), 1e-12)
        info["max_relative_eigen_residual"] = float(np.max(resid))
    return Eigenfunctions(lams, _complex_vectorized(exprs, n), info)


def linearize(system, equilibrium):
    xe = np.asarray(equilibrium, dtype=float)
    xs = expr.state_symbols(system.dim)
    J = sp.Matrix(system.exprs).jacobian(xs)
    A = np.array(sp.lambdify(xs, J, modules="numpy")(*xe), dtype=float)
    fxe = system.rhs(xe[None, :])[0]
    if np.max(np.abs(fxe)) > 1e-9:
        raise ValueError(f"{equilibrium} is not an equilibrium: |f(xe)| = {np.max(np.abs(fxe)):.3g}")
    return xe, A


def path_integral(system, equilibrium, T, rtol=1e-12, atol=1e-14, batch=64, dt=0.02, envelope_window=5.0):
    """Principal eigenfunctions of a stable equilibrium via the path-integral formula.

    x(t) is integrated once per batch of points (dense output); the integral is
    evaluated by composite Simpson quadrature on a grid of spacing about dt. The
    integrand exp(-lam t) w^T r(x(t)) decays for the exact solution, but r(x) is
    a difference of nearly equal terms near xe, so its round-off is amplified by
    exp(-Re(lam) t) and eventually grows again. The integral is therefore
    truncated, per point and eigenfunction, where the envelope of |integrand|
    (its maximum over a sliding window of `envelope_window` time units, which
    removes isolated zeros of oscillating integrands) is smallest; for a cleanly
    decaying integrand this is T itself.
    """
    xe, A = linearize(system, equilibrium)
    lams, W = np.linalg.eig(A.T)  # columns: left eigenvectors of A
    re = np.real(lams)
    if not np.all(re < 0):
        raise NotImplementedError(f"path integral needs a stable equilibrium; Re(lam) = {re}")
    margin = float(re.min() - 2 * re.max())
    if margin <= 0:
        raise ValueError(f"2 max Re(lam) < min Re(lam) fails (Re(lam) = {re}); the integral does not converge")
    n, m = system.dim, len(lams)
    ts = np.linspace(0.0, T, max(2001, int(np.ceil(T / dt)) + 1))
    decay = np.exp(-np.outer(ts, lams))  # (K, m)
    window = max(3, int(round(envelope_window / (ts[1] - ts[0]))) | 1)
    info = {
        "method": "path_integral",
        "equilibrium": xe.tolist(),
        "T": T,
        "rtol": rtol,
        "atol": atol,
        "quadrature_points": len(ts),
        "spectral_margin": margin,
        # diagnostics over all evaluated points: earliest truncation time, and the
        # largest integrand envelope at the truncation time relative to |psi|
        "min_truncation_time": float(T),
        "max_relative_integrand_at_truncation": 0.0,
    }

    def evaluate(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty((X.shape[0], m), dtype=complex)
        for start in range(0, X.shape[0], batch):
            pts = X[start:start + batch]
            N = pts.shape[0]
            sol = solve_ivp(lambda t, y: system.rhs(y.reshape(N, n)).ravel(), (0.0, T), pts.ravel(),
                            method="DOP853", rtol=rtol, atol=atol, dense_output=True)
            if not sol.success:
                raise RuntimeError(f"path-integral integration failed: {sol.message}")
            Xt = sol.sol(ts).T.reshape(len(ts), N, n)
            R = system.rhs(Xt.reshape(-1, n)).reshape(len(ts), N, n) - (Xt - xe) @ A.T
            integrand = decay[:, None, :] * (R @ W)  # (K, N, m)
            cum = cumulative_simpson(integrand, x=ts, axis=0, initial=0)
            envelope = maximum_filter1d(np.abs(integrand), size=window, axis=0, mode="nearest")
            k = np.argmin(envelope, axis=0)  # (N, m), first index of the smallest envelope
            h = np.take_along_axis(cum, k[None], axis=0)[0]
            psi = (pts - xe) @ W + h
            out[start:start + N] = psi
            info["min_truncation_time"] = min(info["min_truncation_time"], float(ts[k.min()]))
            rel = np.take_along_axis(envelope, k[None], axis=0)[0] / np.maximum(np.abs(psi), 1e-300)
            info["max_relative_integrand_at_truncation"] = max(info["max_relative_integrand_at_truncation"],
                                                              float(rel.max()))
        return out

    return Eigenfunctions(lams, evaluate, info)
