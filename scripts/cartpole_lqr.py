"""Cart-pole model check, LQR design and closed-loop simulation (Example 2 of the paper).

The cart-pole is the one of Tedrake, "Underactuated Robotics", section "The Cart-Pole System"
(cart of mass m_c, point mass m_p on a massless rod of length l, horizontal force F on the
cart), in his coordinates: p is the cart position and theta the counter-clockwise pole angle,
zero hanging straight down, so the pole mass is at (p + l sin theta, -l cos theta). His equations,
as printed in the paper, are

    p'     = v
    v'     = [F + m_p sin(theta) (l omega^2 + g cos(theta))] / (m_c + m_p sin(theta)^2)
    theta' = omega
    omega' = [-F cos(theta) - m_p l omega^2 cos(theta) sin(theta) - (m_c + m_p) g sin(theta)]
             / (l (m_c + m_p sin(theta)^2))

with m_c = 2, m_p = 1, l = 1, g = 9.8065 in Example 2. The origin is the hanging equilibrium.
Without control the pole swings without damping (open-loop eigenvalues +-3.835j) and the cart
drifts (a double zero); the LQR feedback F = -K x damps the swing and brings the cart back, as a
crane brings a swinging load to rest. The weights Q = diag(56.5349, 27.434, 1.8689, 77.9197),
R = 6.6884 (found by a search for a lightly damped swing) give K = [2.907, 4.877, -3.770, -1.640]
and the closed-loop eigenvalues -0.771 +- 0.629j, -0.858 +- 3.698j (damping ratio 0.226 for the
swing): a spectrum inside a factor-2 band, as KRTB's path-integral eigenfunctions need
(2 max Re(lam) < min Re(lam)).

From the initial state (0.8, 0.3, -pi/4, 0.2) the pole swings from -45 deg to +36 deg and back a few
times; the pole tip p + l sin(theta) (the hanging mass) reaches the wall at 1.3 but not the one at
3.7. The targets of the unsafe benchmarks are the wall regions inside the box BOX, the bounding box
of the simulated trajectory enlarged by 5 % with p up to 4.0. The box is needed: the pole tip is a
flat output of the linearisation, so a half-space wall contains zeros of every principal
eigenfunction and the eigenfunction ranges on it cannot exclude the wall. The box also has to lie
in the basin of attraction of the closed loop, where the eigenfunctions are defined (--basin
checks this by simulation). KRTB excludes walls from about 2.5 times the pole-tip maximum
(1.359): the wall at 1.7 of the original example cannot be excluded, 3.7 leaves a margin (the
result holds with the box enlarged by 15 %).

The script
  1. derives the equations from the Lagrangian with sympy, checks them against the equations
     above and compares them with the CP_LQR benchmark configs;
  2. designs the LQR gain for the linearisation at the origin with the weights Q and R;
  3. simulates the closed loop from the initial state of the example and reports what the
     example relies on: whether the pole tip p + l sin(theta) passes the walls, whether the
     states at a collision lie in the bounding box of the unsafe benchmark's target, and when
     the state converges into the target shell 1e-30 <= ||x||_inf <= 1e-3 of CP_LQR_FIN_REA_CONVERGE;
  4. draws the time series (PNG) and renders a video of the motion (MP4, needs ffmpeg): the
     cart runs on an overhead rail, the walls stand on the floor up to the height of the mass.

Outputs go to results/cartpole_lqr/. With --write-configs the parameters, equations and
gains are written into the three CP_LQR configs (their results must then be recomputed).

    python -m scripts.cartpole_lqr                               # the default Q, R, x0 and walls
    python -m scripts.cartpole_lqr --q 50 25 2 80 --r 7 --no-video
    python -m scripts.cartpole_lqr --x0 0.4 0.15 -0.4 0.1 --walls 0.8 2.5
"""

import argparse
import itertools
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.linalg import solve_continuous_are  # noqa: E402

from scripts.paper_figures import FONT  # noqa: E402
from tools.common import expr, jsonio, paths  # noqa: E402

PARAMETERS = {"m_c": 2, "m_p": 1, "l": 1, "g": 9.8065}
PARAMETER_NAMES = tuple(PARAMETERS)
EQUATIONS = [  # Tedrake's cart-pole as printed (theta counter-clockwise, zero hanging down), F the force on the cart
    "x2",
    "(F + m_p*sin(x3)*(l*x4^2 + g*cos(x3)))/(m_c + m_p*sin(x3)^2)",
    "x4",
    "(-F*cos(x3) - m_p*l*x4^2*cos(x3)*sin(x3) - (m_c + m_p)*g*sin(x3))/(l*(m_c + m_p*sin(x3)^2))",
]
X0 = np.array([0.8, 0.3, -np.pi / 4, 0.2])  # the initial state of Example 2
WALLS = {"X_U^1": 1.3, "X_U^2": 3.7}  # walls: the pole tip p + l sin(theta) beyond a wall (on its side) is unsafe
WALL_BENCHMARKS = {"CP_LQR_FIN_REA_UNSAFE": "X_U^1", "CP_LQR_FIN_UNR_SAFE": "X_U^2"}  # the wall of each unsafe benchmark
SHELL = (1e-30, 1e-3)  # the target of CP_LQR_FIN_REA_CONVERGE: 1e-30 <= ||x||_inf <= 1e-3 (a box shell)
SHELL_NOTE = (f"The target is a box shell around the equilibrium (the origin): the box of half-width {SHELL[1]:g} "
              f"without the open box of half-width {SHELL[0]:g}. The equilibrium itself is excluded, as the "
              "time-to-reach bounds assume (x_e not in X_F); the hole is far smaller than the sample spacing, so the "
              "target's samples are those of the box.")
BOX = np.array([[-0.06, 4.0], [-0.88, 0.7], [-0.83, 0.66], [-1.72, 2.82]], float)  # bounding box of the unsafe targets (in the basin)
T_END = 12.0
BENCHMARKS = ("CP_LQR_FIN_REA_CONVERGE", "CP_LQR_FIN_REA_UNSAFE", "CP_LQR_FIN_UNR_SAFE")
DEFAULT_Q = (56.5349, 27.434, 1.8689, 77.9197)
DEFAULT_R = 6.6884
OUT_DIR = paths.RESULTS_DIR / "cartpole_lqr"
FPS = 30
STATE_NAMES = ("p", "v", "theta", "omega")


# ----------------------------------------------------------------------------- the model

def symbols():
    x = expr.state_symbols(4)
    F = sp.Symbol("F", real=True)
    params = sp.symbols(" ".join(PARAMETER_NAMES), real=True)
    return x, F, dict(zip(PARAMETER_NAMES, params))


def parse_equations(equations, parameters=None):
    """The equations as sympy expressions in x1..x4 and F; parameters substituted when given."""
    exprs = [expr.parse(e, 4, extra=("F", *PARAMETER_NAMES)) for e in equations]
    if parameters is None:
        return exprs
    _, _, params = symbols()
    return [e.subs({params[k]: parameters[k] for k in PARAMETER_NAMES}) for e in exprs]


def lagrangian_equations():
    """The accelerations of a cart (mass m_c) with a point mass m_p on a massless rod of length l,
    theta counter-clockwise from the hanging vertical, F on the cart, from Lagrange's equations."""
    (x1, x2, x3, x4), F, p = symbols()
    m_c, m_p, l, g = (p[k] for k in PARAMETER_NAMES)
    a, b = sp.symbols("a b")  # p'' and theta''
    vx, vy = x2 + l * sp.cos(x3) * x4, l * sp.sin(x3) * x4  # velocity of the pole mass at (p + l sin, -l cos)
    L = m_c * x2**2 / 2 + m_p * (vx**2 + vy**2) / 2 + m_p * g * l * sp.cos(x3)

    def d_dt(f):  # along the flow, with the unknown accelerations a, b
        return sp.diff(f, x1) * x2 + sp.diff(f, x3) * x4 + sp.diff(f, x2) * a + sp.diff(f, x4) * b

    sol = sp.solve([d_dt(sp.diff(L, x2)) - sp.diff(L, x1) - F, d_dt(sp.diff(L, x4)) - sp.diff(L, x3)],
                   [a, b], dict=True)[0]
    return [x2, sp.simplify(sol[a]), x4, sp.simplify(sol[b])]


def same_vector_field(f, h, rng, n=2000, tol=1e-9):
    """Whether two lists of expressions in x1..x4, F (and possibly the parameters) agree at random points."""
    x, F, params = symbols()
    args = (*x, F, *params.values())
    P = np.column_stack([rng.uniform(-1, 1, (n, 5)) * [2, 3, np.pi, 5, 10], rng.uniform(0.5, 3, (n, 4))])
    fun_f = sp.lambdify(args, f, "numpy")
    fun_h = sp.lambdify(args, h, "numpy")
    for a, b in zip(fun_f(*P.T), fun_h(*P.T)):
        if np.max(np.abs(np.broadcast_to(a, n) - np.broadcast_to(b, n))) > tol:
            return False
    return True


def config_equations(cfg):
    """The open-loop equations of a CP_LQR config (its control law u replaced by F) and its parameters."""
    system = cfg["system"]
    if set(system.get("definitions", {})) != {"u"}:
        raise ValueError("expected exactly the control law u in the config's definitions")
    values = {k: str(v) for k, v in system["parameters"].items()}
    values["u"] = "F"
    return [expr.substitute(e, values) for e in system["equations"]], system["parameters"]


def check_model(K=None, rng=None):
    """Compare the Lagrangian derivation, EQUATIONS and the benchmark configs. Returns a dict."""
    rng = rng or np.random.default_rng(0)
    model = parse_equations(EQUATIONS)
    derived = lagrangian_equations()
    out = {"equations_equal_lagrangian": all(sp.simplify(a - b) == 0 for a, b in zip(model, derived))
           and same_vector_field(model, derived, rng), "configs": {}}
    numeric = parse_equations(EQUATIONS, PARAMETERS)
    for bid in BENCHMARKS:
        cfg = jsonio.read_json(paths.BENCHMARK_DIR / f"{bid}.json")
        texts, params = config_equations(cfg)
        in_config = [expr.parse(e, 4, extra=("F",)) for e in texts]
        gains = np.array([params.get(f"k{i}", np.nan) for i in range(1, 5)], float)
        entry = {"parameters_equal": all(float(params.get(k, np.nan)) == float(PARAMETERS[k]) for k in PARAMETER_NAMES),
                 "equations_equal": same_vector_field(numeric, in_config, rng),
                 "gains": gains.tolist()}
        if K is not None:
            entry["gains_equal"] = bool(np.allclose(gains, K, atol=1e-9))
        out["configs"][bid] = entry
    return out


# ----------------------------------------------------------------------------- LQR

def linearisation():
    """A, B of the open loop at the origin (the hanging equilibrium)."""
    x, F, _ = symbols()
    f = sp.Matrix(parse_equations(EQUATIONS, PARAMETERS))
    at = {**{xi: 0 for xi in x}, F: 0}
    A = np.array(f.jacobian(sp.Matrix(x)).subs(at), float)
    B = np.array(f.diff(F).subs(at), float).reshape(4, 1)
    return A, B


def lqr(A, B, Q, R):
    """The gain K of u = -K x minimising the integral of x'Qx + u'Ru, and the closed-loop eigenvalues."""
    Q, R = np.diag(Q).astype(float), np.array([[float(R)]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P).ravel()
    return K, np.sort_complex(np.linalg.eigvals(A - B @ K[None, :]))


def closed_loop(K):
    """f(t, x) of the closed loop with F = -K x."""
    x, F, _ = symbols()
    fun = sp.lambdify((*x, F), parse_equations(EQUATIONS, PARAMETERS), "numpy")
    K = np.asarray(K, float)

    def f(t, state):
        return np.array(fun(*state, -K @ state), float)
    return f


# ----------------------------------------------------------------------------- simulation

def intervals(mask, t):
    """[t_start, t_end] of the maximal runs where mask is true."""
    out, start = [], None
    for k, m in enumerate(mask):
        if m and start is None:
            start = k
        if not m and start is not None:
            out.append([float(t[start]), float(t[k - 1])])
            start = None
    if start is not None:
        out.append([float(t[start]), float(t[-1])])
    return out


def beyond(tip, wall):
    """Whether the pole tip is past the wall: a wall at a positive position is hit from the left, one at a
    negative position from the right."""
    return tip > wall if wall > 0 else tip < wall


def simulate(K, x0=X0, t_end=T_END, dt=1e-3):
    sol = solve_ivp(closed_loop(K), (0.0, t_end), np.asarray(x0, float), method="DOP853", rtol=1e-10, atol=1e-12,
                    dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    t = np.arange(0.0, t_end + dt / 2, dt)
    X = sol.sol(t).T
    return t, X, -(X @ np.asarray(K, float))


def metrics(t, X, u, walls=WALLS):
    tip = X[:, 0] + PARAMETERS["l"] * np.sin(X[:, 2])
    ninf = np.abs(X).max(axis=1)
    out = {"tip_range": [float(tip.min()), float(tip.max())], "walls": {},
           "shell_intervals": intervals((ninf >= SHELL[0]) & (ninf <= SHELL[1]), t),
           "first_time_norm_below": {f"{lvl:g}": (float(t[np.argmax(ninf < lvl)]) if (ninf < lvl).any() else None) for lvl in SHELL},
           "norm_at_end": float(ninf[-1]), "max_abs_state": dict(zip(STATE_NAMES, np.abs(X).max(axis=0).tolist())),
           "max_abs_force": float(np.abs(u).max())}
    for name, wall in walls.items():
        hit = beyond(tip, wall)
        entry = {"wall": wall, "hit": bool(hit.any()), "intervals": intervals(hit, t)}
        if hit.any():
            states = X[hit]
            entry["state_ranges"] = {n: [float(states[:, i].min()), float(states[:, i].max())] for i, n in enumerate(STATE_NAMES)}
            entry["inside_box"] = bool(((states >= BOX[:, 0]) & (states <= BOX[:, 1])).all())
        out["walls"][name] = entry
    return out


# ----------------------------------------------------------------------------- outputs

def wall_text(name, wall):
    return f"{name} (tip {'>' if wall > 0 else '<'} {wall:g})"


def summary_lines(Q, R, K, eig, m, x0=X0):
    yield f"x0 = {np.asarray(x0).round(6).tolist()}"
    yield f"Q = diag{tuple(float(q) for q in Q)}, R = {float(R):g}"
    yield "K = [" + ", ".join(f"{k:.6f}" for k in K) + "]   (F = -K x)"
    yield "closed-loop eigenvalues: " + ", ".join(f"{e.real:.3f}{e.imag:+.3f}j" for e in eig)
    re = np.sort(eig.real)
    yield f"path-integral condition 2 max Re < min Re: margin {re[0] - 2 * re[-1]:.3f} ({'ok' if re[0] - 2 * re[-1] > 0 else 'FAILS'})"
    yield "pole tip p + l sin(theta) in [{:.3f}, {:.3f}]".format(*m["tip_range"])
    for name, w in m["walls"].items():
        span = ", ".join(f"[{a:.3f}, {b:.3f}]" for a, b in w["intervals"]) if w["hit"] else "never"
        extra = f"; states at the collision inside the box: {w['inside_box']}" if w["hit"] else ""
        yield f"{wall_text(name, w['wall'])}: {span}{extra}"
    shell = ", ".join(f"[{a:.3f}, {b:.3f}]" for a, b in m["shell_intervals"]) or "never"
    yield f"in the target shell {SHELL[0]:g} <= ||x||_inf <= {SHELL[1]:g}: {shell}"
    yield f"||x||_inf at the end: {m['norm_at_end']:.2e}; max |F| = {m['max_abs_force']:.2f}; " \
          + "max |x| = " + ", ".join(f"{k} {v:.3f}" for k, v in m["max_abs_state"].items())


def title_text(K):
    return ("Cart-pole (Tedrake, $\\theta$ from the hanging vertical), $m_c = 2$, $m_p = 1$, $l = 1$, $g = 9.8065$;  "
            "LQR $F = -Kx$ with $K$ = [" + ", ".join(f"{k:.3f}" for k in K) + "]")


def figure(path, t, X, u, K, eig, m, walls=WALLS):
    plt.rcParams["font.family"] = FONT
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for i, (name, label) in enumerate(zip(STATE_NAMES, ("$p$ (m)", "$v$ (m/s)", r"$\theta$ (rad)", r"$\omega$ (rad/s)"))):
        axes[0].plot(t, X[:, i], lw=1.6, label=label)
    axes[0].legend(ncol=4, loc="upper right", fontsize=10)
    axes[0].set_ylabel("state")
    tip = X[:, 0] + PARAMETERS["l"] * np.sin(X[:, 2])
    axes[1].plot(t, tip, color="tab:purple", lw=1.8, label=r"pole tip $p + l\sin\theta$")
    axes[1].plot(t, X[:, 0], color="tab:blue", lw=1.2, label="cart $p$")
    for (name, wall), color in zip(walls.items(), ("#b22222", "#7a1414")):
        axes[1].axhline(wall, color=color, lw=1.2, ls="--", label=f"wall ${name}$ at {wall:g}")
    first = next(iter(m["walls"].values()))
    for a, b in first["intervals"]:
        axes[1].axvspan(a, b, color="#b22222", alpha=0.15, lw=0)
    axes[1].set_ylabel("position (m)")
    axes[1].legend(loc="center right", fontsize=10)
    ninf = np.abs(X).max(axis=1)
    axes[2].semilogy(t, ninf, color="black", lw=1.6, label=r"$\|x\|_\infty$")
    lo, hi = axes[2].get_ylim()  # the band from the bottom of the plot: its inner radius lies far below
    axes[2].axhspan(max(SHELL[0], lo), SHELL[1], color="#3457d5", alpha=0.15, lw=0, label="target shell")
    axes[2].set_ylim(lo, hi)
    axes[2].set_ylabel("norm")
    axes[2].legend(loc="upper right", fontsize=10)
    axes[3].plot(t, u, color="tab:green", lw=1.6)
    axes[3].set_ylabel("$F = -Kx$ (N)")
    axes[3].set_xlabel("$t$ (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title_text(K) + "\nclosed-loop eigenvalues " + ", ".join(f"${e.real:.3f}{e.imag:+.3f}j$" for e in eig), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def video(path, t, X, u, K, walls=WALLS, fps=FPS):
    """Real-time animation of the cart and the pole, the walls and the time series."""
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found: no video", file=sys.stderr)
        return False
    plt.rcParams["font.family"] = FONT
    l = PARAMETERS["l"]
    step = max(1, round(1 / (fps * (t[1] - t[0]))))
    frames = range(0, len(t), step)
    tip = X[:, 0] + l * np.sin(X[:, 2])
    first_name, first_wall = next(iter(walls.items()))
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    scene = fig.add_axes([0.04, 0.08, 0.55, 0.80])
    panels = [fig.add_axes([0.69, y, 0.29, 0.21]) for y in (0.67, 0.38, 0.09)]
    fig.suptitle(title_text(K), fontsize=12, y=0.965)

    cart_w, cart_h = 0.5, 0.2
    floor = -1.35  # the cart runs on an overhead rail at height 0, the pole hangs below it
    lo = min(X[:, 0].min() - cart_w, tip.min(), *walls.values()) - 0.3
    hi = max(X[:, 0].max() + cart_w, tip.max(), *walls.values()) + 0.3
    scene.set_xlim(lo, hi)
    scene.set_ylim(floor - 0.15, 0.6)
    scene.set_aspect("equal")
    scene.set_xlabel("$x$ (m)")
    scene.set_yticks([])
    scene.plot([lo, hi], [0, 0], color="#555555", lw=3, zorder=1)  # the rail
    scene.plot([lo, hi], [floor, floor], color="#999999", lw=2, zorder=1)  # the floor
    # the walls stand on the floor and reach the height of the hanging mass, below the rail: the cart passes over
    # them, the mass hits them (the unsafe sets constrain the pole tip p + l sin(theta) only)
    for (name, wall), (color, ls) in zip(walls.items(), (("#b22222", "-"), ("#7a1414", "--"))):
        scene.plot([wall, wall], [floor, -0.45], color=color, lw=3, ls=ls, zorder=2)
        scene.text(wall, -0.4, f"${name}$ ({wall:g})", color=color, fontsize=11, ha="center", va="bottom")
    cart = Rectangle((X[0, 0] - cart_w / 2, 0), cart_w, cart_h, facecolor="#3457d5", edgecolor="#1b2f8a", lw=1.5, zorder=4)
    scene.add_patch(cart)
    rod, = scene.plot([], [], color="black", lw=3, solid_capstyle="round", zorder=5)
    bob, = scene.plot([], [], "o", color="#ff8c00", ms=16, mec="black", zorder=6)
    trail, = scene.plot([], [], color="#ff8c00", lw=1, alpha=0.6, zorder=3)
    clock = scene.text(0.02, 0.96, "", transform=scene.transAxes, fontsize=14, va="top")
    status = scene.text(0.02, 0.88, "", transform=scene.transAxes, fontsize=12, va="top", color="#b22222")
    scene.set_title("walls: " + ", ".join(f"${n} = \\{{p + l\\sin\\theta {'>' if w > 0 else '<'} {w:g}\\}}$" for n, w in walls.items()),
                    fontsize=11)

    series = [(tip, ("pole tip $p + l\\sin\\theta$ and the walls", "tip (m)"), "tab:purple"),
              (X[:, 2], ("pole angle and cart position", "$\\theta$ (rad), $p$ (m)"), "tab:red"),
              (u, ("control force", "$F$ (N)"), "tab:green")]
    lines, cursors = [], []
    for ax, (y, label, color) in zip(panels, series):
        ax.set_xlim(t[0], t[-1])
        pad = 0.08 * (y.max() - y.min() + 1e-9)
        ax.set_ylim(min(y.min(), 0) - pad, max(y.max(), 0) + pad)
        ax.plot(t, y, color=color, lw=1, alpha=0.25)
        line, = ax.plot([], [], color=color, lw=1.8, label=label[1].split(",")[0])
        cursor = ax.axvline(t[0], color="#555555", lw=0.8)
        ax.set_title(label[0], fontsize=10)
        ax.set_ylabel(label[1], fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)
        lines.append(line)
        cursors.append(cursor)
    for (name, wall), color in zip(walls.items(), ("#b22222", "#7a1414")):
        panels[0].axhline(wall, color=color, lw=1, ls="--")
    panels[0].set_ylim(min(tip.min(), 0, *walls.values()) - 0.15, max(tip.max(), 0, *walls.values()) + 0.15)
    panels[1].plot(t, X[:, 0], color="tab:blue", lw=1, alpha=0.25)
    line_p, = panels[1].plot([], [], color="tab:blue", lw=1.8, label="$p$ (m)")
    panels[1].legend(loc="upper right", fontsize=9, ncol=2)
    panels[1].set_ylim(min(X[:, 0].min(), X[:, 2].min()) - 0.1, max(X[:, 0].max(), X[:, 2].max()) + 0.1)
    panels[2].set_xlabel("$t$ (s)", fontsize=10)

    def update(k):
        p, th = X[k, 0], X[k, 2]
        cart.set_x(p - cart_w / 2)
        bx, by = p + l * np.sin(th), -l * np.cos(th)  # the pivot on the rail, the mass below it
        rod.set_data([p, bx], [0, by])
        bob.set_data([bx], [by])
        hit = beyond(tip[k], first_wall)
        bob.set_color("#b22222" if hit else "#ff8c00")
        trail.set_data(tip[:k + 1], -l * np.cos(X[:k + 1, 2]))
        clock.set_text(f"$t$ = {t[k]:5.2f} s")
        status.set_text(f"collision with ${first_name}$" if hit else "")
        for line, (y, _, _) in zip(lines, series):
            line.set_data(t[:k + 1], y[:k + 1])
        line_p.set_data(t[:k + 1], X[:k + 1, 0])
        for cursor in cursors:
            cursor.set_xdata([t[k], t[k]])

    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart"])
    with writer.saving(fig, str(path), dpi=100):
        for k in frames:
            update(k)
            writer.grab_frame()
    plt.close(fig)
    return True


def target_samples(box, wall, n, rng):
    """Samples of {p + l sin(theta) >= wall} ∩ box as KRTB draws them: uniform points inside, and "vertex" points
    with three coordinates at a bound of the box and p or theta solved onto the level set."""
    l = PARAMETERS["l"]
    U = rng.uniform(box[:, 0], box[:, 1], (200 * n, 4))
    U = U[U[:, 0] + l * np.sin(U[:, 2]) >= wall][:n]
    V = []
    for corner in itertools.product(*box):
        x = np.array(corner, float)
        x[0] = wall - l * np.sin(x[2])
        if box[0, 0] <= x[0] <= box[0, 1]:
            V.append(x.copy())
        x = np.array(corner, float)
        if -1 <= wall - x[0] <= 1:
            x[2] = np.arcsin(wall - x[0])
            if box[2, 0] <= x[2] <= box[2, 1]:
                V.append(x)
    return U, np.array(V).reshape(-1, 4)


def converges(K, X, t_end=12.0, tol=1e-2, blow_up=50.0):
    """Whether the closed loop from each row of X reaches ||x||_inf <= tol by t_end without |x| exceeding blow_up."""
    f = closed_loop(K)

    def blown(t, x):
        return blow_up - np.abs(x).max()
    blown.terminal = True
    out = np.empty(len(X), bool)
    for i, x0 in enumerate(X):
        sol = solve_ivp(f, (0.0, t_end), x0, method="DOP853", rtol=1e-7, atol=1e-9, events=blown)
        out[i] = sol.success and sol.status == 0 and np.abs(sol.y[:, -1]).max() <= tol
    return out


def basin_check(K, box=BOX, walls=WALLS, n=800, seed=0):
    """Do the samples of the unsafe targets {tip >= wall} ∩ box converge to the origin? Yields one line per wall."""
    rng = np.random.default_rng(seed)
    ok = True
    for name, wall in walls.items():
        U, V = target_samples(box, wall, n, rng)
        cu, cv = converges(K, U), converges(K, V)
        ok &= cu.all() and cv.all()
        yield (f"{name} (tip >= {wall:g}) in the box {fmt_box(box)}: {len(U)} uniform samples, {(~cu).sum()} diverge; "
               f"{len(V)} vertex samples, {(~cv).sum()} diverge")
    yield "all samples converge" if ok else "SOME SAMPLES DIVERGE: the target box is not inside the basin of attraction"


def fmt_box(box):
    return " x ".join(f"[{a:g}, {b:g}]" for a, b in box)


def write_configs(K):
    """Write the model (parameters, equations), the gains, the initial state, the walls and the target boxes into
    the CP_LQR configs, with notes that describe them."""
    params = {**PARAMETERS, **{f"k{i + 1}": float(k) for i, k in enumerate(K)}}
    equations = [re.sub(r"\bF\b", "u", e) for e in EQUATIONS]
    model_note = ("State x = (p, v, theta, omega): cart position and velocity, pole angle (counter-clockwise, zero "
                  "hanging down, positive with the pole mass towards positive p) and angular velocity; Tedrake's "
                  "cart-pole as printed. u = -K x is the LQR state feedback (gains k1..k4) designed by "
                  "scripts/cartpole_lqr.py with Q = diag(" + ", ".join(f"{q:g}" for q in DEFAULT_Q) + f"), R = {DEFAULT_R:g}.")
    box_note = (f"The target is restricted to the box {fmt_box(BOX)}, which lies in the basin of attraction of the "
                "LQR equilibrium (needed by eigenfunction-based methods; checked by simulating samples of the target, "
                "scripts/cartpole_lqr.py --basin); the unrestricted pole-tip region extends to states that do not converge.")
    for bid in BENCHMARKS:
        path = paths.BENCHMARK_DIR / f"{bid}.json"
        cfg = jsonio.read_json(path)
        cfg["system"] = {"type": "nonlinear", "parameters": params, "definitions": {"u": "-(k1*x1 + k2*x2 + k3*x3 + k4*x4)"},
                         "equations": equations}
        cfg["initial_sets"] = [{"type": "box", "bounds": [[float(x)] * 2 for x in X0]}]  # theta0 = -pi/4 in full precision
        notes = ["The initial set is a single state (degenerate box).", model_note]
        if bid not in WALL_BENCHMARKS:
            cfg["description"] = ("Cart-pole with LQR feedback (4D), finite horizon: the state converges into a box shell "
                                  "around the hanging equilibrium.")
            notes.append(SHELL_NOTE)
        if bid in WALL_BENCHMARKS:
            wall = WALLS[WALL_BENCHMARKS[bid]]
            (target,) = cfg["target_sets"]
            target.update({"type": "level_set", "function": f"{wall:g} - x1 - sin(x3)", "bounding_box": BOX.tolist()})
            notes.append(box_note)
            verb = "reaches" if cfg["expected_result"] == "reachable" else "never reaches"
            cfg["description"] = (f"Cart-pole with LQR feedback (4D), finite horizon: the pole-tip coordinate x1 + sin(x3) "
                                  f"{verb} {wall:g} inside the box {fmt_box(BOX)}.")
        cfg["notes"] = notes
        jsonio.write_json(path, cfg)
        print(f"wrote {path.relative_to(paths.REPO_ROOT)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--q", type=float, nargs=4, default=DEFAULT_Q, metavar="Q", help=f"diagonal of Q (default {DEFAULT_Q})")
    ap.add_argument("--r", type=float, default=DEFAULT_R, help=f"R (default {DEFAULT_R})")
    ap.add_argument("--x0", type=float, nargs=4, default=X0.tolist(), metavar="X", help="initial state (default: the example's)")
    ap.add_argument("--walls", type=float, nargs=2, default=list(WALLS.values()), metavar="W",
                    help="positions of the walls X_U^1, X_U^2 (negative: on the left; default "
                         + " ".join(f"{w:g}" for w in WALLS.values()) + ")")
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--out", default=str(OUT_DIR), help="output directory (default results/cartpole_lqr)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--write-configs", action="store_true", help="write the model and the gains into the CP_LQR configs")
    ap.add_argument("--basin", action="store_true", help="check that the target boxes of the unsafe benchmarks lie in the basin")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = paths.REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    x0, walls = np.array(args.x0, float), dict(zip(WALLS, args.walls))

    A, B = linearisation()
    K, eig = lqr(A, B, args.q, args.r)
    check = check_model(K)
    print("model: equations equal the Lagrangian derivation:", check["equations_equal_lagrangian"])
    for bid, c in check["configs"].items():
        print(f"  {bid}: parameters equal {c['parameters_equal']}, equations equal {c['equations_equal']}, "
              f"gains equal {c.get('gains_equal')}")
    print("open-loop eigenvalues at the origin:", np.round(np.sort_complex(np.linalg.eigvals(A)), 3))
    t, X, u = simulate(K, x0, args.t_end)
    m = metrics(t, X, u, walls)
    for line in summary_lines(args.q, args.r, K, eig, m, x0):
        print(line)

    sub = slice(None, None, 10)  # 0.01 s in the JSON
    jsonio.write_json(out / "cartpole_lqr.json", {
        "model": {"source": "Tedrake, Underactuated Robotics, 'The Cart-Pole System', as printed (theta counter-clockwise, "
                            "zero hanging down)",
                  "parameters": PARAMETERS, "equations": EQUATIONS, "check": check},
        "lqr": {"Q": [float(q) for q in args.q], "R": float(args.r), "A": A.tolist(), "B": B.ravel().tolist(),
                "K": K.tolist(), "closed_loop_eigenvalues": [[e.real, e.imag] for e in eig]},
        "simulation": {"x0": x0.tolist(), "walls": walls, "t_end": args.t_end, **m,
                       "trajectory": {"t": t[sub].round(4).tolist(), "x": X[sub].round(6).tolist(), "F": u[sub].round(6).tolist()}}})
    figure(out / "cartpole_lqr.png", t, X, u, K, eig, m, walls)
    print(f"wrote {out / 'cartpole_lqr.json'} and cartpole_lqr.png")
    if not args.no_video and video(out / "cartpole_lqr.mp4", t, X, u, K, walls):
        print(f"wrote {out / 'cartpole_lqr.mp4'}")
    if args.basin:
        for line in basin_check(K):
            print(line)
    if args.write_configs:
        write_configs(K)
    return 0


if __name__ == "__main__":
    sys.exit(main())
