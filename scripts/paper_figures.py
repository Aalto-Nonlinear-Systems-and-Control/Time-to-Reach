"""Draw the figures of the paper's examples from the current results.

    python -m scripts.paper_figures [--out results/figures] [--samples 3000] [--seed 0]

Figures (file names are those used by the paper):
  ex30_rev.png             Fig. 2(a): Duffing, non-convex sets, reachable set over [0, T] (gray) and
                           within the KRTB bounds (orange), one simulated trajectory (dashed)
  ex31_rev.png             Fig. 2(b): the same for the disjoint target
  ex60_w1000_h400_new.png  Fig. 4(a): cart-pole states and pole position over time, the unsafe sets,
                           the collision-time bounds (red) and the reach-time bounds (blue)
  ex62_new.png             Fig. 4(b): cart-pole snapshots within the reach-time horizon, the wall of X_U^2
  ex61_new.png             Fig. 4(c): cart-pole snapshots within the collision-time bounds, the wall of X_U^1
Next to each figure a JSON file records the data it was drawn from (bounds, horizon, sample
counts, the trajectory shown). The phase-plane figures share one axes rectangle, so that they
are the same size side by side; their labels and annotations are placed automatically where they
cover none of the drawn content; the trajectory shown starts at the centre of the initial set. The reach-time bounds are read from results/summary/krtb, the
trajectories are simulated with tools.common.simulate from the benchmark's own vector field.
"""

import argparse
import math
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from tools.common import jsonio, load_benchmark, paths  # noqa: E402
from tools.common.results import summary_path  # noqa: E402
from tools.common.sets import Box, Union  # noqa: E402
from tools.common.simulate import simulate  # noqa: E402

OUT_DIR = paths.RESULTS_DIR / "figures"
KRTB_SETTING = "setting_0"

# The style of the paper's figures. scripts.report draws its figures to the same standard.
STYLE = {
    "stream": "#b4bcd9", "initial": "#3457d5", "initial_edge": "#1b2f8a", "target": "#b22222", "target_edge": "#7a1414",
    "reach": "#8f8f8f", "within": "#ff8c00", "trajectory": "black",
    "font": 20, "label": 24, "tick": 14,
}
FONT = ["DejaVu Sans"]
PHASE_FIGSIZE = (6.6, 6.2)  # the phase-plane figures (Fig. 2), saved at 150 dpi
PHASE_AXES = {"left": 0.16, "right": 0.975, "bottom": 0.125, "top": 0.975}  # the same axes rectangle in every figure
STREAM = {"grid": 60, "linewidth": 0.8, "density": 1.6, "arrowsize": 1.1}  # the vector field behind them
FIGURES = {  # paper file name -> (benchmark, drawing function)
    "ex30_rev.png": ("DUFF_FIN_REA_NONCONVEX_LEVEL_SET", "duffing"),
    "ex31_rev.png": ("DUFF_FIN_REA_DISJOINT_BOX", "duffing"),
    "ex60_w1000_h400_new.png": ("CP_LQR_FIN_REA_UNSAFE", "cartpole_time"),
    "ex62_new.png": ("CP_LQR_FIN_REA_UNSAFE", "cartpole_reach"),
    "ex61_new.png": ("CP_LQR_FIN_REA_UNSAFE", "cartpole_collision"),
}


def bounds_of(bid, setting=KRTB_SETTING):
    """The KRTB reach-time bounds and the witness of a benchmark."""
    record = jsonio.read_json(summary_path("krtb", bid, setting))
    return [tuple(map(float, iv)) for iv in record["reach_time_bounds"]], (record["extra"].get("witness") or {}).get("x0")


def ceil2(x):
    return math.ceil(100 * x - 1e-9) / 100


def fmt(iv, digits=2):
    """[a, b] with `digits` decimals; an integer end (the horizon) without decimals."""
    return "[" + ", ".join(f"{v:g}" if float(v).is_integer() else f"{v:.{digits}f}" for v in iv) + "]"


# ----------------------------------------------------------------------------- Duffing (Fig. 2)


def phase_figure():
    """A phase-plane figure of the paper's size, in its font."""
    plt.rcParams["font.family"] = FONT
    return plt.subplots(figsize=PHASE_FIGSIZE)


def draw_streamlines(ax, rhs, xlim, ylim):
    """The vector field as light streamlines behind everything, as in Fig. 2."""
    gx, gy = np.meshgrid(np.linspace(*xlim, STREAM["grid"]), np.linspace(*ylim, STREAM["grid"]))
    with np.errstate(all="ignore"):
        F = rhs(np.c_[gx.ravel(), gy.ravel()])
    F = np.where(np.isfinite(F), F, 0.0)
    ax.streamplot(gx, gy, F[:, 0].reshape(gx.shape), F[:, 1].reshape(gx.shape), color=STYLE["stream"],
                  linewidth=STREAM["linewidth"], density=STREAM["density"], arrowsize=STREAM["arrowsize"], zorder=1)


def finish_phase_axes(ax, xlim, ylim, xlabel="$x_1$", ylabel="$x_2$"):
    """Limits, labels and tick sizes of the paper's phase-plane axes, in the fixed axes rectangle:
    the plot area is the same in every figure, whatever is drawn or written in it."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=STYLE["label"])
    ax.set_ylabel(ylabel, fontsize=STYLE["label"])
    ax.tick_params(labelsize=STYLE["tick"])
    ax.figure.subplots_adjust(**PHASE_AXES)


# ----------------------------------------------------------------------------- placing text

LABEL_GAP = 0.06  # data units kept clear around a text box
ARROW_CLEARANCE = 0.03  # an arrow passing closer than this to a drawn point counts as crossing it
TIP_RADIUS = 0.08  # ... except around its tip, which by design ends on the region it points at
ARROW_LENGTH = 0.35  # the preferred length of an annotation's arrow (data units)
SET_LABEL_GAP = 0.05  # between a set and its name


def text_size(ax, text, fontsize):
    """Width and height of a text in data units (the axes limits and position must be final)."""
    t = ax.text(0, 0, text, fontsize=fontsize)
    bb = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    t.remove()
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return x1 - x0, y1 - y0


def box_points(x, y, w, h, step=0.03):
    """Points filling a box, to count a placed text as an obstacle for the next one."""
    gx, gy = np.meshgrid(np.arange(x, x + w + step, step), np.arange(y, y + h + step, step))
    return np.c_[gx.ravel(), gy.ravel()]


def points_near_segment(P, a, b, clearance, tip_radius=0.0):
    """How many of the points P lie within `clearance` of the segment a-b, ignoring those within
    `tip_radius` of b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    near = np.linalg.norm(P - (a + t[:, None] * ab), axis=1) < clearance
    return int(np.sum(near & (np.linalg.norm(P - b, axis=1) >= tip_radius)))


def clear_spot(ax, w, h, candidates, obstacles, anchor=None, region=None):
    """The lower-left corner (x, y) of the w x h text box, chosen among the candidates: inside the axes,
    covering the fewest obstacle points and, among equals, nearest to `anchor`.

    With `region` (points of what the text describes), the text gets an arrow to the region's point nearest
    to it, and the obstacle points that the arrow would cross count as covered too. Returns (x, y, tip)."""
    (x_lo, x_hi), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
    best = None
    for x, y in candidates:
        if x < x_lo + LABEL_GAP or y < y_lo + LABEL_GAP or x + w > x_hi - LABEL_GAP or y + h > y_hi - LABEL_GAP:
            continue
        g = LABEL_GAP
        covered = int(np.sum((obstacles[:, 0] >= x - g) & (obstacles[:, 0] <= x + w + g)
                             & (obstacles[:, 1] >= y - g) & (obstacles[:, 1] <= y + h + g)))
        c = np.array([x + w / 2, y + h / 2])
        tip = None
        if region is not None:
            tip = region[np.argmin(np.linalg.norm(region - c, axis=1))]
            # the arrow leaves the box's edge; the obstacles it crosses on its way count, those at its tip do not
            start = c + (tip - c) * min(1.0, (w / 2) / max(abs(tip[0] - c[0]), 1e-9), (h / 2) / max(abs(tip[1] - c[1]), 1e-9))
            covered += points_near_segment(obstacles, start, tip, ARROW_CLEARANCE, TIP_RADIUS)
        if tip is not None:  # among clear spots, the one whose arrow has the preferred length
            dist = abs(float(np.linalg.norm(tip - start)) - ARROW_LENGTH)
        else:
            dist = 0.0 if anchor is None else float(np.linalg.norm(c - np.asarray(anchor)))
        key = (covered, dist)
        if best is None or key < best[0]:
            best = (key, (x, y, tip))
    return best[1]


def label_set(ax, s, text, obstacles, rng):
    """Write the name of a set next to it, where it covers the least; returns the points of the text box."""
    if isinstance(s, Box):
        (x0, x1), (y0, y1) = s.bounds
    else:  # the extent of the set itself, not of its declared bounding box
        P = s.sample(2000, rng)
        (x0, y0), (x1, y1) = P.min(axis=0), P.max(axis=0)
    w, h = text_size(ax, text, STYLE["label"])
    g = SET_LABEL_GAP
    cx, cy = (x0 + x1) / 2 - w / 2, (y0 + y1) / 2 - h / 2
    candidates = [(cx, y1 + g), (cx, y0 - g - h), (x1 + g, cy), (x0 - g - w, cy),  # above, below, right, left
                  (x1 + g, y1 + g), (x0 - g - w, y1 + g), (x1 + g, y0 - g - h), (x0 - g - w, y0 - g - h)]  # corners
    x, y, _ = clear_spot(ax, w, h, candidates, obstacles, anchor=((x0 + x1) / 2, (y0 + y1) / 2))
    ax.text(x, y, text, fontsize=STYLE["label"], ha="left", va="bottom", zorder=8)
    return box_points(x, y, w, h)


def annotate_region(ax, text, region, obstacles, step=0.1):
    """Write `text` (two lines) with an arrow to the nearest point of `region`, on the spot of a grid over the
    axes that covers the least and whose arrow crosses the least; returns the points of the text box."""
    w, h = text_size(ax, text, STYLE["font"])
    (x_lo, x_hi), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
    candidates = [(x, y) for x in np.arange(x_lo, x_hi, step) for y in np.arange(y_lo, y_hi, step)]
    region = np.asarray(region)[::max(1, len(region) // 2000)]
    x, y, tip = clear_spot(ax, w, h, candidates, obstacles, region=region)
    ax.annotate(text, xy=tuple(tip), xytext=(x, y), fontsize=STYLE["font"], ha="left", va="bottom",
                arrowprops={"arrowstyle": "->", "lw": 1.3, "shrinkB": 2}, zorder=9)
    return box_points(x, y, w, h)


def central_point(s, rng):
    """A point in the middle of a set: the centre of a box, else the sample nearest to the samples' mean."""
    if isinstance(s, Box):
        return s.bounds.mean(axis=1)
    P = s.sample(2000, rng)
    return P[np.argmin(np.linalg.norm(P - P.mean(axis=0), axis=1))]


def draw_set(ax, s, color, edge, label=None, where="above", n=400):
    """A box as a rectangle, anything else as the filled zero sub-level set of its inside function;
    the label, if any, above or below the set."""
    if isinstance(s, Box):
        (x0, x1), (y0, y1) = s.bounds
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor=edge, lw=1.2, zorder=6))
        cx, top, bottom = (x0 + x1) / 2, y1, y0
    else:
        (x0, x1), (y0, y1) = s.bounding_box()
        gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
        g = s.inside_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.contourf(gx, gy, g, levels=[-np.inf, 0], colors=[color], zorder=6)
        ax.contour(gx, gy, g, levels=[0], colors=[edge], linewidths=1.2, zorder=6)
        inside = np.c_[gx.ravel(), gy.ravel()][g.ravel() <= 0]
        cx, top, bottom = inside[:, 0].mean(), inside[:, 1].max(), inside[:, 1].min()
    if label is None:
        return
    if where == "above":
        ax.text(cx + 0.12, top + 0.02, label, fontsize=STYLE["label"], ha="center", va="bottom", zorder=8)
    else:
        ax.text(cx + 0.12, bottom - 0.03, label, fontsize=STYLE["label"], ha="center", va="top", zorder=8)


def reachable_points(bm, horizon, n_samples, rng, n_points):
    """Trajectory points of sampled initial states: (times, states) stacked over all samples."""
    X0 = np.vstack([bm.initial_set.sample(n_samples - n_samples // 3, rng), bm.initial_set.boundary_sample(n_samples // 3, rng)])
    T, X = [], []
    for x0 in X0:
        t, x, _ = simulate(bm.problem_system, x0, horizon, n_points=n_points)
        T.append(t)
        X.append(x)
    return np.concatenate(T), np.vstack(X), len(X0)


def figure_duffing(bid, n_samples, rng, xlim=(-1.0, 2.1), ylim=(-1.0, 2.05), t_end=None):
    """The reachable set of a 2D benchmark within its KRTB reach-time bounds (orange) and outside them (grey),
    with a trajectory through the middle of the initial set, up to t_end (default: the end of the last bound
    rounded up to 0.01 s). Fig. 2 uses the defaults."""
    bm = load_benchmark(bid)
    intervals, _ = bounds_of(bid)
    t_end = ceil2(max(hi for _, hi in intervals)) if t_end is None else float(t_end)
    horizon = (0.0, t_end)
    T, X, n0 = reachable_points(bm, horizon, n_samples, rng, n_points=int(300 * t_end))
    within = np.zeros(len(T), dtype=bool)
    for lo, hi in intervals:
        within |= (T >= lo) & (T <= hi)
    x0 = np.asarray(central_point(bm.initial_set, rng), float)  # a trajectory through the middle of the reachable set
    t_w, X_w, _ = simulate(bm.problem_system, x0, horizon, n_points=int(300 * t_end))

    fig, ax = phase_figure()
    finish_phase_axes(ax, xlim, ylim)  # first: the text placement below measures texts in data units
    draw_streamlines(ax, bm.problem_system.rhs, xlim, ylim)
    ax.scatter(X[~within, 0], X[~within, 1], s=1.5, color=STYLE["reach"], lw=0, zorder=2, rasterized=True)
    ax.scatter(X[within, 0], X[within, 1], s=1.5, color=STYLE["within"], lw=0, zorder=3, rasterized=True)
    ax.plot(X_w[:, 0], X_w[:, 1], "--", color=STYLE["trajectory"], lw=1.2, zorder=4)
    draw_set(ax, bm.initial_set, STYLE["initial"], STYLE["initial_edge"])
    targets = bm.target_set.sets if isinstance(bm.target_set, Union) else [bm.target_set]
    # several targets are numbered from the top, as in the paper (X_F^0 is reached first)
    targets = sorted(targets, key=lambda s: -s.bounding_box()[1, 1])
    for s in targets:
        draw_set(ax, s, STYLE["target"], STYLE["target_edge"])
    # texts go where they cover nothing that is drawn: the reachable set, the trajectory, the sets, other texts
    obstacles = [X[::max(1, len(X) // 40000)], X_w]
    for s in (bm.initial_set, *targets):
        obstacles.append(s.sample(300, rng))
    obstacles.append(label_set(ax, bm.initial_set, "$X_0$", np.vstack(obstacles), rng))
    for k, s in enumerate(targets):
        obstacles.append(label_set(ax, s, "$X_F$" if len(targets) == 1 else f"$X_F^{k}$", np.vstack(obstacles), rng))
    # one annotation per interval, pointing at the orange points of that interval (upper one first)
    regions = [X[(T >= lo) & (T <= hi)] for lo, hi in intervals]
    for (lo, hi), R in sorted(zip(intervals, regions), key=lambda z: -z[1][:, 1].mean()):
        obstacles.append(annotate_region(ax, f"Reachable set in\n{fmt((lo, hi))}", R, np.vstack(obstacles)))
    data = {"benchmark": bid, "reach_time_bounds": intervals, "horizon": list(horizon), "num_initial_samples": n0,
            "trajectory_x0": x0.tolist(), "trajectory": {"t": t_w.round(4).tolist(), "x": X_w.round(5).tolist()}}
    return fig, data


# ----------------------------------------------------------------------------- cart-pole (Fig. 4)

WALL_BENCHMARKS = {"X_U^1": "CP_LQR_FIN_REA_UNSAFE", "X_U^2": "CP_LQR_FIN_UNR_SAFE"}


def cartpole_walls():
    """The walls w of the unsafe sets {p + l sin(theta) >= w}, read from the targets 'w - x1 - sin(x3)' of the configs."""
    return {name: float(load_benchmark(bid).raw["target_sets"][0]["function"].split(" - ")[0])
            for name, bid in WALL_BENCHMARKS.items()}


def wall_label(name, wall):
    return f"${name}=\\{{x\\,|\\,p+l\\sin{{(\\theta)}}\\geq {wall:g}\\}}$"


def cartpole_trajectory(bm, t_end, n_points=2401):
    x0 = bm.initial_set.bounds[:, 0]  # the initial set is a point
    t, X, _ = simulate(bm.problem_system, x0, (0.0, t_end), n_points=n_points)
    l = bm.raw["system"]["parameters"]["l"]
    return t, X, X[:, 0] + l * np.sin(X[:, 2])


def cartpole_bounds():
    """Collision-time bounds (unsafe benchmark) and reach-time bounds (convergence benchmark)."""
    collision, _ = bounds_of("CP_LQR_FIN_REA_UNSAFE")
    reach, _ = bounds_of("CP_LQR_FIN_REA_CONVERGE")
    return collision[0], reach[0]


def figure_cartpole_time(bid):
    bm = load_benchmark(bid)
    collision, reach = cartpole_bounds()
    t_end = float(load_benchmark("CP_LQR_FIN_REA_CONVERGE").time_horizon[1])
    t, X, tip = cartpole_trajectory(bm, t_end)
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, (name, c) in enumerate(zip(("p", "v", r"$\theta$", r"$\dot{\theta}$"), ("tab:blue", "tab:orange", "tab:green", "tab:red"))):
        ax.plot(t, X[:, k], color=c, lw=1.1, label=name)
    ax.plot(t, tip, color="tab:purple", lw=3, alpha=0.8, label="Pole position")
    walls = cartpole_walls()
    w1, w2 = walls["X_U^1"], walls["X_U^2"]
    ylim = (min(-1.5, X.min() - 0.3), max(2.5, X.max() + 0.3, w2 + 0.8))  # the band of X_U^2 above its wall
    ax.axhspan(w2, ylim[1], color="mediumseagreen", alpha=0.6, lw=0, zorder=0)
    ax.axhspan(w1, w2, color="slategray", alpha=0.5, lw=0, zorder=0)
    # the texts sit in the right part of the plot, where the state has settled: the wall labels in their bands (right
    # of the legend, which takes the top left corner), the bounds below the settled state
    x_text = 0.45 * t_end
    ax.text(0.62 * t_end, (w2 + ylim[1]) / 2, wall_label("X_U^2", w2), fontsize=15, va="center")
    ax.text(0.62 * t_end, (w1 + w2) / 2, wall_label("X_U^1", w1), fontsize=15, va="center")
    ax.axvspan(*collision, color="tab:red", alpha=0.3, lw=1.5, ec="tab:red", zorder=0)
    ax.axvspan(*reach, color="royalblue", alpha=0.3, lw=1.5, ec="royalblue", zorder=0)
    # the bounds below the settled state: the collision text right of its band; the reach text left of its band when
    # the band starts in the right half, otherwise beside the collision text
    y_c, y_r = ylim[0] + 0.4, ylim[0] + 1.0
    ax.annotate(f"Collision-time bounds {fmt(collision, 3)}", xy=(collision[1], y_c), xytext=(collision[1] + 0.4, y_c),
                fontsize=17, va="center", arrowprops={"arrowstyle": "->", "lw": 1.5})
    right = reach[0] > 0.5 * t_end
    ax.annotate(f"Reach-time bounds {fmt(reach, 3)}", xy=(reach[0], y_r), xytext=(reach[0] - 0.4 if right else x_text, y_r),
                fontsize=17, va="center", ha="right" if right else "left", arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.set_xlim(0, t_end)
    ax.set_ylim(*ylim)
    ax.set_xlabel("t", fontsize=17)
    ax.tick_params(labelsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0.005, 0.985), ncol=5, fontsize=10, framealpha=0.95,
              columnspacing=1.2).set_zorder(10)
    fig.tight_layout()
    data = {"benchmark": bid, "collision_time_bounds": list(collision), "reach_time_bounds": list(reach),
            "horizon": [0.0, t_end], "x0": bm.initial_set.bounds[:, 0].tolist(),
            "trajectory": {"t": t.round(4).tolist(), "x": X.round(5).tolist(), "pole_position": tip.round(5).tolist()}}
    return fig, data


RAIL, WALL_TOP, YLIM = 1.45, 0.95, (-0.3, 1.95)  # snapshots: overhead rail, top of the walls, vertical range


def draw_cartpole(ax, p, theta, l, alpha, cart_color, pole_color, tip_color, lw=2.2, size=0.05, zorder=3):
    """The cart on the overhead rail and the pole hanging from it (theta from the hanging vertical)."""
    w, h = 0.8, 0.18
    ax.add_patch(Rectangle((p - w / 2, RAIL), w, h, facecolor=cart_color, edgecolor="black", lw=1.2, alpha=alpha, zorder=zorder))
    tip = (p + l * np.sin(theta), RAIL - l * np.cos(theta))
    ax.plot([p, tip[0]], [RAIL, tip[1]], color=pole_color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=zorder + 1)
    ax.plot([tip[0]], [tip[1]], "o", color=tip_color, ms=14, alpha=alpha, zorder=zorder + 2)
    return tip


def draw_scene(ax, x_lo, x_hi):
    """The floor (hatched below) and the overhead rail."""
    ax.axhline(0, color="black", lw=2, zorder=1)
    ax.add_patch(Rectangle((x_lo, -0.2), x_hi - x_lo, 0.2, facecolor="none", edgecolor="black", hatch="/", lw=0, zorder=1))
    ax.axhline(RAIL, color="black", lw=3, zorder=1)


def draw_wall(ax, wall):
    """The wall constrains the pole tip p + l sin(theta): it stands on the floor up to the height of the hanging mass,
    below the rail, so the cart passes over it."""
    ax.add_patch(Rectangle((wall, 0), 0.15, WALL_TOP, facecolor="none", edgecolor="black", hatch="///", lw=1.5, zorder=2))
    ax.text(wall + 0.22, 0.45, f"$p + l\\sin\\theta$\n$\\geq {wall:g}$", fontsize=14, ha="left", va="center")


def figure_cartpole_snapshots(bid, interval, wall, label_wall, n_snapshots=40):
    bm = load_benchmark(bid)
    t_end = float(load_benchmark("CP_LQR_FIN_REA_CONVERGE").time_horizon[1])
    t, X, tip = cartpole_trajectory(bm, t_end)
    l = bm.raw["system"]["parameters"]["l"]
    lo, hi = interval
    times = np.linspace(lo, hi, n_snapshots)
    idx = [int(np.argmin(np.abs(t - tau))) for tau in times]
    x_lo = min(X[idx, 0].min() - 0.65, -0.65)
    reach = max(X[idx, 0].max() + 0.45, tip[idx].max() + 0.3)  # the right end of the motion (carts are 0.8 wide)
    broken = wall - reach > 1.2  # a far wall: break the p axis between the motion and the wall
    x_hi = reach if broken else max(reach, wall + 1.3)  # room for the wall label right of the wall
    if broken:  # two panels at the same scale, the gap between them left out (marked by the break symbols)
        w_lo, w_hi = wall - 0.3, wall + 0.85
        fig, (ax, ax_w) = plt.subplots(1, 2, figsize=(5.2, 5.2), sharey=True,
                                       gridspec_kw={"width_ratios": [x_hi - x_lo, w_hi - w_lo], "wspace": 0.06})
        draw_scene(ax_w, w_lo, w_hi)
        draw_wall(ax_w, wall)
        ax_w.set_xlim(w_lo, w_hi)
        ax_w.set_aspect("equal")
        ax_w.spines["left"].set_visible(False)
        ax_w.tick_params(left=False, labelsize=12)
        ax.spines["right"].set_visible(False)
        for a, xs in ((ax, (1, 1)), (ax_w, (0, 0))):  # the break symbols on the top and bottom frame lines
            for y in (0, 1):
                a.plot([xs[0] - 0.02, xs[1] + 0.02], [y - 0.02, y + 0.02], transform=a.transAxes, color="black",
                       lw=1.2, clip_on=False)
    else:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        draw_wall(ax, wall)
    draw_scene(ax, x_lo, x_hi)
    for j, i in enumerate(idx):
        alpha = 1.0 if j == 0 else 0.12 + 0.6 * j / (len(idx) - 1) if label_wall == "end" else 0.9 * (1 - j / len(idx)) + 0.08
        first = j == 0
        tip_xy = draw_cartpole(ax, X[i, 0], X[i, 2], l, alpha, "royalblue" if first else "white",
                               "royalblue" if first else "royalblue", "black" if first else "gray", lw=2.2 if first else 0.8,
                               zorder=6 if first else 3)
        if first:  # the time label below the hanging mass, or left of it when the masses move on to the right
            label = f"$t={lo:g}$" if float(lo).is_integer() else f"$t={lo:.3f}$"
            if label_wall == "start":
                ax.text(tip_xy[0] - 0.12, tip_xy[1], label, fontsize=17, ha="right", va="center")
            else:
                ax.text(tip_xy[0], tip_xy[1] - 0.13, label, fontsize=17, ha="center", va="top")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.tick_params(labelsize=12)
    if broken:
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.1, top=0.97)
        ax.set_xlabel("p", fontsize=17)  # centred under both panels
        left, right = x_hi - x_lo, w_hi - w_lo
        ax.xaxis.set_label_coords(0.5 * (left + right + 0.03 * (left + right)) / left, -0.1)
    else:
        ax.set_xlabel("p", fontsize=17)
        fig.tight_layout()
    data = {"benchmark": bid, "interval": list(interval), "wall": wall, "snapshot_times": times.round(4).tolist(),
            "x0": bm.initial_set.bounds[:, 0].tolist(), "max_pole_position_in_interval": float(tip[idx].max()),
            "broken_axis": bool(broken)}
    fig.tight_bbox = True  # saved without the empty margins left by the equal aspect
    return fig, data


def figure_cartpole_reach(bid):
    _, reach = cartpole_bounds()
    collision, _ = cartpole_bounds()
    return figure_cartpole_snapshots(bid, (collision[1], reach[1]), cartpole_walls()["X_U^2"], "end")


def figure_cartpole_collision(bid):
    collision, _ = cartpole_bounds()
    return figure_cartpole_snapshots(bid, collision, cartpole_walls()["X_U^1"], "start")


DRAW = {"duffing": None, "cartpole_time": figure_cartpole_time, "cartpole_reach": figure_cartpole_reach,
        "cartpole_collision": figure_cartpole_collision}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--samples", type=int, default=3000, help="initial states simulated for the reachable sets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="+", help="file names to draw (default: all)")
    args = parser.parse_args(argv)
    out = paths.REPO_ROOT / args.out if not str(args.out).startswith("/") else type(OUT_DIR)(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = FONT
    for name, (bid, kind) in FIGURES.items():
        if args.only and name not in args.only:
            continue
        if kind == "duffing":
            fig, data = figure_duffing(bid, args.samples, np.random.default_rng(args.seed))
        else:
            fig, data = DRAW[kind](bid)
        fig.savefig(out / name, dpi=150, **({"bbox_inches": "tight", "pad_inches": 0.05} if getattr(fig, "tight_bbox", False) else {}))
        plt.close(fig)
        jsonio.write_json(out / (name.rsplit(".", 1)[0] + ".json"), {"figure": name, **data})
        print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
