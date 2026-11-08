# Time-to-Reach Verification Using the Koopman Spectrum

**Online report: <https://aalto-nonlinear-systems-and-control.github.io/Time-to-Reach/>.** The report covers every benchmark. For each one it shows the specification, the result of every tool, and the simulation baseline the results are checked against.

A shorter version shows only KRTB and the simulation: <https://aalto-nonlinear-systems-and-control.github.io/Time-to-Reach/krtb.html>.

This repository holds the code, the benchmarks and the results of the paper *Time-to-reach Bounds for Verification of Dynamical Systems Using the Koopman Spectrum* by Jianqiang Ding and Shankar A. Deka. The preprint is [arXiv:2411.05554](https://arxiv.org/abs/2411.05554). The repository contains:

- **KRTB**, the method of the paper. It computes reach-time bounds from the principal Koopman eigenfunctions (`tools/krtb`).
- **Runners for the five tools of the comparison** (Table II): CORA, JuliaReach, dReach, hj_reachability and SOSTOOLS (`tools/`).
- **The 16 benchmarks** as JSON configs, and the settings of every tool (`configs/`).
- **The results** behind the paper's tables, and the simulation baseline they are checked against (`results/summary/`).
- **The scripts** that write the paper's tables and figures (`scripts/`).

The benchmarks are documented in [Benchmark_Specifications.pdf](Benchmark_Specifications.pdf). Changes since its first published version are shown in blue.

## Contents

- [Branches](#branches)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Reproducing the paper](#reproducing-the-paper)
- [Benchmarks](#benchmarks)
- [Results](#results)
- [Tool settings](#tool-settings)
- [Tests](#tests)

## Branches

| Branch                  | Content                                                                                                                                                                                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main`                  | What is needed to reproduce the paper: the benchmarks, the tool runners, the scripts for the tables and figures, the results and the tests.                                                                                      |
| `refactor/code-cleanup` | The development branch, with the full history. On top of `main` it has the generator of the online report (`scripts/report.py` and the scripts for its animated pipeline figure) and the original hand-written dReach models (`benchmarks/`). |
| `gh-pages`              | The online report (`index.html`, `krtb.html`), built with `python -m scripts.report` on the development branch.                                                                                                                  |

## Repository layout

```text
configs/benchmarks/     one JSON file per benchmark: dynamics, domain, initial and target sets, horizon, expected result
configs/settings/       the parameters of every tool (one comparison setting, setting_0)
configs/schema/         JSON schemas of the configs and of the result records
tools/common/           shared code: loading benchmarks, sets, simulation, result records, memory and time limits
tools/krtb/             KRTB: principal eigenfunctions (analytic or path integral) and reach-time bounds
tools/cora/             CORA runner (MATLAB; CORA is the submodule tools/cora/CORA)
tools/juliareach/       JuliaReach runner (Julia; pinned Project.toml and Manifest.toml)
tools/dreach/           dReach runner and Dockerfile; the .drh models are generated from the configs
tools/hj/               hj_reachability runner
tools/sosbc/            SOS barrier certificates (MATLAB; SOSTOOLS is the submodule tools/sosbc/SOSTOOLS)
scripts/                simulation baseline, evaluation, tables, figures, robustness study, cart-pole LQR design
results/summary/        result records of every tool, the simulation baseline, the evaluation and the tables
tests/                  pytest suite
```

Everything a run writes outside `results/summary/` goes to other folders under `results/`, such as raw tool output, figures and videos. These folders are not tracked.

## Installation

```sh
git clone --recurse-submodules https://github.com/Aalto-Nonlinear-Systems-and-Control/Time-to-Reach.git
cd Time-to-Reach
mamba env create -f environment.yml   # Python 3.12 with pinned packages
conda activate time_to_reach
```

KRTB, the simulation baseline, HJ and all scripts need only this environment. KRTB solves small linear-fractional programs with MOSEK through cvxpy. MOSEK needs a licence file, by default at `~/mosek/mosek.lic`; academic licences are free. The other tools need:

| Tool           | Needs                                                                                                                                                          |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CORA           | MATLAB R2025a, found through `$MATLAB_BIN`, `/Applications/MATLAB_R2025a.app` or `matlab` on the `PATH`. CORA v2026.1.0 is the submodule `tools/cora/CORA`. |
| SOSTOOLS       | MATLAB R2025a with MOSEK for MATLAB; the results were computed with MOSEK 11.0.25. SOSTOOLS v4.03 is the submodule `tools/sosbc/SOSTOOLS`.                   |
| JuliaReach     | Julia 1.13.1 (for example with juliaup), then `julia --project=tools/juliareach -e 'using Pkg; Pkg.instantiate()'`.                                            |
| dReach         | Docker. Build the image with `docker build --platform linux/amd64 -t dreach:dreal3-7f31797b tools/dreach`. It is x86-64 only, so it runs under emulation on Apple Silicon. |

## Reproducing the paper

All commands run from the repository root in the `time_to_reach` environment.

### Table II: comparison of the tools

```sh
python -m scripts.check_benchmarks       # simulation baseline -> results/summary/ground_truth/
python -m tools.krtb.run --all           # KRTB on all benchmarks -> results/summary/krtb/
python -m tools.cora.run --all           # the same for juliareach, dreach, hj and sosbc
python -m scripts.evaluate_results       # check every result against the simulation baseline
python -m scripts.paper_table            # Table II (paper_table.tex) and Table I (paper_table_mas.tex)
python -m scripts.aggregate              # every verdict with its timing breakdown (table.md, .csv, .tex)
```

Each runner also takes `--benchmark ID` for a single benchmark. A runner writes one record per benchmark to `results/summary/<tool>/<ID>/setting_0.json`. A record holds the verdict, the reach-time bounds, the timings, the parameters and the provenance (commit, platform, versions).

`scripts.evaluate_results` compares each record with the simulation baseline. It exits with an error if a tool claims "unreachable" for a benchmark that a simulated trajectory reaches. The tables in `results/summary/` are generated from the records, so they do not need editing by hand.

### Figures

```sh
python -m scripts.paper_figures          # Fig. 2 (Duffing) and Fig. 4 (cart-pole) -> results/figures/
python -m scripts.robustness             # Fig. 3, robustness to eigenfunction errors -> results/robustness/
```

The figures are drawn from the stored KRTB results and from freshly simulated trajectories. Next to every figure is a JSON file with the data it was drawn from.

The robustness study perturbs the eigenfunction values with multiplicative noise and recomputes the bounds with the KRTB code. The details are in the docstring of [`scripts/robustness.py`](scripts/robustness.py).

### Example 2: the cart-pole and its LQR controller

The cart-pole is Tedrake's cart-pole as printed in the paper. The pole angle is measured counter-clockwise from the hanging vertical, so the origin is the hanging equilibrium.

Without control, the pole swings without damping and the cart drifts. The LQR feedback damps the swing and brings the cart back, like a crane bringing a swinging load to rest:

- Weights Q = diag(56.5349, 27.434, 1.8689, 77.9197) and R = 6.6884.
- Gain K = [2.907, 4.877, −3.770, −1.640], applied as F = −Kx.
- Closed-loop eigenvalues −0.771 ± 0.629j and −0.858 ± 3.698j.

```sh
python -m scripts.cartpole_lqr                  # model check, LQR design, simulation, figure and video (needs ffmpeg)
python -m scripts.cartpole_lqr --write-configs  # also write the model and the gain into the CP_LQR configs
```

The script checks the equations against a Lagrangian derivation. It simulates the closed loop from x0 = (0.8, 0.3, −π/4, 0.2). The pole tip reaches the wall at 1.3 but not the one at 3.7.

## Benchmarks

The IDs of the configs are longer than the names in the paper and the PDF. The mapping is:

| Benchmark ID (`configs/benchmarks/`) | Name in the paper  | Dim | Initial set | Target set             | Horizon  | Expected    |
| ------------------------------------ | ------------------ | --- | ----------- | ---------------------- | -------- | ----------- |
| `DUFF_FIN_REA_BOX`                   | DUFF_REA_BOX       | 2   | box         | box                    | [0, 6]   | reachable   |
| `DUFF_FIN_REA_BALL`                  | DUFF_REA_CVX_LS    | 2   | ball        | ball                   | [0, 6]   | reachable   |
| `DUFF_FIN_REA_NONCONVEX_LEVEL_SET`   | DUFF_REA_NCVX_LS   | 2   | level set   | level set (non-convex) | [0, 6]   | reachable   |
| `DUFF_FIN_REA_DISJOINT_BOX`          | DUFF_REA_DSJT_BOX  | 2   | box         | two disjoint boxes     | [0, 6]   | reachable   |
| `DUFF_FIN_UNR_BOX`                   | DUFF_UNR_BOX       | 2   | box         | box                    | [0, 20]  | unreachable |
| `DUFF_FIN_UNR_BALL`                  | DUFF_UNR_CVX_LS    | 2   | ball        | ball                   | [0, 6]   | unreachable |
| `NL_SLC_FIN_REA_BOX`                 | NL-SLC_REA_BOX     | 2   | box         | box                    | [0, 15]  | reachable   |
| `NL_SLC_FIN_UNR_BOX`                 | NL-SLC_UNR_BOX     | 2   | box         | box                    | [0, 15]  | unreachable |
| `NL_EIG_FIN_BACKWARD_REA_BOX`        | NL-EIG_BWD_REA_BOX | 2   | box         | box (backward problem) | [0, 1.3] | reachable   |
| `NL_EIG_FIN_BACKWARD_UNR_BOX`        | NL-EIG_BWD_UNR_BOX | 2   | box         | box (backward problem) | [0, 1.3] | unreachable |
| `CP_LQR_FIN_REA_CONVERGE`            | CP-LQR_REA_CVG     | 4   | point       | small box at the origin | [0, 12] | reachable   |
| `CP_LQR_FIN_REA_UNSAFE`              | CP-LQR_REA_USF     | 4   | point       | wall p + l sin θ > 1.3 | [0, 10]  | reachable   |
| `CP_LQR_FIN_UNR_SAFE`                | CP-LQR_UNR_SAF     | 4   | point       | wall p + l sin θ > 3.7 | [0, 10]  | unreachable |
| `MAS_CON_FIN_REA_BOX`                | MAS-CON_REA_BOX    | 16  | box         | box                    | [0, 7]   | reachable   |
| `MAS_CON_FIN_UNR_SAFE_0`             | MAS-CON_UNR_SAF_0  | 16  | box         | box                    | [0, 7]   | unreachable |
| `MAS_CON_FIN_UNR_SAFE_1`             | MAS-CON_UNR_SAF_1  | 16  | box         | box                    | [0, 7]   | unreachable |

How to read the table:

- **Expected result.** "Reachable" means that some trajectory from the initial set enters the target within the horizon. "Unreachable" means that none does.
- **Simulation baseline.** `scripts.check_benchmarks` confirms every expected result by simulation. It also records the simulated in-target times, which are used to judge the reach-time bounds.
- **Cart-pole sets.** The cart-pole initial set is a single state. Its wall targets are restricted to the box X = [−0.06, 4] × [−0.88, 0.7] × [−0.83, 0.66] × [−1.72, 2.82].

## Results

Table II of the paper, from `results/summary/` (`python -m scripts.paper_table`).

Times are in seconds. They count each tool's core computation: everything needed for the verdict, but not JIT compilation, MATLAB start-up, CORA derivative generation or container start-up. KRTB's time has two parts:

- the principal eigenpairs ("analytic" when the eigenfunctions are known in closed form);
- the reach-time bounds.

| System                         | Benchmark            | Expected | KRTB eigenpairs | KRTB bounds | CORA         | JuliaReach   | dReach          | HJ            | SOSTOOLS    |
| ------------------------------ | -------------------- | -------- | --------------- | ----------- | ------------ | ------------ | --------------- | ------------- | ----------- |
| Duffing oscillator (Example 1) | DUFF_REA_BOX         | Rea      | 4.81            | Rea (0.01)  | Ukn (9.19)   | Ukn (1.27)   | δ-Rea (446.86)  | Rea (19.81)   | Ukn (1.70)  |
|                                | DUFF_REA_CVX_LS      | Rea      | 4.90            | Rea (0.01)  | N/A          | N/A          | δ-Rea (14.31)   | Rea (21.61)   | Ukn (1.43)  |
|                                | DUFF_REA_NCVX_LS     | Rea      | 4.78            | Rea (0.01)  | N/A          | N/A          | δ-Rea (1589.39) | Rea (16.19)   | N/A         |
|                                | DUFF_REA_DSJT_BOX    | Rea      | 7.37            | Rea (0.01)  | N/A          | N/A          | δ-Rea (49.06)   | Rea (17.60)   | Ukn (1.56)  |
|                                | DUFF_UNR_BOX         | Unr      | 4.87            | Inc (0.01)  | Ukn (9.97)   | Ukn (1.40)   | T/O             | Unr (65.02)   | Unr (1.33)  |
|                                | DUFF_UNR_CVX_LS      | Unr      | 4.84            | Inc (0.01)  | N/A          | N/A          | Unr (1725.68)   | Unr (22.03)   | Unr (1.24)  |
| Stable limit cycle             | NL-SLC_REA_BOX       | Rea      | analytic        | PRea (0.01) | S/E          | Ukn (2.51)   | T/O             | Rea (268.87)  | N/A         |
|                                | NL-SLC_UNR_BOX       | Unr      | analytic        | Unr (0.01)  | S/E          | Unr (2.48)   | T/O             | Unr (267.15)  | N/A         |
| Saddle point system            | NL-EIG_BWD_REA_BOX   | Rea      | analytic        | Rea (0.01)  | N/A          | N/A          | T/O             | Rea (6.95)    | N/A         |
|                                | NL-EIG_BWD_UNR_BOX   | Unr      | analytic        | Unr (0.01)  | N/A          | N/A          | T/O             | Unr (7.87)    | N/A         |
| Cart-pole (Example 2)          | CP-LQR_REA_CVG       | Rea      | 6.82            | Rea (0.02)  | Rea (14.71)  | Rea (14.85)  | δ-Rea (9.46)    | OOM           | N/A         |
|                                | CP-LQR_REA_USF       | Rea      | 55.91           | Rea (0.01)  | N/A          | N/A          | δ-Rea (2.95)    | OOM           | N/A         |
|                                | CP-LQR_UNR_SAF       | Unr      | 54.49           | Unr (0.01)  | N/A          | N/A          | Unr (0.55)      | OOM           | N/A         |
| Multi-agent consensus (Ex. 3)  | MAS-CON_REA_BOX      | Rea      | 56.93           | Rea (0.01)  | Rea (52.18)  | Rea (47.37)  | T/O             | OOM           | N/A         |
|                                | MAS-CON_UNR_SAF_0    | Unr      | 60.40           | Unr (0.01)  | Unr (51.76)  | Unr (43.81)  | Unr (8.21)      | OOM           | N/A         |
|                                | MAS-CON_UNR_SAF_1    | Unr      | 60.45           | Unr (0.01)  | Unr (52.18)  | Unr (43.32)  | Unr (8.41)      | OOM           | N/A         |

Labels:

- **Verdicts.** Rea: reachable. PRea: periodically reachable (several reach-time windows for one target). Unr: unreachable. Ukn: unknown. Inc: inconclusive (KRTB bounds within the horizon, but no simulated trajectory reaches the target inside them). δ-Rea: dReach's δ-sat.
- **No verdict.** N/A: not applicable (the tool does not handle the benchmark itself). S/E: set explosion. OOM: out of memory (8 GB). T/O: time-out (1800 s).

The tools ran on a computer with an Apple M5 CPU and 32 GB of RAM; dReach runs in a Docker container under x86-64 emulation. What each verdict proves differs between tools:

- **CORA, JuliaReach and dReach.** An "Unr" is a proof. CORA's and JuliaReach's "Rea" is also a proof (a reachable set lies inside the target).
- **SOSTOOLS.** An "Unr" is a barrier certificate that has been checked on dense samples.
- **KRTB and HJ.** Their verdicts rest on samples of the sets. KRTB's "Rea" comes with a simulated trajectory inside the bounds.

`results/summary/table.md` lists every verdict with its full timing breakdown. The online report shows the results benchmark by benchmark, next to the simulation.

## Tool settings

Every tool runs in one setting, `setting_0` in `configs/settings/<tool>.json`. Each setting comes with a description of its parameters. Every tool is used with its native capabilities only: a runner does not split sets or translate a problem for a tool. A benchmark outside a tool's scope is reported as N/A, with the reason given in its result record.

- **KRTB.** The principal eigenfunctions come from one of two sources:
  - analytic expressions, for NL-SLC and NL-EIG;
  - path integrals, around the stable equilibrium, with integration time T = 100 (Duffing), 50 (cart-pole) or 70 (multi-agent).

  The eigenfunctions are evaluated on samples of every set:
  - Default: 1000 uniform interior points and 1000 boundary points.
  - Cart-pole targets: 5000–10000 interior points, 2000 boundary points and 20000 points towards the vertices of the bounding box.
  - Multi-agent sets: 100 interior points, 100 boundary points and 2000 vertex points.

  With these counts the sampled extrema have converged. The cart-pole wall targets are restricted to a box inside the basin of attraction of the closed loop, where the eigenfunctions are defined; `python -m scripts.cartpole_lqr --basin` checks this.
- **CORA v2026.1.0.** Uses `nonlinearSys` with the conservative linearisation algorithm `lin` and one initial zonotope. The parameters are taylorTerms 4, zonotopeOrder 50, tensorOrder 2 and timeStep 0.01, with two exceptions:
  - DUFF_FIN_REA_BOX: timeStep 0.005;
  - cart-pole: taylorTerms 3 and timeStep 0.05.

  CORA is applied to forward problems from one box to one box.
- **JuliaReach.** ReachabilityAnalysis.jl 0.33.0 with TMJets and one initial box. The parameters are:
  - 2D benchmarks: abstol 1e-12 and orderT 7;
  - cart-pole and multi-agent: abstol 1e-10 and orderT 5;
  - all benchmarks: orderQ 1 and maxsteps 20000.

  JuliaReach is applied to forward problems from one box to one box.
- **dReach.** dReal3 at commit 7f31797b, with one mode and no jumps (`-k 0`). The precision δ is 1e-3, except 1e-5 for CP_LQR_FIN_REA_CONVERGE.
- **hj_reachability 0.7.0.** Computes the backward reachable tube on a uniform grid:
  - 2D benchmarks: 500 nodes per dimension, accuracy `medium`;
  - 4D and 16D benchmarks: 100 nodes per dimension, accuracy `low`.

  Each computation runs in its own process with a memory limit of 8 GB.
- **SOSTOOLS v4.03 with MOSEK.** Searches for a barrier certificate of degree 16, with B ≥ 1 on the boundary of the domain. It applies to polynomial dynamics and sets only.

The time limit is 1800 s for every tool.

## Tests

```sh
python -m pytest -m "not slow and not matlab and not julia and not docker"   # fast tests, Python only
python -m pytest                                                            # everything (needs MATLAB, Julia and Docker)
```

The tests cover the following:

- **Configs.** They check that every config loads, that its sets lie in the domain and that its expected result and direction match its name. They also check that the cart-pole configs carry exactly the model and the gain of `scripts/cartpole_lqr.py`.
- **Tools.** They check the set handling, the simulation and the KRTB bounds, and the verdict logic of every runner.
- **Tables and figures.** They check that the tables and figures follow from the stored results.
