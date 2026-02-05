# Time-to-Reach verification through Koopman Operator spectrum

This repository contains a comprehensive collection of benchmarks for time-to-reach analysis in dynamical systems. The benchmarks cover various types of reachability analysis problems and are primarily used for comparing the performance and accuracy of different reachability analysis approaches.

## Table of Contents

- [Time-to-Reach verification through Koopman Operator spectrum](#time-to-reach-verification-through-koopman-operator-spectrum)
  - [Table of Contents](#table-of-contents)
  - [Benchmark Settings Summary](#benchmark-settings-summary)
    - [Quick Reference Table](#quick-reference-table)
    - [Tool-Specific Settings](#tool-specific-settings)
      - [Hamilton-Jacobi Reachability (HJ)](#hamilton-jacobi-reachability-hj)
      - [Koopman Reach-Time Bounds (KRTB)](#koopman-reach-time-bounds-krtb)
      - [CORA (Zonotope Reachability)](#cora-zonotope-reachability)
      - [JuliaReach (Taylor Model Reachability)](#juliareach-taylor-model-reachability)
      - [Sum-of-Squares Barrier Certificates (SOSBC)](#sum-of-squares-barrier-certificates-sosbc)
      - [dReach (δ-Complete SMT Solver)](#dreach-δ-complete-smt-solver)
    - [General Notes](#general-notes)
  - [Documentation](#documentation)

## Benchmark Settings Summary

### Quick Reference Table

| Benchmark                            | Dim | Initial/Target Sets    | Time Horizon | Expected Result | Tools Available                           |
| ------------------------------------ | --- | ---------------------- | ------------ | --------------- | ----------------------------------------- |
| **CP-LQR_FIN_REA_CONVERGE**          | 4D  | Box → Box (tiny)       | [0, 10]      | Reachable       | HJ, KRTB, CORA, JuliaReach, dReach        |
| **CP-LQR_FIN_REA_UNSAFE**            | 4D  | Box → Box (large)      | [0, 10]      | Reachable       | HJ, KRTB, CORA, JuliaReach, dReach        |
| **CP-LQR_FIN_UNR_SAFE**              | 4D  | Box → Box (large)      | [0, 10]      | Reachable       | HJ, KRTB, CORA, JuliaReach, dReach        |
| **DUFF_FIN_REA_BALL**                | 2D  | Ball → Ball            | [0, 6]       | Reachable       | HJ, KRTB, CORA, SOSBC, dReach             |
| **DUFF_FIN_REA_BOX**                 | 2D  | Box → Box              | [0, 6]       | Reachable       | HJ, KRTB, CORA, JuliaReach, SOSBC, dReach |
| **DUFF_FIN_REA_DISJOINT_BOX**        | 2D  | Box → 2 Disjoint Boxes | [0, 6]       | Reachable       | HJ, KRTB, CORA, JuliaReach, SOSBC, dReach |
| **DUFF_FIN_REA_NONCONVEX_LEVEL_SET** | 2D  | Nonconvex → Nonconvex  | [0, 6]       | Reachable       | HJ, KRTB, CORA, JuliaReach, SOSBC, dReach |
| **DUFF_FIN_UNR_BALL**                | 2D  | Ball → Ball            | [0, 6]       | Unreachable     | HJ, KRTB, CORA, SOSBC, dReach             |
| **DUFF_FIN_UNR_BOX**                 | 2D  | Box → Box              | [0, 20]      | Unreachable     | HJ, KRTB, CORA, JuliaReach, SOSBC, dReach |
| **MAS-CON_FIN_REA_BOX**              | 16D | Box → Box              | [0, 10]      | Reachable       | KRTB, CORA, JuliaReach, dReach            |
| **MAS-CON_FIN_UNR_SAFE_0**           | 16D | Box → Box              | [0, 10]      | Unreachable     | KRTB, CORA, JuliaReach, dReach            |
| **MAS-CON_FIN_UNR_SAFE_1**           | 16D | Box → Box              | [0, 10]      | Unreachable     | KRTB, CORA, JuliaReach, dReach            |
| **NL-EIG_FIN_BACKWARD_REA_BOX**      | 2D  | Box → Box (backward)   | [0, 1.3]     | Reachable       | HJ, KRTB, CORA, JuliaReach, dReach        |
| **NL-EIG_FIN_BACKWARD_UNR_BOX**      | 2D  | Box → Box (backward)   | [0, 1.3]     | Unreachable     | HJ, KRTB, CORA, JuliaReach, dReach        |
| **NL-SLC_FIN_REA_BOX**               | 2D  | Box → Box              | [0, 15]      | Reachable       | HJ, KRTB, CORA, JuliaReach, dReach        |
| **NL-SLC_FIN_UNR_BOX**               | 2D  | Box → Box              | [0, 15]      | Unreachable     | HJ, KRTB, CORA, JuliaReach, dReach        |

### Tool-Specific Settings

#### Hamilton-Jacobi Reachability (HJ)

Settings for HJ reachability analysis using level set methods:

| Benchmark                        | Grid Size | Accuracy | Reachability Type | Notes                          |
| -------------------------------- | --------- | -------- | ----------------- | ------------------------------ |
| CP-LQR (all variants)            | 100       | low      | backward          | 4D system requires coarse grid |
| DUFF_FIN_REA_BALL                | 500       | medium   | backward          | Level set initial/target       |
| DUFF_FIN_REA_BOX                 | 500       | medium   | backward          | Box sets                       |
| DUFF_FIN_REA_DISJOINT_BOX        | 500       | medium   | backward          | Multiple target boxes          |
| DUFF_FIN_REA_NONCONVEX_LEVEL_SET | 200       | medium   | backward          | Nonconvex level sets           |
| DUFF_FIN_UNR_BOX                 | 500       | medium   | backward          | Unreachability verification    |
| NL-EIG (both variants)           | 500       | medium   | backward          | Backward reachability          |
| NL-SLC (both variants)           | 500       | medium   | backward          | Long time horizon              |

**HJ Parameters:**

- `grid_size`: Number of grid points per dimension (higher = more accurate, slower)
- `accuracy`: Solver precision ("low", "medium", "high", "very_high")
- `reachability_type`: "backward" (target to initial) or "forward" (initial to target)

#### Koopman Reach-Time Bounds (KRTB)

Path integral-based method using Koopman operator theory:

| Benchmark                        | Equilibrium Point | Initial Samples | Target Samples | Integration Time (T) | Notes               |
| -------------------------------- | ----------------- | --------------- | -------------- | -------------------- | ------------------- |
| CP-LQR_FIN_REA_CONVERGE          | [0,0,0,0]         | 10              | 1000           | 50                   | Origin equilibrium  |
| CP-LQR_FIN_REA_UNSAFE            | [0,0,0,0]         | 10              | 10000          | 50                   | Large target set    |
| CP-LQR_FIN_UNR_SAFE              | [0,0,0,0]         | 5               | 10000          | 50                   | Safety verification |
| DUFF_FIN_REA_BALL                | [1,0]             | 1000            | 1000           | 100                  | Ball sets           |
| DUFF_FIN_REA_BOX                 | [1,0]             | 1000            | 1000           | 100                  | Box sets            |
| DUFF_FIN_REA_DISJOINT_BOX        | [1,0]             | 1000            | 1000           | 100                  | Multiple targets    |
| DUFF_FIN_REA_NONCONVEX_LEVEL_SET | [1,0]             | 1000            | 1000           | 100                  | Nonconvex sets      |
| DUFF_FIN_UNR_BALL                | [1,0]             | 1000            | 1000           | 100                  | Unreachability      |
| DUFF_FIN_UNR_BOX                 | [1,0]             | 1000            | 1000           | 100                  | Unreachability      |
| MAS-CON_FIN_REA_BOX              | [0,...,0] (16D)   | 100             | 1000           | 70                   | High-dimensional    |
| MAS-CON_FIN_UNR_SAFE_0           | [0,...,0] (16D)   | 100             | 1000           | 70                   | Synchronization     |
| MAS-CON_FIN_UNR_SAFE_1           | [0,...,0] (16D)   | 100             | 1000           | 70                   | Synchronization     |
| NL-EIG_FIN_BACKWARD_REA_BOX      | [0,0]             | 1000            | 1000           | 100                  | Backward reach      |
| NL-EIG_FIN_BACKWARD_UNR_BOX      | [0,0]             | 1000            | 1000           | 100                  | Backward unreach    |
| NL-SLC_FIN_REA_BOX               | [0,0]             | 1000            | 2000           | -                    | Spiral dynamics     |
| NL-SLC_FIN_UNR_BOX               | [0,0]             | 1000            | 2000           | -                    | Spiral dynamics     |

**KRTB Path Integral Parameters:**

- `equilibrium`: Reference equilibrium point for linearization and principal eigenfunction computation
- `num_samples_initial`: Monte Carlo samples from initial set for path integral evaluation
- `num_samples_target`: Monte Carlo samples from target set for path integral evaluation
- `T`: Integration time horizon for path integral (larger T = more accurate eigenfunction values)
- `num_t_eval`: Number of time evaluation points (default: 150, uniformly spaced in [0, T])
- `method`: ODE solver method (default: "RK45" - Runge-Kutta 4th/5th order adaptive)

For more details on the path-integral method, see [this paper](https://ieeexplore.ieee.org/abstract/document/10384288).

#### CORA (Zonotope Reachability)

Set-based reachability using zonotope representations:

| Benchmark      | Algorithm | Taylor Terms | Zonotope Order | Time Step  | Notes                             |
| -------------- | --------- | ------------ | -------------- | ---------- | --------------------------------- |
| All benchmarks | lin       | 4            | 50             | 0.005-0.01 | Conservative linear approximation |

**CORA Parameters:**

- `alg`: Algorithm type ("lin" for linear approximation of nonlinear dynamics)
- `taylorTerms`: Order of Taylor expansion for reachability computation
- `zonotopeOrder`: Maximum zonotope order before reduction (controls complexity)
- `timeStep`: Integration time step (smaller = more accurate, slower)

#### JuliaReach (Taylor Model Reachability)

Taylor model-based flowpipe construction:

| Tool Setting  | Value  | Description                                    |
| ------------- | ------ | ---------------------------------------------- |
| Algorithm     | TMJets | Taylor model integration for nonlinear systems |
| Time Step (δ) | 0.01   | Integration time step                          |
| Variables     | (1, 2) | Output projection dimensions                   |

**JuliaReach Notes:**

- Only supports box initial/target sets (level sets are skipped)
- Uses Taylor models for rigorous overapproximation of nonlinear dynamics
- Automatically adapts to system dimensions

#### Sum-of-Squares Barrier Certificates (SOSBC)

SOS optimization for barrier certificate synthesis:

| Benchmark           | Barrier Degree (deg_B) | SOS Multiplier (deg_s) | Solver | Notes                            |
| ------------------- | ---------------------- | ---------------------- | ------ | -------------------------------- |
| DUFF_FIN_REA (most) | 8                      | 6                      | mosek  | Standard polynomial degree       |
| DUFF_FIN_UNR_BOX    | 14                     | 10                     | mosek  | Higher degree for unreachability |

**SOSBC Parameters:**

- `deg_B`: Polynomial degree of barrier certificate (higher = more expressive, slower)
- `deg_s`: Degree of SOS multipliers in constraints
- `epsilon`: Small positive constant for strict inequalities (default: 1e-6)
- `solver`: SDP solver ("mosek" recommended for numerical stability)

#### dReach (δ-Complete SMT Solver)

Bounded model checking with δ-decidability:

| Tool Setting | Value         | Description                         |
| ------------ | ------------- | ----------------------------------- |
| Docker Image | dreach:latest | Containerized dReach environment    |
| Timeout      | 300s          | Maximum verification time per query |
| δ-precision  | Auto          | Numerical precision parameter       |

**dReach Notes:**

- Uses hybrid system specifications (.drh files)
- Provides δ-complete decidability (sound up to numerical precision)
- Best for bounded-time reachability queries

### General Notes

- **All systems** are continuous-time nonlinear dynamical systems
- **Box sets** are defined by interval bounds: $[l_1, u_1] \times [l_2, u_2] \times \cdots$
- **Level sets** are defined by implicit functions: $\{x : \phi(x) \leq 0\}$
- **Equilibrium points** are used by KRTB for computing principal Koopman eigenfunctions
- **Grid resolution** and **sample counts** trade off accuracy vs computational cost
- The **HEAT3D125_FIN_REA** benchmark directory exists but lacks configuration files

## Documentation

For detailed information about each benchmark, including:

- System specifications and dynamics
- Initial conditions and target sets
- Analysis methods and tools used
- Performance results and comparisons
- Visualization

Please refer to the **`Benchmark_Specifications.pdf`** file, which provides comprehensive documentation for all benchmarks and their corresponding results.
