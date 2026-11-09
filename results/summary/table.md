| benchmark | expected | KRTB (setting_0) | HJ (setting_0) | JuliaReach (setting_0) | CORA (setting_0) | SOSBC (setting_0) | dReach (setting_0) |
|---|---|---|---|---|---|---|---|
| CP_LQR_FIN_REA_CONVERGE | reachable | REA witness (eigenpairs 6.82 s + bounds 0.0199 s + witness 0.0199 s) | out of memory | REA proven (14.9 s) | REA proven (14.7 s) | n/a | REA possible (9.46 s) |
| CP_LQR_FIN_REA_UNSAFE | reachable | REA witness (eigenpairs 55.9 s + bounds 0.0134 s + witness 0.00686 s) | out of memory | n/a | n/a | n/a | REA possible (2.95 s) |
| CP_LQR_FIN_UNR_SAFE | unreachable | UNR est. (eigenpairs 54.5 s + bounds 0.00716 s + witness 1.54e-06 s) | out of memory | n/a | n/a | n/a | UNR proven (0.55 s) |
| DUFF_FIN_REA_BALL | reachable | REA witness (eigenpairs 4.9 s + bounds 0.00881 s + witness 1.56 s) | REA est. (21.6 s) | n/a | n/a | unknown (1.43 s) | REA possible (14.3 s) |
| DUFF_FIN_REA_BOX | reachable | REA witness (eigenpairs 4.81 s + bounds 0.00889 s + witness 2.63 s) | REA est. (19.8 s) | REA possible (1.27 s) | REA possible (9.19 s) | unknown (1.7 s) | REA possible (447 s) |
| DUFF_FIN_REA_DISJOINT_BOX | reachable | REA witness (eigenpairs 7.37 s + bounds 0.00984 s + witness 0.00465 s) | REA est. (17.6 s) | n/a | n/a | unknown (1.56 s) | REA possible (49.1 s) |
| DUFF_FIN_REA_NONCONVEX_LEVEL_SET | reachable | REA witness (eigenpairs 4.78 s + bounds 0.0083 s + witness 0.00695 s) | REA est. (16.2 s) | n/a | n/a | n/a | REA possible (1.59e+03 s) |
| DUFF_FIN_UNR_BALL | unreachable | unknown (eigenpairs 4.84 s + bounds 0.00867 s + witness 0.226 s) | UNR est. (22 s) | n/a | n/a | UNR proven (1.24 s) | UNR proven (1.73e+03 s) |
| DUFF_FIN_UNR_BOX | unreachable | unknown (eigenpairs 4.87 s + bounds 0.00858 s + witness 0.347 s) | UNR est. (65 s) | REA possible (1.4 s) | REA possible (9.97 s) | UNR proven (1.33 s) | timeout (1.8e+03 s) |
| MAS_CON_FIN_REA_BOX | reachable | REA witness (eigenpairs 56.9 s + bounds 0.0123 s + witness 0.0291 s) | out of memory | REA proven (47.4 s) | REA proven (52.2 s) | n/a | timeout (1.8e+03 s) |
| MAS_CON_FIN_UNR_SAFE_0 | unreachable | UNR est. (eigenpairs 60.4 s + bounds 0.0108 s + witness 6.25e-07 s) | out of memory | UNR proven (43.8 s) | UNR proven (51.8 s) | n/a | UNR proven (8.21 s) |
| MAS_CON_FIN_UNR_SAFE_1 | unreachable | UNR est. (eigenpairs 60.4 s + bounds 0.00848 s + witness 4.16e-07 s) | out of memory | UNR proven (43.3 s) | UNR proven (52.2 s) | n/a | UNR proven (8.41 s) |
| NL_EIG_FIN_BACKWARD_REA_BOX | reachable | REA witness (eigenpairs 0.00585 s + bounds 0.00636 s + witness 0.234 s) | REA est. (6.95 s) | n/a | n/a | n/a | timeout (1.8e+03 s) |
| NL_EIG_FIN_BACKWARD_UNR_BOX | unreachable | UNR est. (eigenpairs 0.00438 s + bounds 0.0065 s + witness 7.5e-07 s) | UNR est. (7.87 s) | n/a | n/a | n/a | timeout (1.8e+03 s) |
| NL_SLC_FIN_REA_BOX | reachable | REA witness (eigenpairs 0.0149 s + bounds 0.00898 s + witness 0.00655 s) | REA est. (269 s) | REA possible (2.51 s) | set explosion (3.15 s) | n/a | timeout (1.8e+03 s) |
| NL_SLC_FIN_UNR_BOX | unreachable | UNR est. (eigenpairs 0.00552 s + bounds 0.00905 s + witness 5e-07 s) | UNR est. (267 s) | UNR proven (2.48 s) | set explosion (3.11 s) | n/a | timeout (1.8e+03 s) |

| tool | UNR proven | REA proven | REA witness | REA possible | REA est. | UNR est. | unknown | n/a | set explosion | out of memory | timeout | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KRTB (setting_0) | 0 | 0 | 9 | 0 | 0 | 5 | 2 | 0 | 0 | 0 | 0 | 0 |
| HJ (setting_0) | 0 | 0 | 0 | 0 | 6 | 4 | 0 | 0 | 0 | 6 | 0 | 0 |
| JuliaReach (setting_0) | 3 | 2 | 0 | 3 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| CORA (setting_0) | 2 | 2 | 0 | 2 | 0 | 0 | 0 | 8 | 2 | 0 | 0 | 0 |
| SOSBC (setting_0) | 2 | 0 | 0 | 0 | 0 | 0 | 3 | 11 | 0 | 0 | 0 | 0 |
| dReach (setting_0) | 4 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |

Times are each tool's core computation in seconds: everything needed for the verdict, excluding
JIT compilation, MATLAB start-up, CORA derivative generation and container start-up. KRTB's is
split into the principal eigenpairs (eigenvalues and eigenfunction values on the samples), the
reach-time bounds and the witness simulation. Measured on: macOS-27.0.1-arm64-arm-64bit; macOS-27.2-arm64-arm-64bit.
