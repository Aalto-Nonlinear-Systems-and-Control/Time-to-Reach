"""
Dynamics class creation for HJ reachability analysis.

This module provides functions for creating HJ dynamics classes from system configurations.
"""

from typing import Any, Dict, List

import numpy as np
import sympy as sp

# Optional imports for HJ reachability library
try:
    import hj_reachability as hj
    import jax.numpy as jnp

    HJ_AVAILABLE = True
except ImportError:
    HJ_AVAILABLE = False
    print(
        "Warning: hj_reachability library not available. Some features may be limited."
    )


def create_hj_dynamics_class(system_config: Dict[str, Any]):
    """
    Create a dynamics class for HJ_Reachability that follows the library pattern.

    Args:
        system_config: System configuration from JSON

    Returns:
        Dynamics class instance compatible with hj_reachability
    """
    if not HJ_AVAILABLE:
        raise ImportError("hj_reachability library is required for this function")

    system_type = system_config.get("type", "unknown")

    if system_type == "linear":
        # Linear system: dx/dt = A*x
        if "A" not in system_config:
            raise ValueError("Linear system requires 'A' matrix")

        A = np.array(system_config["A"])
        n_dim = A.shape[0]

        class LinearDynamics(hj.ControlAndDisturbanceAffineDynamics):
            def __init__(
                self,
                control_mode="min",
                disturbance_mode="max",
                control_space=None,
                disturbance_space=None,
            ):
                # Default to no control/disturbance for basic linear system
                if control_space is None:
                    control_space = hj.sets.Box(jnp.array([0.0]), jnp.array([0.0]))
                if disturbance_space is None:
                    disturbance_space = hj.sets.Box(jnp.array([0.0]), jnp.array([0.0]))

                super().__init__(
                    control_mode, disturbance_mode, control_space, disturbance_space
                )

            def open_loop_dynamics(self, state, time):
                return A @ state

            def control_jacobian(self, state, time):
                return jnp.zeros((n_dim, 1))

            def disturbance_jacobian(self, state, time):
                return jnp.zeros((n_dim, 1))

        return LinearDynamics()

    elif system_type == "nonlinear":
        # Nonlinear system: parse symbolic equations
        if (
            "dynamics" not in system_config
            or "equations" not in system_config["dynamics"]
        ):
            raise ValueError("Nonlinear system requires 'dynamics.equations'")

        equations = system_config["dynamics"]["equations"]

        # Parse equations and create dynamics class
        return _create_nonlinear_dynamics_class(equations)

    else:
        raise ValueError(f"Unsupported system type: {system_type}")


def _create_nonlinear_dynamics_class(equations: List[str]):
    """
    Create a nonlinear dynamics class from equation strings following HJ library pattern.

    Args:
        equations: List of equation strings

    Returns:
        Dynamics class instance
    """
    if not HJ_AVAILABLE:
        raise ImportError("hj_reachability library is required for this function")

    n_dim = len(equations)

    # Create symbolic variables and parse equations
    x_symbols = [sp.Symbol(f"x{i+1}") for i in range(n_dim)]
    parsed_equations = []

    for eq_str in equations:
        try:
            eq_str_python = eq_str.replace("^", "**")
            eq_sympy = sp.sympify(eq_str_python)
            parsed_equations.append(eq_sympy)
        except Exception as e:
            raise ValueError(f"Failed to parse equation '{eq_str}': {e}")

    # Create numerical function using JAX-compatible modules
    import jax.numpy as jnp

    jax_modules = {
        "sin": jnp.sin,
        "cos": jnp.cos,
        "tan": jnp.tan,
        "exp": jnp.exp,
        "log": jnp.log,
        "sqrt": jnp.sqrt,
        "abs": jnp.abs,
    }
    dynamics_func = sp.lambdify(
        x_symbols, parsed_equations, modules=[jax_modules, "jax"]
    )

    class NonlinearDynamics(hj.ControlAndDisturbanceAffineDynamics):
        def __init__(
            self,
            control_mode="min",
            disturbance_mode="max",
            control_space=None,
            disturbance_space=None,
        ):
            # Default to no control/disturbance for autonomous systems
            if control_space is None:
                control_space = hj.sets.Box(jnp.array([0.0]), jnp.array([0.0]))
            if disturbance_space is None:
                disturbance_space = hj.sets.Box(jnp.array([0.0]), jnp.array([0.0]))

            super().__init__(
                control_mode, disturbance_mode, control_space, disturbance_space
            )

        def open_loop_dynamics(self, state, time):
            """Evaluate the dynamics equations at the given state."""
            try:
                result = dynamics_func(*state)
                if n_dim == 1:
                    return jnp.array([result])
                elif isinstance(result, (list, tuple)):
                    return jnp.array(result)
                else:
                    return jnp.array([result])
            except Exception as e:
                # Re-raise with more context
                raise RuntimeError(f"Error evaluating dynamics at state {state}: {e}")

        def control_jacobian(self, state, time):
            """Jacobian with respect to control (zero for autonomous systems)."""
            return jnp.zeros((n_dim, 1))

        def disturbance_jacobian(self, state, time):
            """Jacobian with respect to disturbance (zero for autonomous systems)."""
            return jnp.zeros((n_dim, 1))

    return NonlinearDynamics()


def create_hj_system(system_config: Dict[str, Any]) -> Any:
    """
    Create a system dynamics class for HJ reachability analysis.

    Args:
        system_config: System configuration from JSON

    Returns:
        HJ dynamics class instance
    """
    return create_hj_dynamics_class(system_config)
