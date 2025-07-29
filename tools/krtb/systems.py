import numpy as np
import sympy as sp
from sympy import Matrix, lambdify, symbols


def create_system_from_config(system_config):
    """Create system dynamics from configuration."""
    system_type = system_config["type"]

    if system_type == "linear":
        # Linear system: dot{x} = Ax
        if "matrix" in system_config["dynamics"]:
            A = np.array(system_config["dynamics"]["matrix"])
        elif "matrix_file" in system_config["dynamics"]:
            A = np.loadtxt(system_config["dynamics"]["matrix_file"])
        else:
            raise ValueError("Linear system must specify 'matrix' or 'matrix_file'")

        return LinearSystem(A)

    elif system_type == "nonlinear":
        equations = system_config["dynamics"]["equations"]
        return NonlinearSystem(equations)

    else:
        raise ValueError(f"Unknown system type: {system_type}")


class LinearSystem:
    """Linear system wrapper for KRTB compatibility."""

    def __init__(self, A):
        self.A = A
        self.dimension = A.shape[0]
        self.dim = self.dimension  # For compatibility

        # Create symbolic variables and expressions
        self.x = symbols([f"x{i+1}" for i in range(self.dimension)])

        # Create symbolic dynamics: f = A * x
        x_vector = Matrix(self.x)
        A_matrix = Matrix(A)
        self.f = A_matrix * x_vector

    def ff(self, *args):
        """System dynamics function: dot{x} = Ax"""
        if len(args) == 1:
            x = args[0]
        else:
            x = np.array(args)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        return self.A @ x


class NonlinearSystem:
    """Nonlinear system wrapper for KRTB compatibility."""

    def __init__(self, equations):
        self.equations = equations
        self.dimension = len(equations)
        self.dim = self.dimension  # For compatibility

        # Create symbolic variables
        self.x = symbols([f"x{i+1}" for i in range(self.dimension)])

        # Create symbolic dynamics
        self.f = Matrix([sp.sympify(eq) for eq in equations])

    def ff(self, *args):
        """System dynamics function from equations"""
        if len(args) == 1:
            x = args[0]
        else:
            x = np.array(args)

        # Create numerical function from symbolic expressions
        f_lambdified = lambdify(self.x, self.f, "numpy")

        if x.ndim == 1:
            # Single point
            result = f_lambdified(*x)
        else:
            # Multiple points - need to evaluate point by point
            if x.shape[0] == self.dimension:
                # x is (dim, num_points)
                result = np.array([f_lambdified(*x[:, i]) for i in range(x.shape[1])])
                result = result.T  # Make it (dim, num_points)
            else:
                # x is (num_points, dim)
                result = np.array([f_lambdified(*x[i, :]) for i in range(x.shape[0])])
                result = result.T  # Make it (dim, num_points)

        return np.array(result).reshape(-1, 1) if result.ndim == 1 else result
