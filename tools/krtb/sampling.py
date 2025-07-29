import math

import numpy as np


def sample_box_set(bounds, num_samples, random_seed=None):
    """Sample points uniformly from a box set."""
    np.random.seed(random_seed)  # For reproducibility

    dimension = len(bounds)
    samples = np.zeros((num_samples, dimension))

    for i in range(dimension):
        x_min, x_max = bounds[i]
        samples[:, i] = np.random.uniform(x_min, x_max, num_samples)

    return samples


def sample_level_set(level_set_def, num_samples, random_seed=None):
    """Sample points from a level set defined by f(x) <= 0."""
    np.random.seed(random_seed)  # For reproducibility

    func_str = level_set_def["function"].replace("^", "**")  # Fix for Python
    domain = level_set_def.get("domain", [[-3, 3], [-3, 3]])
    dimension = len(domain)

    allowed_funcs = {
        k: getattr(math, k)
        for k in ["exp", "sin", "cos", "tan", "log", "sqrt", "fabs", "pow"]
    }

    # Create level set function
    def level_set_func(*args):
        var_dict = {f"x{i+1}": args[i] for i in range(dimension)}
        try:
            return eval(func_str, allowed_funcs, var_dict)
        except Exception:
            return np.inf  # Return infinity for invalid points

    # Sample points and filter those in level set
    samples = []
    attempts = 0
    max_attempts = num_samples * 100  # Increased max attempts

    while len(samples) < num_samples and attempts < max_attempts:
        # Sample point in domain
        point = []
        for i in range(dimension):
            x_min, x_max = domain[i]
            point.append(np.random.uniform(x_min, x_max))

        # Check if point is in level set
        try:
            value = level_set_func(*point)
            if value <= 0 and not np.isinf(value):
                samples.append(point)
        except:
            pass  # Skip invalid points

        attempts += 1

    if len(samples) == 0:
        print(
            f"Warning: No valid samples found for level set after {attempts} attempts"
        )
        print(f"Function: {func_str}")
        print(f"Domain: {domain}")
        # Diagnostic: print min/max value of the function over a grid
        try:
            grid_x = np.linspace(domain[0][0], domain[0][1], 50)
            grid_y = np.linspace(domain[1][0], domain[1][1], 50)
            vals = np.array([level_set_func(x, y) for x in grid_x for y in grid_y])
            print(f"Level set function min: {vals.min()}, max: {vals.max()}")
        except Exception as e:
            print(f"Could not evaluate grid diagnostics: {e}")
        return np.array([]).reshape(0, dimension)

    print(f"Level set sampling: {len(samples)} valid samples from {attempts} attempts")
    return np.array(samples)


def point_in_set(point, set_def):
    """Check if a point is inside a given set (box or level_set)."""
    if set_def["type"] == "box":
        for i, (p, bound) in enumerate(zip(point, set_def["bounds"])):
            if not (bound[0] <= p <= bound[1]):
                return False
        return True

    elif set_def["type"] == "level_set":
        func_str = set_def["function"].replace("^", "**")
        dimension = len(point)

        allowed_funcs = {
            k: getattr(math, k)
            for k in ["exp", "sin", "cos", "tan", "log", "sqrt", "fabs", "pow"]
        }

        var_dict = {f"x{i+1}": point[i] for i in range(dimension)}

        try:
            return eval(func_str, allowed_funcs, var_dict) <= 0
        except Exception:
            return False

    else:
        raise ValueError(f"Unknown set type: {set_def['type']}")


def sample_sets(sets, num_samples, random_seed=None):
    """Sample points from a list of sets."""
    all_points = []

    for set_def in sets:
        if set_def["type"] == "box":
            points = sample_box_set(set_def["bounds"], num_samples, random_seed)
        elif set_def["type"] == "level_set":
            points = sample_level_set(set_def, num_samples, random_seed)
        else:
            raise ValueError(f"Unknown set type: {set_def['type']}")

        if len(points) > 0:
            all_points.append(points)

    return np.vstack(all_points) if all_points else np.array([])
