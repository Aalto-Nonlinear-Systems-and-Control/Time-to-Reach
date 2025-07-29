"""
Set creation functions for HJ reachability analysis.

This module provides functions for creating initial and target sets from configurations.
"""

import re
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import hj_reachability as hj
    import jax.numpy as jnp

    HJ_AVAILABLE = True
except ImportError:
    HJ_AVAILABLE = False


def detect_ball_from_level_set(function_str: str) -> Optional[Dict[str, Any]]:
    """
    Detect if a level set function defines a ball (circle in 2D).

    Expected format: (x1-center_x)^2+(x2-center_y)^2-radius^2 = 0
    or variations like: (x1+0.75)^2+(x2-1.78)^2-0.01

    Args:
        function_str: Level set function string

    Returns:
        Dictionary with center and radius if it's a ball, None otherwise
    """
    # Normalize the function string
    func = function_str.strip()

    # Pattern to match ball/circle equations
    # Matches: (x1±a)^2+(x2±b)^2-r^2 or (x1±a)^2+(x2±b)^2-r
    pattern = r"\(\s*x1\s*([+-])\s*([0-9.]+)\s*\)\s*\^\s*2\s*\+\s*\(\s*x2\s*([+-])\s*([0-9.]+)\s*\)\s*\^\s*2\s*-\s*([0-9.]+)"

    match = re.search(pattern, func)
    if not match:
        return None

    # Extract components
    x1_sign, x1_offset, x2_sign, x2_offset, radius_squared = match.groups()

    # Calculate center coordinates
    center_x = -float(x1_offset) if x1_sign == "+" else float(x1_offset)
    center_y = -float(x2_offset) if x2_sign == "+" else float(x2_offset)

    # Calculate radius
    radius_squared_value = float(radius_squared)
    if radius_squared_value <= 0:
        return None

    radius = np.sqrt(radius_squared_value)

    return {"center": [center_x, center_y], "radius": radius, "type": "ball"}


def create_hj_ball_set(center: list, radius: float) -> Any:
    """
    Create HJ Ball set object.

    Args:
        center: Ball center coordinates
        radius: Ball radius

    Returns:
        HJ Ball set object
    """
    if not HJ_AVAILABLE:
        raise ImportError("hj_reachability library is required for Ball sets")

    center_jnp = jnp.array(center)
    return hj.sets.Ball(center_jnp, radius)


def create_initial_set(initial_set_config: Dict[str, Any]) -> Union[np.ndarray, Any]:
    """
    Create initial set from configuration.

    Args:
        initial_set_config: Initial set configuration

    Returns:
        Array representing the initial set bounds (for box) or HJ Ball object (for ball)
    """
    if initial_set_config["type"] == "box":
        bounds = initial_set_config["bounds"]
        return np.array(bounds)
    elif initial_set_config["type"] == "level_set":
        # Check if the level set defines a ball
        function_str = initial_set_config["function"]
        ball_info = detect_ball_from_level_set(function_str)

        if ball_info is not None:
            # It's a ball, create HJ Ball object
            return {
                "type": "ball",
                "hj_set": create_hj_ball_set(ball_info["center"], ball_info["radius"]),
                "center": ball_info["center"],
                "radius": ball_info["radius"],
            }
        else:
            raise ValueError(
                f"Level set '{function_str}' does not define a supported ball format"
            )
    else:
        raise ValueError(f"Unsupported initial set type: {initial_set_config['type']}")


def create_target_set(target_set_config) -> Union[np.ndarray, Any]:
    """
    Create target set from configuration.

    Can handle either a single target set or a list of target sets for disjoint unions.

    Args:
        target_set_config: Target set configuration (single dict or list of dicts)

    Returns:
        Array representing the target set bounds (for box) or HJ Ball object (for ball),
        or dict with type 'union' for multiple target sets
    """
    # Handle single target set (backward compatibility)
    if isinstance(target_set_config, dict):
        if target_set_config["type"] == "box":
            bounds = target_set_config["bounds"]
            return np.array(bounds)
        elif target_set_config["type"] == "level_set":
            # Check if the level set defines a ball
            function_str = target_set_config["function"]
            ball_info = detect_ball_from_level_set(function_str)

            if ball_info is not None:
                # It's a ball, create HJ Ball object
                return {
                    "type": "ball",
                    "hj_set": create_hj_ball_set(
                        ball_info["center"], ball_info["radius"]
                    ),
                    "center": ball_info["center"],
                    "radius": ball_info["radius"],
                }
            else:
                raise ValueError(
                    f"Level set '{function_str}' does not define a supported ball format"
                )
        else:
            raise ValueError(
                f"Unsupported target set type: {target_set_config['type']}"
            )

    # Handle multiple target sets (union)
    elif isinstance(target_set_config, list):
        if len(target_set_config) == 1:
            # Single target set in list, process normally
            return create_target_set(target_set_config[0])

        # Multiple target sets - create union
        target_sets = []
        for config in target_set_config:
            target_sets.append(create_target_set(config))

        return {"type": "union", "sets": target_sets}

    else:
        raise ValueError(
            f"Unsupported target set configuration type: {type(target_set_config)}"
        )


def get_set_bounds(set_obj: Union[np.ndarray, Dict[str, Any]]) -> np.ndarray:
    """
    Get bounding box for a set object.

    Args:
        set_obj: Set object (box bounds array, ball dict, or union dict)

    Returns:
        Array representing the bounding box
    """
    if isinstance(set_obj, np.ndarray):
        # It's a box, return as is
        return set_obj
    elif isinstance(set_obj, dict):
        if set_obj.get("type") == "ball":
            # It's a ball, compute bounding box
            center = np.array(set_obj["center"])
            radius = set_obj["radius"]

            bounds = []
            for i in range(len(center)):
                bounds.append([center[i] - radius, center[i] + radius])

            return np.array(bounds)
        elif set_obj.get("type") == "union":
            # It's a union of sets, compute overall bounding box
            all_bounds = []
            for sub_set in set_obj["sets"]:
                sub_bounds = get_set_bounds(sub_set)
                all_bounds.append(sub_bounds)

            # Compute union bounding box
            all_bounds = np.array(all_bounds)  # Shape: (n_sets, n_dims, 2)

            union_bounds = []
            n_dims = all_bounds.shape[1]
            for dim in range(n_dims):
                # For each dimension, take min of all lower bounds and max of all upper bounds
                min_bound = np.min(all_bounds[:, dim, 0])
                max_bound = np.max(all_bounds[:, dim, 1])
                union_bounds.append([min_bound, max_bound])

            return np.array(union_bounds)
        else:
            raise ValueError(f"Unsupported set object type: {set_obj.get('type')}")
    else:
        raise ValueError(f"Unsupported set object type: {type(set_obj)}")
