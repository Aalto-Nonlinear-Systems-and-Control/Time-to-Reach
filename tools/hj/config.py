"""
Configuration management for HJ reachability analysis.

This module provides functions for loading and validating benchmark configurations.
"""

import json
import os
from typing import Any, Dict


def load_benchmark_config(config_path: str) -> Dict[str, Any]:
    """
    Load benchmark configuration from JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Dictionary containing the benchmark configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Validate required fields
        required_fields = ["system", "initial_sets", "domain_bounds", "verification"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in config")

        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
