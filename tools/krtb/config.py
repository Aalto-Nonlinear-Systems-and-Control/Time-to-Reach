import json


def load_benchmark_config(config_path):
    """Load benchmark configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
