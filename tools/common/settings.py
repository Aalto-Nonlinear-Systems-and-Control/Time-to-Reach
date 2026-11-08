"""Per-tool settings (configs/settings/<tool>.json).

Each setting has "defaults" and per-benchmark overrides under "benchmarks". A
setting may name another one in "inherits"; its defaults and benchmark entries
are then merged over the parent's.
"""

import copy

from . import jsonio, paths


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve(all_settings, name, seen=()):
    if name in seen:
        raise ValueError(f"cyclic 'inherits' chain: {' -> '.join(seen + (name,))}")
    block = all_settings[name]
    defaults = block.get("defaults", {})
    benchmarks = block.get("benchmarks", {})
    if "inherits" in block:
        parent = _resolve(all_settings, block["inherits"], seen + (name,))
        defaults = _merge(parent["defaults"], defaults)
        benchmarks = {
            bid: _merge(parent["benchmarks"].get(bid, {}), benchmarks.get(bid, {}))
            for bid in set(parent["benchmarks"]) | set(benchmarks)
        }
    return {"defaults": defaults, "benchmarks": benchmarks}


def _setting(tool, setting):
    data = jsonio.read_json(paths.SETTINGS_DIR / f"{tool}.json")
    return _resolve(data["settings"], setting)


def setting_names(tool):
    return list(jsonio.read_json(paths.SETTINGS_DIR / f"{tool}.json")["settings"])


def resolved(tool, setting):
    """{"defaults", "benchmarks", "description", "inherits"} of a setting, with its 'inherits' chain merged."""
    block = jsonio.read_json(paths.SETTINGS_DIR / f"{tool}.json")["settings"][setting]
    return {**_setting(tool, setting), "description": block.get("description", ""),
            "inherits": block.get("inherits")}


def load_settings(tool, setting, benchmark_id):
    """Settings of `tool` for one benchmark; raises KeyError if the benchmark is not configured."""
    block = _setting(tool, setting)
    if benchmark_id not in block["benchmarks"]:
        raise KeyError(f"{tool}/{setting} has no settings for {benchmark_id}")
    return _merge(block["defaults"], block["benchmarks"][benchmark_id])


def configured_benchmarks(tool, setting):
    return sorted(_setting(tool, setting)["benchmarks"])
