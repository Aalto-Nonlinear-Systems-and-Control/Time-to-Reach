"""The result record shared by all tools (configs/schema/result.schema.json)."""

import datetime as _dt
import importlib.metadata
import platform
import subprocess
from pathlib import Path

from . import jsonio, paths
from .spec import validate_against_schema

SCHEMA_VERSION = 1

VERDICTS = (
    "unreachable_proven",
    "reachable_proven",
    "reachable_witness",
    "reachable_possible",
    "reachable_estimated",
    "unreachable_estimated",
    "unknown",
    "unsupported",
    "set_explosion",
    "out_of_memory",
    "timeout",
    "error",
)

_ASSERTS = {
    "unreachable_proven": "unreachable",
    "unreachable_estimated": "unreachable",
    "reachable_proven": "reachable",
    "reachable_witness": "reachable",
    "reachable_estimated": "reachable",
}


def asserted_answer(verdict):
    """'reachable' / 'unreachable' if the verdict asserts an answer, else None."""
    return _ASSERTS.get(verdict)


def agrees_with_expected(verdict, expected_result):
    """True/False if the verdict asserts an answer, None if it does not (possible, unknown, ...)."""
    claim = asserted_answer(verdict)
    return None if claim is None else claim == expected_result


def git_state(repo=paths.REPO_ROOT):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, check=True).stdout.strip()
        # Result files written during a run must not mark the code as modified.
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".",
                                     ":(exclude)results"], cwd=repo,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def package_versions(names):
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def provenance(packages=("numpy", "scipy", "sympy")):
    commit, dirty = git_state()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": {"python": platform.python_version(), **package_versions(packages)},
    }


def make_result(benchmark, tool, setting, verdict, runtime_s, params, *, detail="",
                horizon_covered=None, reach_time_bounds=None, extra=None, packages=("numpy", "scipy", "sympy")):
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    record = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark.id,
        "tool": tool,
        "setting": setting,
        "verdict": verdict,
        "expected_result": benchmark.expected_result,
        "agrees_with_expected": agrees_with_expected(verdict, benchmark.expected_result),
        "detail": detail,
        "time_horizon": list(benchmark.time_horizon),
        "horizon_covered": horizon_covered,
        "reach_time_bounds": reach_time_bounds,
        "runtime_s": runtime_s,
        "params": params,
        "provenance": provenance(packages),
        "extra": extra or {},
    }
    record = jsonio.to_jsonable(record)
    validate_against_schema(record, "result.schema.json")
    return record


def summary_path(tool, benchmark_id, setting):
    return paths.SUMMARY_DIR / tool / benchmark_id / f"{setting}.json"


def write_result(record, path=None):
    path = Path(path) if path else summary_path(record["tool"], record["benchmark"], record["setting"])
    return jsonio.write_json(path, record)
