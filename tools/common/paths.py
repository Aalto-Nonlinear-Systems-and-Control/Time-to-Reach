"""Repository locations shared by all Python tools and scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
BENCHMARK_DIR = CONFIG_DIR / "benchmarks"
SETTINGS_DIR = CONFIG_DIR / "settings"
SCHEMA_DIR = CONFIG_DIR / "schema"
RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_DIR = RESULTS_DIR / "summary"
RAW_DIR = RESULTS_DIR / "raw"
