"""Loads configs/config.yaml and resolves paths relative to the repository root.

Input: path to a YAML config (default: configs/config.yaml).
Output: a plain dict; `resolve_path(cfg, ...)` turns relative paths into absolute ones.
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p):
    """Absolute paths are kept; relative ones are taken from the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def dataset_root(cfg, name):
    """Folder of one dataset: <data_root>/<datasets.<name>.root>."""
    return resolve_path(cfg["paths"]["data_root"]) / cfg["datasets"][name]["root"]
