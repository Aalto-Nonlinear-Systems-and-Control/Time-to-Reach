"""JSON reading/writing with readable formatting and standard-compliant infinities."""

import json
import math
import re
from pathlib import Path

import numpy as np

# A list spread over several lines whose items contain no nested list/object.
_FLAT_LIST = re.compile(r"\[\n\s*([^\[\]{}]*?)\n\s*\]")


def to_jsonable(obj):
    """Convert numpy types and non-finite floats into plain JSON values.

    +-inf become the strings "inf" / "-inf"; NaN becomes null.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(obj, Path):
        return str(obj)
    return obj


def from_jsonable_float(x):
    """Inverse of the infinity encoding used by to_jsonable."""
    if x == "inf":
        return math.inf
    if x == "-inf":
        return -math.inf
    return x


def dumps(obj):
    """Indented JSON in which innermost lists of scalars stay on one line."""
    text = json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False)

    def collapse(match):
        items = [s.strip() for s in match.group(1).split(",\n")]
        return "[" + ", ".join(items) + "]"

    return _FLAT_LIST.sub(collapse, text) + "\n"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj))
    return path


def read_json(path):
    return json.loads(Path(path).read_text())
