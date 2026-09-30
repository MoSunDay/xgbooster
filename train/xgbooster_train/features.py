"""Feature schema (single source of truth) and schema-driven vectorization."""

import numpy as np

FEATURE_SCHEMA = [
    {"idx": 0, "name": "amount", "type": "number"},
    {"idx": 1, "name": "channel", "type": "categorical",
     "mapping": {"api": 2.0, "app": 1.0, "web": 0.0}},
    {"idx": 2, "name": "hour", "type": "number"},
    {"idx": 3, "name": "is_new_user", "type": "number"},
    {"idx": 4, "name": "txn_count_24h", "type": "number"},
    {"idx": 5, "name": "card_age_days", "type": "number"},
]


def n_features(schema: list) -> int:
    """Return the number of feature columns defined by the schema."""
    return len(schema)


def _to_number(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"expected number, got {type(value).__name__}: {value!r}")


def _to_categorical(value, mapping: dict) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str):
        return float(mapping.get(value, float("nan")))
    raise ValueError(f"expected string, got {type(value).__name__}: {value!r}")


def vectorize(sample: dict, schema: list) -> list[float]:
    """Map one raw sample dict to a feature vector ordered by schema idx."""
    vec = []
    for field in schema:
        value = sample.get(field["name"])
        if field["type"] == "categorical":
            vec.append(_to_categorical(value, field.get("mapping", {})))
        else:
            vec.append(_to_number(value))
    return vec


def to_matrix(samples: list, schema: list) -> np.ndarray:
    """Vectorize samples into a float32 matrix of shape (n, n_features)."""
    if not samples:
        return np.empty((0, n_features(schema)), dtype=np.float32)
    rows = [vectorize(sample, schema) for sample in samples]
    return np.asarray(rows, dtype=np.float32)
