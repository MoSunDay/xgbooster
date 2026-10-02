"""UCI Adult dataset loading and schema derivation (pure functions).

Source: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
Files are gzip-compressed CSV; missing values are coded as "?". The test
file carries a "|1x3 Cross validator" junk first line and labels with a
trailing "." which parse_line normalizes away.
"""

import gzip
import pathlib

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"

COLUMNS = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
)

NUMERIC_COLUMNS = frozenset({
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
})

LABEL_POSITIVE = ">50K"
LABEL_NEGATIVE = "<=50K"


def parse_line(line: str) -> tuple[dict, int] | None:
    """Parse one Adult CSV line into (sample, label); None when skippable.

    Skips blank lines and "|" comment lines (the test-file junk header).
    Raises ValueError on wrong field count, non-integer numeric fields or
    unknown labels.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("|"):
        return None
    fields = [field.strip() for field in stripped.split(",")]
    if len(fields) != len(COLUMNS) + 1:
        raise ValueError(
            f"expected {len(COLUMNS) + 1} csv fields, got {len(fields)}: {stripped!r}"
        )
    sample = {}
    for name, value in zip(COLUMNS, fields[:-1]):
        if value == "?":
            sample[name] = None
        elif name in NUMERIC_COLUMNS:
            sample[name] = int(value)
        else:
            sample[name] = value
    label = fields[-1].rstrip(".")
    if label == LABEL_POSITIVE:
        return sample, 1
    if label == LABEL_NEGATIVE:
        return sample, 0
    raise ValueError(f"unknown label {fields[-1]!r} in line: {stripped!r}")


def load(path) -> tuple[list[dict], list[int]]:
    """Load a gzip CSV Adult file into (samples, labels), skipping junk."""
    samples: list[dict] = []
    labels: list[int] = []
    with gzip.open(pathlib.Path(path), "rt", encoding="ascii") as fh:
        for line in fh:
            parsed = parse_line(line)
            if parsed is None:
                continue
            sample, label = parsed
            samples.append(sample)
            labels.append(label)
    return samples, labels


def build_schema(samples: list) -> list[dict]:
    """Derive a FEATURE_SCHEMA-shaped schema from samples (train-only!).

    Categorical mappings enumerate the sorted distinct non-None values seen
    in the given samples; values absent from this set (e.g. test-only
    categories) vectorize to NaN on both the Python and Rust sides.
    """
    schema = []
    for idx, name in enumerate(COLUMNS):
        if name in NUMERIC_COLUMNS:
            schema.append({"idx": idx, "name": name, "type": "number"})
            continue
        values = sorted({
            sample[name]
            for sample in samples
            if sample.get(name) is not None
        })
        mapping = {value: float(i) for i, value in enumerate(values)}
        schema.append({
            "idx": idx,
            "name": name,
            "type": "categorical",
            "mapping": mapping,
        })
    return schema
