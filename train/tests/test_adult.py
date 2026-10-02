"""Unit tests for xgbooster_train.adult (pure functions, no classes)."""

import gzip
import math
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_GZ = REPO_ROOT / "datasets" / "adult" / "adult.data.gz"
TEST_GZ = REPO_ROOT / "datasets" / "adult" / "adult.test.gz"
N_HEAD_LINES = 40


def _head_lines(path, n):
    """First n text lines of a gzip dataset file."""
    with gzip.open(path, "rt", encoding="ascii") as fh:
        return [line for _, line in zip(range(n), fh)]


def _expect_value_error(fn, what: str) -> None:
    """Assert that fn() raises ValueError."""
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {what}")


def test_parse_line_basic(adult, features):
    sample, label = adult.parse_line(
        "39, State-gov, 77516, Bachelors, 13, Never-married, "
        "Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, "
        "United-States, <=50K\n")
    assert label == 0
    assert sample["age"] == 39
    assert sample["workclass"] == "State-gov"
    assert sample["fnlwgt"] == 77516
    assert sample["native_country"] == "United-States"
    assert set(sample) == set(adult.COLUMNS)


def test_parse_line_strips_and_maps_labels(adult, features):
    def row(label):
        return (f"50, Private, 83311, Bachelors, 13, Married-civ-spouse, "
                f"Exec-managerial, Husband, White, Male, 0, 0, 13, "
                f"United-States, {label}")
    _, pos = adult.parse_line(row(">50K"))
    _, pos_dot = adult.parse_line(row(">50K.") + " ")
    _, neg_dot = adult.parse_line(row("<=50K."))
    assert (pos, pos_dot, neg_dot) == (1, 1, 0)


def test_parse_line_blank_and_comment_are_none(adult, features):
    assert adult.parse_line("") is None
    assert adult.parse_line("   \n") is None
    assert adult.parse_line("|1x3 Cross validator\n") is None


def test_parse_line_question_mark_is_none(adult, features):
    sample, _ = adult.parse_line(
        "39, ?, 77516, Bachelors, 13, Never-married, ?, Not-in-family, "
        "White, Male, 2174, 0, 40, ?, <=50K")
    assert sample["workclass"] is None
    assert sample["occupation"] is None
    assert sample["native_country"] is None
    assert sample["age"] == 39


def test_parse_line_field_count_and_label_errors(adult, features):
    _expect_value_error(lambda: adult.parse_line("39, State-gov, <=50K"),
                        "wrong field count")
    _expect_value_error(
        lambda: adult.parse_line(
            "39, ?, 77516, Bachelors, 13, Never-married, ?, Not-in-family, "
            "White, Male, 2174, 0, 40, United-States, 50K"),
        "unknown label")
    _expect_value_error(
        lambda: adult.parse_line(
            "39, ?, x, Bachelors, 13, Never-married, ?, Not-in-family, "
            "White, Male, 2174, 0, 40, United-States, <=50K"),
        "non-integer numeric field")


def test_load_head_matches_parse(adult, features):
    lines = _head_lines(DATA_GZ, N_HEAD_LINES)
    samples, labels = adult.load(DATA_GZ)
    assert len(samples) == len(labels)
    first = adult.parse_line(lines[0])
    assert samples[0] == first[0] and labels[0] == first[1]
    assert all(sample["age"] > 0 for sample in samples[:N_HEAD_LINES])


def test_build_schema_shape_and_order(adult, features):
    samples, _ = adult.load(DATA_GZ)
    schema = adult.build_schema(samples)
    assert len(schema) == len(adult.COLUMNS) == 14
    assert [f["idx"] for f in schema] == list(range(14))
    assert [f["name"] for f in schema] == list(adult.COLUMNS)
    numeric = {f["name"] for f in schema if f["type"] == "number"}
    assert numeric == set(adult.NUMERIC_COLUMNS)


def test_build_schema_deterministic_sorted_mapping(adult, features):
    samples, _ = adult.load(DATA_GZ)
    first = adult.build_schema(samples)
    reordered = adult.build_schema(list(reversed(samples)))
    assert first == reordered
    mapping = first[1]["mapping"]  # workclass
    keys = [k for k in mapping if k is not None]
    assert keys == sorted(keys)
    assert set(mapping.values()) == set(range(len(mapping)))
    assert mapping.get("?") is None  # None never enters the mapping


def test_build_schema_empty_categorical(adult, features):
    schema = adult.build_schema([])
    assert len(schema) == 14
    empty = [f for f in schema if f["type"] == "categorical"]
    assert all(f["mapping"] == {} for f in empty)


def test_vectorize_adult_schema(adult, features):
    samples, _ = adult.load(DATA_GZ)
    schema = adult.build_schema(samples)
    vec = features.vectorize(samples[0], schema)
    assert len(vec) == 14
    assert vec[0] == float(samples[0]["age"])
    assert vec[1] == schema[1]["mapping"]["State-gov"]
    unknown = {**samples[0], "native_country": "Atlantis"}
    assert math.isnan(features.vectorize(unknown, schema)[13])
    missing = {**samples[0], "occupation": None}
    assert math.isnan(features.vectorize(missing, schema)[6])

    matrix = features.to_matrix(samples[:10], schema)
    assert matrix.shape == (10, 14)
    assert matrix.dtype == np.float32


def main() -> int:
    """Discover and run every test_* function here; return exit code."""
    if __package__ in (None, ""):  # run as a script: add the package root
        root = str(pathlib.Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
    from xgbooster_train import adult, features

    if not DATA_GZ.is_file() or not TEST_GZ.is_file():
        print(f"SKIP: adult datasets missing under {REPO_ROOT / 'datasets'}")
        return 0

    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn(adult, features)
        except Exception as exc:  # report failures, keep running
            failed += 1
            print(f"FAIL {name}: {exc!r}")
        else:
            print(f"ok {name}")
    if failed:
        print(f"{failed}/{len(tests)} tests failed")
        return 1
    print(f"ok {len(tests)} tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
