"""Unit tests for xgbooster_train.features (pure functions, no classes)."""

import math
import pathlib
import sys

import numpy as np

CHANNEL = {"api": 2.0, "app": 1.0, "web": 0.0}


def _sample() -> dict:
    """Return one complete raw sample covering every FEATURE_SCHEMA key."""
    return {
        "amount": 120.5,
        "channel": "api",
        "hour": 23,
        "is_new_user": 1,
        "txn_count_24h": 4,
        "card_age_days": 30,
    }


def _expect_value_error(fn, what: str) -> None:
    """Assert that fn() raises ValueError."""
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {what}")


def test_to_number_booleans(features):
    assert features._to_number(True) == 1.0
    assert features._to_number(False) == 0.0


def test_to_number_none_is_nan(features):
    assert math.isnan(features._to_number(None))


def test_to_number_numeric_coercion(features):
    assert features._to_number(7) == 7.0
    assert isinstance(features._to_number(7), float)
    assert features._to_number(2.5) == 2.5


def test_to_number_rejects_string_and_list(features):
    _expect_value_error(lambda: features._to_number("12"), "string value")
    _expect_value_error(lambda: features._to_number([1.0]), "list value")


def test_to_categorical_known_keys(features):
    assert features._to_categorical("web", CHANNEL) == 0.0
    assert features._to_categorical("api", CHANNEL) == 2.0


def test_to_categorical_unknown_and_none_are_nan(features):
    assert math.isnan(features._to_categorical("weeb", CHANNEL))
    assert math.isnan(features._to_categorical(None, CHANNEL))


def test_to_categorical_rejects_number(features):
    _expect_value_error(lambda: features._to_categorical(3, CHANNEL),
                        "number value")


def test_vectorize_full_sample(features):
    schema = features.FEATURE_SCHEMA
    assert features.vectorize(_sample(), schema) == [120.5, 2.0, 23.0, 1.0,
                                                     4.0, 30.0]


def test_vectorize_missing_key_is_nan(features):
    sample = _sample()
    del sample["amount"]
    vec = features.vectorize(sample, features.FEATURE_SCHEMA)
    assert len(vec) == 6
    assert math.isnan(vec[0])
    assert vec[1] == 2.0


def test_vectorize_wrong_type_raises(features):
    sample = _sample()
    sample["amount"] = "120"
    _expect_value_error(
        lambda: features.vectorize(sample, features.FEATURE_SCHEMA),
        "wrong-typed amount")


def test_to_matrix_shape_dtype_and_values(features):
    samples = [_sample(), {**_sample(), "channel": "web"}]
    matrix = features.to_matrix(samples, features.FEATURE_SCHEMA)
    assert matrix.shape == (2, 6)
    assert matrix.dtype == np.float32
    assert matrix[1][1] == 0.0


def test_to_matrix_empty(features):
    matrix = features.to_matrix([], features.FEATURE_SCHEMA)
    assert matrix.shape == (0, 6)
    assert matrix.dtype == np.float32


def test_n_features_matches_schema_length(features):
    schema = features.FEATURE_SCHEMA
    assert features.n_features(schema) == len(schema) == 6


def main() -> int:
    """Discover and run every test_* function here; return exit code."""
    if __package__ in (None, ""):  # run as a script: add the package root
        root = str(pathlib.Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
    from xgbooster_train import features

    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn(features)
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
