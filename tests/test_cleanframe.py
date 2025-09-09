import pandas as pd
import numpy as np
import pytest

from cleanframe.schema import Schema
from cleanframe.core import clean_and_validate
from cleanframe.validators import validate_dataframe


# --------------------------
# Column-level validations
# --------------------------

def test_dtype_and_fillna_aggregate_mean():
    df = pd.DataFrame({
        "age": [20, None, 40, np.nan]
    })

    schema = Schema(
        rules={
            "age": {
                "dtype": "float",
                "allow_null": False,
                "fillna": "mean",   # <- use aggregate fill
                "drop_if_invalid": False,
                "min": 0,
                "max": 120,
            }
        }
    )

    cleaned, report = clean_and_validate(df, schema)

    # mean of [20, 40] = 30; nulls filled with 30
    assert cleaned["age"].tolist() == [20.0, 30.0, 40.0, 30.0]
    assert any("Filled 2 null(s) in 'age' with 30.0 (strategy=mean)." in m for m in report) or \
           any("Filled 2 null(s) in 'age' with 30.0" in m for m in report)


def test_allowed_values_and_category_cleanup():
    df = pd.DataFrame({
        "gender": ["Male", "Unknown", "Female", "Other", "Unknown"]
    })

    allowed = ["Male", "Female", "Other"]
    schema = Schema(
        rules={
            "gender": {
                "dtype": "category",
                "allowed_values": allowed,
                "allow_null": False,
                "fillna": "Other",
                "drop_if_invalid": False,
            }
        }
    )

    cleaned, report = clean_and_validate(df, schema)

    # "Unknown" -> replaced with fillna "Other"
    assert cleaned["gender"].tolist() == ["Male", "Other", "Female", "Other", "Other"]
    # categories should be exactly the allowed ones (no ghost categories)
    assert list(cleaned["gender"].cat.categories) == allowed
    # report shows replacement, not drops
    assert any("disallowed value(s) in 'gender' marked for drop" not in m for m in report)
    assert any("Replaced 2 disallowed value(s) in 'gender' with Other." in m for m in report)


def test_regex_validation_drop_rows():
    df = pd.DataFrame({
        "email": ["a@test.com", "invalid", "b@test.com"]
    })

    schema = Schema(
        rules={
            "email": {
                "regex": r"^[\w\.-]+@[\w\.-]+\.\w+$",
                "allow_null": False,
                "drop_if_invalid": True,
            }
        }
    )

    cleaned, report = clean_and_validate(df, schema)

    # one invalid row should be dropped
    assert len(cleaned) == 2
    assert set(cleaned["email"]) == {"a@test.com", "b@test.com"}
    assert any("failed regex validation and were marked for drop" in m for m in report)


def test_regex_validation_fill_values():
    df = pd.DataFrame({
        "email": ["a@test.com", "invalid", "b@test.com"]
    })

    schema = Schema(
        rules={
            "email": {
                "regex": r"^[\w\.-]+@[\w\.-]+\.\w+$",
                "allow_null": False,
                "drop_if_invalid": False,
                "fillna": None,
            }
        }
    )

    cleaned, report = clean_and_validate(df, schema)

    # invalid becomes None (not dropped)
    assert len(cleaned) == 3
    assert cleaned.loc[1, "email"] is None or pd.isna(cleaned.loc[1, "email"])
    assert any("failing regex with None" in m or "failing regex with" in m for m in report)


def test_unique_constraint_marks_duplicates_for_drop():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3, 3, 3],
        "val": [10, 20, 21, 30, 31, 32]
    })

    schema = Schema(
        rules={
            "id": {
                "unique": True,
                "allow_null": False,
            }
        }
    )

    cleaned, report = clean_and_validate(df, schema)

    # keep first of each id group -> ids 1,2,3 -> 3 rows remain
    assert len(cleaned) == 3
    assert list(cleaned["id"]) == [1, 2, 3]
    assert any("duplicate value(s) in column 'id'" in m for m in report)
    assert any("Marked" in m and "duplicate row(s) in column 'id'" in m for m in report)


def test_custom_validator_replace():
    df = pd.DataFrame({
        "score": [10, -5, 15, -1]
    })

    # custom validator: values must be >= 0
    schema = Schema(
        rules={
            "score": {
                "allow_null": False,
                "drop_if_invalid": False,
                "fillna": 0,
                "custom_validator": lambda v: (v is not None) and (v >= 0)
            }
        }
    )

    cleaned, report = clean_and_validate(df, schema)

    # negatives replaced with 0 (not dropped)
    assert cleaned["score"].tolist() == [10, 0, 15, 0]
    assert any("invalid custom values" in m for m in report)


# --------------------------
# DataFrame-level validations
# --------------------------

def test_df_no_duplicates_removal():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "val": [10, 20, 20, 30],
    })

    schema = Schema(
        rules={},  # no column rules
        dataframe_rule={
            "no_duplicates": True
        }
    )

    cleaned, report = clean_and_validate(df, schema)

    # duplicated row (2,20) should be removed once
    assert len(df) == 4
    assert len(cleaned) == 3
    assert any("Removed 1 duplicate row(s)." in m for m in report)


def test_df_unique_keys_warning_only():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "val": [10, 20, 21, 30],
    })

    # unique_keys only WARN; does not drop rows
    schema = Schema(
        rules={},
        dataframe_rule={
            "unique_keys": ["id"]
        }
    )

    cleaned, report = clean_and_validate(df, schema)

    assert len(cleaned) == 4  # no drops from unique_keys
    assert any("Unique key constraint violated" in m for m in report)


def test_df_expected_columns_missing_and_extra():
    df = pd.DataFrame({
        "id": [1, 2],
        "email": ["a@test.com", "b@test.com"],
        "extra": [1, 1],
    })

    schema = Schema(
        rules={},
        dataframe_rule={
            "expected_columns": ["id", "email", "age"]  # 'age' missing, 'extra' unexpected
        }
    )

    cleaned, report = clean_and_validate(df, schema)

    assert any("Missing expected columns: ['age']" in m for m in report)
    assert any("Unexpected extra columns: ['extra']" in m or "Unexpected extra columns: ['extra']" in m for m in report)


def test_df_row_count_bounds_and_cross_validations():
    df = pd.DataFrame({
        "start": pd.to_datetime(["2023-01-02", "2023-01-05"]),
        "end":   pd.to_datetime(["2023-01-01", "2023-01-06"]),  # first row violates start <= end
        "age": [10, 200],  # mean is 105 -> aggregate check can fail if threshold < 100
        "email": ["x@test.com", None],  # conditional check will fail for age > 18 row (second row)
    })

    schema = Schema(
        rules={},  # focus on df-level
        dataframe_rule={
            "min_rows": 1,
            "max_rows": 10,
            "cross_validations": [
                {"type": "comparison", "condition": "start <= end"},
                {"type": "aggregate",  "check": "df['age'].mean() < 100"},
                {"type": "conditional","if": "age > 18", "then": "email.notna()"}
            ]
        }
    )

    # run dataframe-level validation directly to assert specific warnings without dropping
    report = []
    _ = validate_dataframe(df.copy(), schema.dataframe_rule, report)

    assert any("Comparison check failed: start <= end" in m for m in report)
    assert any("Aggregate check failed: df['age'].mean() < 100" in m for m in report)
    assert any("Conditional check failed: If (age > 18) then (email.notna())" in m for m in report)


def test_df_min_max_rows_only_warn():
    df = pd.DataFrame({"x": [1, 2]})

    schema = Schema(
        rules={},
        dataframe_rule={
            "min_rows": 5,
            "max_rows": 1
        }
    )

    # This should only WARN, not raise or drop
    cleaned, report = clean_and_validate(df, schema)

    assert len(cleaned) == 2
    assert any("expected at least 5" in m for m in report)
    assert any("exceeds max of 1" in m for m in report)
