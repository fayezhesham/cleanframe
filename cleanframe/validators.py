import pandas as pd
import re
from .schema import ColumnRule
from .reporting import (
    log_info,
    log_warning,
    log_error,
    log_duplicates_found,
    log_duplicates_removed,
)
from .transformers import convert_dtype, apply_constraints
from .utils import apply_custom_validator


def validate_column(df: pd.DataFrame, col: str, rule: ColumnRule, report: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    rows_to_drop = pd.Series(False, index=df.index)

    try:
        # Handle missing values
        null_mask = df[col].isnull()
        if null_mask.sum():
            if not rule.allow_null:
                if rule.drop_if_invalid:
                    rows_to_drop |= null_mask
                    log_warning(f"{null_mask.sum()} null(s) in '{col}' marked for drop.", report)
                else:
                    fill_value = rule.fillna
                    if isinstance(fill_value, str) and fill_value.lower() in ["mean", "median", "min", "max"]:
                        try:
                            agg_func = fill_value.lower()
                            if agg_func == "mean":
                                fill_value = df[col].mean()
                            elif agg_func == "median":
                                fill_value = df[col].median()
                            elif agg_func == "min":
                                fill_value = df[col].min()
                            elif agg_func == "max":
                                fill_value = df[col].max()
                        except Exception as e:
                            log_error(f"Failed to compute {rule.fillna} for '{col}': {e}", report)
                            fill_value = None
                    if fill_value is not None:
                        df.loc[null_mask, col] = fill_value
                        log_info(f"Filled {null_mask.sum()} null(s) in '{col}' with {fill_value} (strategy={rule.fillna}).", report)

        # Regex validation (before dtype casting to avoid ghost categories issue)
        if rule.regex:
            try:
                invalid_mask = ~df[col].astype(str).str.match(rule.regex, na=True)
                if invalid_mask.sum():
                    if rule.drop_if_invalid:
                        rows_to_drop |= invalid_mask
                        log_warning(f"{invalid_mask.sum()} value(s) in '{col}' failed regex validation and were marked for drop.", report)
                    else:
                        df.loc[invalid_mask, col] = rule.fillna
                        log_info(f"Replaced {invalid_mask.sum()} value(s) in '{col}' failing regex with {rule.fillna}.", report)
            except re.error as e:
                log_error(f"Invalid regex pattern for column '{col}': {e}", report)

        # Type conversion
        df, type_drop_mask = convert_dtype(df, col, rule, report)
        rows_to_drop |= type_drop_mask

        # Constraint validation
        df, constraint_drop_mask = apply_constraints(df, col, rule, report)
        rows_to_drop |= constraint_drop_mask

        # Custom validator
        df, custom_drop_mask = apply_custom_validator(df, col, rule, report)
        rows_to_drop |= custom_drop_mask

        # Unique constraint handling
        if rule.unique:
            duplicate_mask = df.duplicated(subset=[col], keep=False)
            if duplicate_mask.any():
                log_duplicates_found(col, duplicate_mask.sum(), report)

                # Use resolve_duplicates function to decide which to keep, if provided
                if rule.resolve_duplicates:
                    keep_indices = df.loc[duplicate_mask].groupby(col, group_keys=False).apply(rule.resolve_duplicates).index
                    drop_duplicates_mask = duplicate_mask.copy()
                    drop_duplicates_mask.loc[keep_indices] = False
                else:
                    # Default: keep the first occurrence
                    keep_indices = df.loc[duplicate_mask].drop_duplicates(subset=[col], keep='first').index
                    drop_duplicates_mask = duplicate_mask.copy()
                    drop_duplicates_mask.loc[keep_indices] = False

                rows_to_drop |= drop_duplicates_mask
                log_duplicates_removed(col, drop_duplicates_mask.sum(), report)

    except Exception as e:
        log_error(f"Unexpected error handling column '{col}': {e}", report)

    return df, rows_to_drop
