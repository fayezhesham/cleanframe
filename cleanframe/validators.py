import pandas as pd
import re
from .schema import ColumnRule, DataFrameRule
from .reporting import (
    log_info,
    log_warning,
    log_error,
    log_duplicates_found,
    log_duplicates_removed,
)
from .transformers import convert_dtype, apply_constraints
from .utils import apply_custom_validator


def validate_dataframe(df: pd.DataFrame, df_rule: DataFrameRule, report: list[str]) -> pd.DataFrame:
    """Validates an entire DataFrame against a set of DataFrame-level rules.

    This function applies various checks and cleaning operations that relate to the
    structure and overall state of the DataFrame, as defined by `DataFrameRule`.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        df_rule (DataFrameRule): The object containing the DataFrame-wide rules.
        report (list[str]): The list to append validation log messages to.

    Returns:
        pd.DataFrame: The validated DataFrame, with any modifications applied
            (e.g., dropped rows, removed duplicates).

    How it Works:
        The function applies a sequence of checks:
        1. **Row count:** Logs a warning if the DataFrame's row count is outside of the `min_rows` or `max_rows` range.
        2. **Full duplicates:** Removes exact duplicate rows if `no_duplicates` is True.
        3. **Unique keys:** Logs a warning if duplicates are found within the specified `unique_keys` subset.
        4. **Expected columns:** Logs warnings for any missing or unexpected columns based on `expected_columns`.
        5. **Cross-validations:** Iterates through `cross_validations` rules.
           - **'comparison' type:** Evaluates a boolean condition using `df.eval()` and can drop rows that fail.
           - **'aggregate' type:** Evaluates an aggregate check on the entire DataFrame. It only logs a warning and cannot drop rows.
           - **'conditional' type:** Evaluates a `then` condition for rows that satisfy an `if` condition. It can drop rows that fail the check.
    """
    try:
        # Row count checks
        if df_rule.min_rows is not None and len(df) < df_rule.min_rows:
            log_warning(f"DataFrame has only {len(df)} rows; expected at least {df_rule.min_rows}.", report)
        if df_rule.max_rows is not None and len(df) > df_rule.max_rows:
            log_warning(f"DataFrame has {len(df)} rows; exceeds max of {df_rule.max_rows}.", report)

        # Remove duplicates
        if df_rule.no_duplicates:
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                df = df.drop_duplicates()
                log_info(f"Removed {dup_count} duplicate row(s).", report)

        # Unique keys check
        if df_rule.unique_keys:
            dup_keys = df.duplicated(subset=df_rule.unique_keys).sum()
            if dup_keys > 0:
                log_warning(f"Unique key constraint violated: {dup_keys} duplicate(s) found in {df_rule.unique_keys}.", report)

        # Expected columns check
        if df_rule.expected_columns:
            missing = [c for c in df_rule.expected_columns if c not in df.columns]
            extra = [c for c in df.columns if c not in df_rule.expected_columns]
            if missing:
                log_warning(f"Missing expected columns: {missing}", report)
            if extra:
                log_warning(f"Unexpected extra columns: {extra}", report)

        # Cross validations
        if df_rule.cross_validations:
            for check in df_rule.cross_validations:
                try:
                    action = check.get("action", "warn")  # default to warning
                    invalid_mask = None

                    if check.get("type") == "comparison":
                        cond = check["condition"]
                        mask = df.eval(cond)
                        invalid_mask = ~mask

                        if invalid_mask.any():
                            if action == "drop":
                                df = df.loc[mask].copy()
                                log_info(f"Dropped {invalid_mask.sum()} row(s) failing comparison: {cond}", report)
                            else:
                                log_warning(f"Comparison check failed: {cond}", report)

                    elif check.get("type") == "aggregate":
                        agg_check = check["check"]
                        result = eval(agg_check, {}, {"df": df})
                        if not result:
                            if action == "drop":
                                # Aggregate checks don't identify specific rows, so we can't drop selectively
                                log_warning(f"Aggregate check failed and cannot drop rows: {agg_check}", report)
                            else:
                                log_warning(f"Aggregate check failed: {agg_check}", report)

                    elif check.get("type") == "conditional":
                        if_cond = df.eval(check["if"])
                        then_cond = df.eval(check["then"])
                        invalid_mask = if_cond & ~then_cond

                        if invalid_mask.any():
                            if action == "drop":
                                df = df.loc[~invalid_mask].copy()
                                log_info(
                                    f"Dropped {invalid_mask.sum()} row(s) failing conditional: If ({check['if']}) then ({check['then']})",
                                    report,
                                )
                            else:
                                log_warning(
                                    f"Conditional check failed: If ({check['if']}) then ({check['then']})",
                                    report,
                                )

                except Exception as e:
                    log_error(f"Error evaluating cross-validation {check}: {e}", report)

    except Exception as e:
        log_error(f"Unexpected error in DataFrame validation: {e}", report)

    return df


def validate_column(df: pd.DataFrame, col: str, rule: ColumnRule, report: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Validates and cleans a single DataFrame column based on a set of rules.

    This function is a pipeline that applies a series of cleaning and validation steps
    to a specific column, accumulating rows to be dropped.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to be validated.
        col (str): The name of the column to validate.
        rule (ColumnRule): The object containing the column-specific rules.
        report (list[str]): The list to append validation log messages to.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - The DataFrame with any in-place cleaning applied (e.g., filled nulls, type conversions).
            - A boolean Series of rows marked for removal due to validation failures.

    How it Works:
        The function sequentially applies validation rules and combines the results:
        1. **Null handling:** Replaces or marks null values for dropping based on `allow_null` and `fillna`.
        2. **Regex validation:** Checks for a valid pattern and replaces or drops non-matching values.
        3. **Type conversion:** Converts the column to the specified `dtype`. Rows that fail conversion are marked for dropping.
        4. **Constraint validation:** Applies `min`, `max`, and `allowed_values` rules. Rows that fail are marked for dropping.
        5. **Custom validation:** Applies the user-defined `custom_validator` function.
        6. **Uniqueness:** Checks for and handles duplicate values based on the `unique` and `resolve_duplicates` rules.
    """
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
