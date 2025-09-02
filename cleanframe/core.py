import pandas as pd
from typing import Tuple
from .schema import Schema
from .reporting import log_info, log_warning
from .validators import validate_column, validate_dataframe

def clean_and_validate(df: pd.DataFrame, schema: Schema) -> Tuple[pd.DataFrame, list[str]]:
    df_cleaned = df.copy()
    rows_to_drop = pd.Series(False, index=df_cleaned.index)
    report: list[str] = []

    # 1. Validate DataFrame-level rules
    if schema.dataframe_rule:
        df_cleaned = validate_dataframe(df_cleaned, schema.dataframe_rule, report)

    # 2. Validate columns
    for col, rule in schema.rules.items():
        if col not in df_cleaned.columns:
            log_warning(f"Column '{col}' is missing from DataFrame.", report)
            continue

        df_cleaned, updated_rows_to_drop = validate_column(df_cleaned, col, rule, report)
        rows_to_drop |= updated_rows_to_drop

    # 3. Drop invalid rows
    if rows_to_drop.sum():
        log_info(f"Dropping {rows_to_drop.sum()} row(s) due to validation.", report)
        df_cleaned = df_cleaned[~rows_to_drop]

    return df_cleaned.reset_index(drop=True), report
