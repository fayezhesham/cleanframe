import pandas as pd
from typing import Tuple
from .schema import Schema
from .reporting import log_info, log_warning
from .validators import validate_column, validate_dataframe

def clean_and_validate(df: pd.DataFrame, schema: Schema) -> Tuple[pd.DataFrame, list[str]]:
    """Cleans and validates a pandas DataFrame based on a user-defined schema.

    This is the core function of the library. It applies a series of cleaning and 
    validation rules specified in a `Schema` object, including checks for the 
    entire DataFrame and individual columns. It returns a cleaned DataFrame and 
    a detailed report of all actions taken.

    Args:
        df (pd.DataFrame): The pandas DataFrame to be cleaned and validated.
        schema (Schema): A `Schema` object that defines the validation and 
            cleaning rules. It can be created from a `Schema` instance or a 
            dictionary.

    Returns:
        Tuple[pd.DataFrame, list[str]]: A tuple containing:
            - **pd.DataFrame**: The cleaned and validated DataFrame. Invalid rows 
              that fail validation with `drop_if_invalid=True` are removed.
            - **list[str]**: A list of strings representing the report of all 
              cleaning and validation actions (e.g., warnings, info messages). 
              This list can be passed to `reporting.display_report` for a 
              nicely formatted output.

    Example:
        >>> from pandas import DataFrame
        >>> from cleanframe import Schema
        >>> from cleanframe import clean_and_validate
        >>>
        >>> data = {'id': [1, 2, 3, 4],
        ...         'name': ['Alice', 'Bob', 'Charlie', None],
        ...         'age': [25, 30, 18, 99],
        ...         'zip_code': ['12345', 'abcde', '67890', '98765']}
        >>> df_to_clean = DataFrame(data)
        >>>
        >>> # Define a schema using a dictionary
        >>> my_schema = Schema(
        ...     rules={
        ...         'id': {'dtype': 'int', 'unique': True},
        ...         'name': {'dtype': 'string', 'allow_null': False, 'fillna': 'Unknown'},
        ...         'age': {'dtype': 'int', 'min': 18, 'max': 60, 'drop_if_invalid': True},
        ...         'zip_code': {'dtype': 'string', 'regex': r'^\d{5}$'},
        ...     },
        ...     dataframe_rule={'min_rows': 3, 'no_duplicates': True}
        ... )
        >>>
        >>> cleaned_df, report = clean_and_validate(df_to_clean, my_schema)
        >>>
        >>> print(cleaned_df)
           id     name  age zip_code
        0   1    Alice   25    12345
        1   2      Bob   30    abcde
        2   3  Charlie   18    67890
        >>> # The report would contain the messages:
        >>> # 'Filled 1 null(s) in 'name' with Unknown (strategy=Unknown).'
        >>> # '1 value(s) in 'age' above max marked for drop.'
        >>> # 'Replaced 1 value(s) in 'zip_code' failing regex with None.'
        >>> # 'Dropped 1 row(s) due to validation.'
    """
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
