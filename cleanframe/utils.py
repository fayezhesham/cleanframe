import pandas as pd
from .schema import ColumnRule
from .reporting import log_info, log_warning, log_error

def apply_custom_validator(df: pd.DataFrame, col: str, rule: ColumnRule, report: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Applies a user-defined custom validation function to a DataFrame column.

    This function iterates through the specified column and uses the custom validator 
    function from the `ColumnRule` to check the validity of each value.

    Args:
        df (pd.DataFrame): The DataFrame being processed.
        col (str): The name of the column to validate.
        rule (ColumnRule): The rule object containing the custom validator function.
        report (list[str]): The list to append validation log messages to.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - The DataFrame, potentially with invalid values replaced by `fillna`.
            - A boolean Series indicating which rows failed validation and are marked for drop.

    How it Works:
        The function applies `rule.custom_validator` to each element of the column. The validator 
        function should accept a single value and return `True` for valid values and `False` for invalid ones.
        - If a value fails and `rule.drop_if_invalid` is `True`, the corresponding row is marked for dropping.
        - If a value fails and `rule.drop_if_invalid` is `False`, the value is replaced by `rule.fillna`.

    Example:
        >>> from pandas import DataFrame
        >>> from cleanframe.schema import Schema, ColumnRule
        >>> from cleanframe.utils import apply_custom_validator
        >>>
        >>> # Define a custom validator function
        >>> def is_positive(value):
        ...     return isinstance(value, (int, float)) and value > 0
        >>>
        >>> # Create a rule using the custom validator
        >>> my_rule = ColumnRule(custom_validator=is_positive, drop_if_invalid=True)
        >>>
        >>> # Example DataFrame
        >>> df = DataFrame({'data': [1, 5, -3, 10, 0, 7]})
        >>>
        >>> # Apply the validator (internal library call)
        >>> cleaned_df, rows_to_drop = apply_custom_validator(df, 'data', my_rule, [])
        >>>
        >>> print(cleaned_df[~rows_to_drop])
           data
        0     1
        1     5
        3    10
        5     7
    """
    rows_to_drop = pd.Series(False, index=df.index)

    if rule.custom_validator:
        try:
            invalid = ~df[col].apply(lambda row: rule.custom_validator(row))
            if invalid.sum():
                if rule.drop_if_invalid:
                    rows_to_drop |= invalid
                    log_warning(f"{invalid.sum()} value(s) failed custom validation in '{col}' and were marked for drop.", report)
                else:
                    df.loc[invalid, col] = rule.fillna
                    log_info(f"Replaced {invalid.sum()} invalid custom values in '{col}' with {rule.fillna}.", report)
        except Exception as e:
            log_error(f"Error applying custom validator to '{col}': {e}", report)

    return df, rows_to_drop