import pandas as pd
from .schema import ColumnRule
from .reporting import log_info, log_warning, log_error

def apply_custom_validator(df: pd.DataFrame, col: str, rule: ColumnRule, report: list[str]) -> tuple[pd.DataFrame, pd.Series]:
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