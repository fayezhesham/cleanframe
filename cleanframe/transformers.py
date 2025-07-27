import pandas as pd
from .schema import ColumnRule
from .reporting import log_info, log_warning, log_error

def convert_dtype(df: pd.DataFrame, col: str, rule: ColumnRule, report: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    rows_to_drop = pd.Series(False, index=df.index)

    if rule.dtype:
        try:
            if rule.dtype == 'datetime':
                converted = pd.to_datetime(df[col], errors='coerce')
            elif rule.dtype in ['int', 'float']:
                converted = pd.to_numeric(df[col], errors='coerce')
            else:
                converted = df[col].astype(rule.dtype)

            invalid_type = converted.isnull() & df[col].notnull()
            if invalid_type.sum():
                if rule.drop_if_invalid:
                    rows_to_drop |= invalid_type
                    log_warning(f"{invalid_type.sum()} invalid type(s) in '{col}' marked for drop.", report)
                else:
                    log_info(f"{invalid_type.sum()} value(s) in '{col}' coerced to NaN.", report)
            df[col] = converted
        except Exception as e:
            log_error(f"Failed type conversion for column '{col}': {e}", report)

    return df, rows_to_drop

def apply_constraints(df: pd.DataFrame, col: str, rule: ColumnRule, report: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    rows_to_drop = pd.Series(False, index=df.index)

    if rule.min is not None:
        below_min = df[col] < rule.min
        if below_min.sum():
            if rule.drop_if_invalid:
                rows_to_drop |= below_min
                log_warning(f"{below_min.sum()} value(s) in '{col}' below min marked for drop.", report)
            else:
                df.loc[below_min, col] = rule.min
                log_info(f"Replaced {below_min.sum()} value(s) in '{col}' below min with {rule.min}.", report)

    if rule.max is not None:
        above_max = df[col] > rule.max
        if above_max.sum():
            if rule.drop_if_invalid:
                rows_to_drop |= above_max
                log_warning(f"{above_max.sum()} value(s) in '{col}' above max marked for drop.", report)
            else:
                df.loc[above_max, col] = rule.max
                log_info(f"Replaced {above_max.sum()} value(s) in '{col}' above max with {rule.max}.", report)

    if rule.allowed_values:
        not_allowed = ~df[col].isin(rule.allowed_values)
        if not_allowed.sum():
            if rule.drop_if_invalid:
                rows_to_drop |= not_allowed
                log_warning(f"{not_allowed.sum()} disallowed value(s) in '{col}' marked for drop.", report)
            else:
                df.loc[not_allowed, col] = rule.fillna
                log_info(f"Replaced {not_allowed.sum()} disallowed value(s) in '{col}' with {rule.fillna}.", report)

    return df, rows_to_drop