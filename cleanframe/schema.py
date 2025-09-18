from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from .schema_validator import SchemaValidator, SchemaValidationError


@dataclass
class ColumnRule:
    """Defines a set of validation and cleaning rules for a single DataFrame column.

    Attributes:
        dtype (Optional[str]): The expected data type of the column (e.g., 'int', 'float', 'string', 'datetime').
        allow_null (bool): If False, null/NaN values will be handled based on `fillna` or `drop_if_invalid`.
        drop_if_invalid (bool): If True, rows with values that fail a specific rule are dropped.
            If False, the value will be replaced with `fillna` if a fill strategy is defined.
        fillna (Optional[Any]): The value to use when replacing invalid or null values.
            For numeric columns, can be a specific value or a string 'mean', 'median', 'min', or 'max'
            to fill with a calculated aggregate of the valid data.
        min (Optional[Union[int, float]]): The minimum allowed value for a numeric or date column.
        max (Optional[Union[int, float]]): The maximum allowed value for a numeric or date column.
        allowed_values (Optional[List[Any]]): A list of specific values that are permitted in the column.
        regex (Optional[str]): A regular expression pattern for string validation.
        custom_validator (Optional[Callable[[Any, Dict[str, Any]], bool]]): A custom validation function
            that takes a value and the row dictionary, returning True for a valid value.
        unique (bool): If True, ensures all values in the column are unique. Duplicates are handled
            by `resolve_duplicates`.
        resolve_duplicates (Optional[Callable[[Any], Any]]): A custom function to handle duplicate
            values. It receives a DataFrame slice of duplicates and returns the row to keep.
    """
    dtype: Optional[str] = None
    allow_null: bool = True
    drop_if_invalid: bool = False
    fillna: Optional[Any] = None  # can be value OR "mean"/"median"/"min"/"max"
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    regex: Optional[str] = None  # NEW: regex validation for string values
    custom_validator: Optional[Callable[[Any, Dict[str, Any]], bool]] = None
    unique: bool = False
    resolve_duplicates: Optional[Callable[[Any], Any]] = None


@dataclass
class DataFrameRule:
    """Defines a set of validation rules that apply to the entire DataFrame.

    Attributes:
        min_rows (Optional[int]): The minimum number of rows required in the DataFrame.
        max_rows (Optional[int]): The maximum number of rows allowed in the DataFrame.
        no_duplicates (bool): If True, removes exact duplicate rows from the DataFrame.
        unique_keys (Optional[List[str]]): A list of column names that, when combined, must be unique.
        expected_columns (Optional[List[str]]): A list of columns that are expected to be present.
        cross_validations (Optional[List[Dict[str, Any]]]): A list of dictionaries defining cross-column
            validation rules using pandas' `query` and `eval` syntax.

    Example of `cross_validations` dictionary:
    [
        {"type": "comparison", "condition": "start_date <= end_date", "action": "drop"},
        {"type": "aggregate", "check": "df['sales'].sum() > 0"},
        {"type": "conditional", "if": "country == 'US'", "then": "state.notnull()"}
    ]
    """
    min_rows: Optional[int] = None
    max_rows: Optional[int] = None
    no_duplicates: bool = False
    unique_keys: Optional[List[str]] = None
    expected_columns: Optional[List[str]] = None
    cross_validations: Optional[List[Dict[str, Any]]] = None
    # Example:
    # [
    #   {"type": "comparison", "condition": "start_date <= end_date"},
    #   {"type": "aggregate", "check": "df['sales'].sum() > 0"},
    #   {"type": "conditional", "if": "country == 'US'", "then": "state.notnull()"}
    # ]


@dataclass
class Schema:
    """The central class for defining a data validation and cleaning schema.

    A `Schema` object is composed of a set of column-specific rules and optional
    DataFrame-level rules. It validates its own definition upon creation to
    prevent logical inconsistencies.
    """
    rules: Dict[str, ColumnRule] = field(default_factory=dict)
    dataframe_rule: Optional[DataFrameRule] = None

    def __init__(
        self,
        rules: Optional[Dict[str, Union[ColumnRule, Dict[str, Any]]]] = None,
        dataframe_rule: Optional[Union[DataFrameRule, Dict[str, Any]]] = None
    ):
        self.rules = {}

        # Convert column rules
        if rules:
            for col_name, rule in rules.items():
                if isinstance(rule, dict):
                    rule_obj = ColumnRule(**rule)
                elif isinstance(rule, ColumnRule):
                    rule_obj = rule
                else:
                    raise ValueError(f"Invalid rule type for column '{col_name}': {type(rule)}")
                self.add_column_rule(col_name, rule_obj)

        # Convert dataframe rule
        if dataframe_rule:
            if isinstance(dataframe_rule, dict):
                self.dataframe_rule = DataFrameRule(**dataframe_rule)
            elif isinstance(dataframe_rule, DataFrameRule):
                self.dataframe_rule = dataframe_rule
            else:
                raise ValueError(f"Invalid dataframe_rule type: {type(dataframe_rule)}")

        SchemaValidator.validate_schema(
            schema={k: vars(v) for k, v in self.rules.items()},
            dataframe_rules=vars(self.dataframe_rule) if self.dataframe_rule else None
        )

    def add_column_rule(self, column_name: str, rule: ColumnRule):
        """Adds a new `ColumnRule` to the schema.

        Args:
            column_name (str): The name of the column to apply the rule to.
            rule (ColumnRule): The `ColumnRule` instance defining the rules.
        """
        self.rules[column_name] = rule

    def get(self, column_name: str) -> Optional[ColumnRule]:
        """Retrieves a `ColumnRule` by its column name.

        Args:
            column_name (str): The name of the column.

        Returns:
            Optional[ColumnRule]: The `ColumnRule` instance or None if not found.
        """
        return self.rules.get(column_name)
