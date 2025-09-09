from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from .schema_validator import SchemaValidator, SchemaValidationError


@dataclass
class ColumnRule:
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
        self.rules[column_name] = rule

    def get(self, column_name: str) -> Optional[ColumnRule]:
        return self.rules.get(column_name)
