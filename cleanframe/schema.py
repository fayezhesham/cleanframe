from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

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
class Schema:
    rules: Dict[str, ColumnRule] = field(default_factory=dict)

    def __init__(self, rules: Optional[Dict[str, Union[ColumnRule, Dict[str, Any]]]] = None):
        self.rules = {}
        if rules:
            for col_name, rule in rules.items():
                if isinstance(rule, dict):
                    rule_obj = ColumnRule(**rule)
                elif isinstance(rule, ColumnRule):
                    rule_obj = rule
                else:
                    raise ValueError(f"Invalid rule type for column '{col_name}': {type(rule)}")
                self.add_column_rule(col_name, rule_obj)

    def add_column_rule(self, column_name: str, rule: ColumnRule):
        self.rules[column_name] = rule

    def get(self, column_name: str) -> Optional[ColumnRule]:
        return self.rules.get(column_name)
