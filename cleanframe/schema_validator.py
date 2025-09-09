import warnings


class SchemaValidationError(Exception):
    """Custom Exception for schema definition errors."""
    pass


class SchemaValidator:
    """
    Validates column and dataframe schema definitions for logical consistency
    before running data validation.
    """

    NUMERIC_DTYPES = {"int", "float"}
    STRING_DTYPES = {"str", "string", "object"}
    DATE_DTYPES = {"datetime", "date"}

    @classmethod
    def validate_schema(cls, schema: dict, dataframe_rules: dict = None):
        """
        Validate both column-level and dataframe-level schema definitions.

        Args:
            schema (dict): Column-level schema definition.
            dataframe_rules (dict): Optional dataframe-level schema definition.

        Raises:
            SchemaValidationError: If schema definitions are invalid.
        """
        if not isinstance(schema, dict):
            raise SchemaValidationError("Schema must be a dictionary.")

        for col, rules in schema.items():
            rules = {key: value for key, value in rules.items() if value is not None}
            if not isinstance(rules, dict):
                raise SchemaValidationError(f"Schema for column '{col}' must be a dictionary.")

            dtype = rules.get("dtype")
            if not dtype:
                raise SchemaValidationError(f"Column '{col}' schema must specify a 'dtype'.")

            # Check regex validity
            if "regex" in rules and dtype not in cls.STRING_DTYPES:
                raise SchemaValidationError(
                    f"Column '{col}': 'regex' validation is only allowed for string columns. "
                    f"Current dtype: {dtype}"
                )

            # Check min/max applicability
            if ("min" in rules or "max" in rules) and dtype not in cls.NUMERIC_DTYPES.union(cls.DATE_DTYPES):
                raise SchemaValidationError(
                    f"Column '{col}': 'min' and 'max' are only valid for numeric or date columns. "
                    f"Current dtype: {dtype}"
                )

            # FillNA and Allow Null logic
            if rules.get("allow_null") is False and rules.get("fillna") is None and not rules.get("drop_if_invalid"):
                warnings.warn(
                    f"Column '{col}': Nulls are not allowed, but no 'fillna' or 'drop_if_invalid' strategy is defined.",
                    UserWarning
                )

        # Validate dataframe-level rules
        if dataframe_rules:
            cls._validate_dataframe_rules(dataframe_rules)

    @classmethod
    def _validate_dataframe_rules(cls, dataframe_rules: dict):
        """Validate dataframe-wide schema definitions."""
        if not isinstance(dataframe_rules, dict):
            raise SchemaValidationError("Dataframe rules must be a dictionary.")

        min_rows = dataframe_rules.get("min_rows")
        max_rows = dataframe_rules.get("max_rows")

        if min_rows and max_rows and min_rows > max_rows:
            warnings.warn(
                f"Dataframe schema: 'min_rows' ({min_rows}) is greater than 'max_rows' ({max_rows}).",
                UserWarning
            )

        # Add more dataframe-level validations here
        # Example: Ensure unique ID column is specified
        if dataframe_rules.get("unique_id") and not isinstance(dataframe_rules["unique_id"], str):
            raise SchemaValidationError("'unique_id' in dataframe rules must be a string representing a column name.")
