# `cleanframe`: Lightweight Dataframe Cleaning & Validation

`cleanframe` is a lightweight Python library designed to streamline the process of cleaning and validating dataframes using a simple, schema-based approach. Define your data's expected structure and properties, and let `cleanframe` handle the rest, providing a clean dataframe and a detailed report of any issues found.

---

### ✨ Features

* **Schema-based Validation**: Define your data's structure, rules, and constraints in a simple Python dictionary.
* **Comprehensive Reporting**: Get a detailed report of all validation issues, including which columns or rows were affected.
* **Automatic Cleaning**: Automatically drop or fill rows that don't conform to your schema.
* **Lightweight & Simple**: Designed for ease of use in notebooks, production pipelines, or general data ingestion tasks.
* **Cross-Validation**: Perform complex checks on your data, like ensuring one column is a function of others.

---

### 🚀 Installation

You can install `cleanframe` directly from PyPI using `pip`:

```bash
pip install cleanframe
```

### 📖 Quick Start

Get started in just a few lines of code. The following example demonstrates how to define a schema, validate a sample dataframe, and inspect the results.

```python
from cleanframe.data import sample_data
from cleanframe import clean_and_validate, Schema
import pandas as pd
```
# 1. Load your dataframe
```python
df = sample_data()
print("Original DataFrame shape:", df.shape)
```
# 2. Define your schema using column and dataframe rules
```python
column_rules = {
    'transaction_id': {
        'dtype': 'string',
        'regex': r'^TXN_\d{7}$',
        'allow_null': False,
        'drop_if_invalid': True
    },
    'customer_id': {
        'dtype': 'string',
        'regex': r'^CUST_\d{2}$',
        'allow_null': False,
        'drop_if_invalid': True
    },
    'category': {
        'dtype': 'category',
        'allowed_values': ['Patisserie', 'Milk Products', 'Butchers', 'Beverages', 'Food', 'Computers and electric accessories'],
        'allow_null': False,
        'drop_if_invalid': True,
        'fillna': 'Other'
    },
    'price_per_unit': {
        'dtype': 'float',
        'min': 0,
        'max': 10000,
        'allow_null': False,
        'drop_if_invalid': True,
        'fillna': 0
    },
    'quantity': {
        'dtype': 'int',
        'min': 1,
        'max': 1000,
        'allow_null': False,
        'drop_if_invalid': True,
        'fillna': 1
    },
    'total_spent': {
        'dtype': 'float',
        'min': 0,
        'max': 1000000,
        'allow_null': False,
        'drop_if_invalid': True,
        'fillna': 0
    },
    'payment_method': {
        'dtype': 'category',
        'allowed_values': ['Digital Wallet', 'Credit Card', 'Cash'],
        'allow_null': False,
        'drop_if_invalid': True,
        'fillna': 'Other'
    },
    'location': {
        'dtype': 'category',
        'allowed_values': ['Online', 'In-store'],
        'allow_null': False,
        'drop_if_invalid': True,
        'fillna': 'Other'
    },
    'transactoin_date': {
        'dtype': 'datetime',
        'allow_null': False,
        'drop_if_invalid': True,
        'max': pd.Timestamp.today()
    },
    'discount_applied': {
        'dtype': 'boolean',
        'allow_null': False,
        'drop_if_invalid': True
    }
}

dataframe_rule = {
    "min_rows": 1,
    "unique_keys": ["transaction_id"],
    "no_duplicates": True,
    "cross_validations": [{"type": "comparison", "condition": "total_spent == price_per_unit * quantity"}]
}

schema = Schema(rules = column_rules, dataframe_rule = dataframe_rule)
```
# 3. Clean and validate your data
```python
cleaned_df, report = clean_and_validate(df, schema)
```
# 4. View the report and the cleaned dataframe
```python
print("\nValidation Report:")
print(report)
print("\nCleaned DataFrame shape:", cleaned_df.shape)
```

### 🔧 Schema Definition

The schema is a Python dictionary that defines the validation rules for your dataframe. It consists of two main parts:

* **`column_rules`**: A dictionary where each key is a column name and the value is a dictionary of rules for that column (e.g., `dtype`, `min`, `max`, `regex`, `allowed_values`, `allow_null`, `fillna`, and `drop_if_invalid`).
* **`dataframe_rule`**: A dictionary for rules that apply to the entire dataframe (e.g., `min_rows`, `unique_keys`, `no_duplicates`, `cross_validations`).

This structure allows for granular control over every aspect of your data's quality.

### 🤝 Contributing

We welcome contributions! If you find a bug or have a suggestion, please open an issue or submit a pull request on our GitHub repository.

### 📜 License

This project is licensed under the MIT License.