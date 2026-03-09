# Cleansly 🧹

**Enterprise-grade data cleansing library for Python.**  
A composable, auditable, and production-ready toolkit for cleaning, validating, and transforming pandas DataFrames.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![Pandas](https://img.shields.io/badge/pandas-1.5%2B-green)](https://pandas.pydata.org)
[![Tests](https://img.shields.io/badge/tests-46%20passing-brightgreen)](#testing)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Components](#components)
  - [CleaningPipeline](#cleaningpipeline)
  - [TextCleaner](#textcleaner)
  - [NumericCleaner](#numericcleaner)
  - [DateTimeCleaner](#datetimecleaner)
  - [MissingValueHandler](#missingvaluehandler)
  - [SchemaValidator](#schemavalidator)
  - [RuleValidator](#rulevalidator)
  - [Standardizer](#standardizer)
  - [Encoder](#encoder)
  - [DataProfiler](#dataprofiler)
- [Pipeline Reports](#pipeline-reports)
- [Error Handling](#error-handling)
- [Logging](#logging)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Features

| Capability | Details |
|---|---|
| **Text Cleaning** | Whitespace, casing, HTML, URLs, Unicode, PII redaction, regex |
| **Numeric Cleaning** | Type coercion, range clipping, IQR/z-score outlier handling, rounding |
| **DateTime Parsing** | Multi-format parsing, timezone handling, range validation, component extraction |
| **Missing Values** | Mean/median/mode/forward/backward/interpolation/constant fill, column/row drop thresholds |
| **Schema Validation** | Dtype, nullability, ranges, allowed values, regex, string length, uniqueness |
| **Rule Validation** | Row-level and frame-level custom business logic rules |
| **Standardization** | Column renaming, dtype casting, snake_case normalization, value maps, deduplication |
| **Encoding** | Label encoding (with invertible mappings) and one-hot encoding |
| **Data Profiling** | Null rates, unique counts, numeric stats, top values, memory usage, duplicate detection |
| **Pipeline** | Chainable steps, execution reports, per-step metrics, pre/post profiling |
| **Logging** | Structured JSON or human-readable logs with configurable levels |

---

## Installation

**From source (development):**
```bash
git clone https://github.com/your-org/cleansly.git
cd cleansly
pip install -e ".[dev]"
```

**Requirements:** Python 3.9+, pandas ≥ 1.5, numpy ≥ 1.23

---

## Quick Start

```python
import pandas as pd
from cleansly import (
    CleaningPipeline, TextCleaner, NumericCleaner,
    MissingValueHandler, Standardizer, SchemaValidator,
)
from cleansly.validators.schema import ColumnSchema

df = pd.read_csv("customers.csv")

schema = {
    "customer_id": ColumnSchema(dtype="int64", nullable=False, unique=True),
    "email":       ColumnSchema(nullable=False, regex=r"^[^@]+@[^@]+\.[^@]+$"),
    "age":         ColumnSchema(min_value=0, max_value=120),
    "status":      ColumnSchema(allowed_values=["active", "inactive", "pending"]),
}

pipeline = (
    CleaningPipeline(name="CustomerPipeline", profile=True)
    .add_step("standardize",    Standardizer(column_name_style="snake_case", drop_duplicates=True))
    .add_step("clean_text",     TextCleaner(strip_whitespace=True, normalize_case="lower", remove_html=True))
    .add_step("handle_missing", MissingValueHandler(strategy="fill_median"))
    .add_step("clean_numerics", NumericCleaner(outlier_method="iqr", clip_lower=0))
    .add_step("validate",       SchemaValidator(schema, on_error="warn"))
)

df_clean, report = pipeline.run(df)
print(report.summary())
df_clean.to_csv("customers_clean.csv", index=False)
```

---

## Core Concepts

### Composable Steps
Every component exposes a consistent interface:
- **Cleaners** → `.clean(df, columns=None) -> DataFrame`
- **Transformers** → `.transform(df) -> DataFrame`
- **Validators** → `.validate(df) -> ValidationResult`
- **Callables** → `(df) -> DataFrame`

All of these work seamlessly as pipeline steps.

### Immutability
Every step receives a **copy** of the DataFrame — the original is never mutated.

### Auditability
The `CleaningPipeline` records every step's row/column deltas, execution time, and errors in a `PipelineReport`.

---

## Components

### CleaningPipeline

The central orchestrator. Chains steps in order, collects metrics, and generates a full execution report.

```python
from cleansly import CleaningPipeline

pipeline = (
    CleaningPipeline(
        name="MyPipeline",
        profile=True,       # Enable pre/post data profiling
        on_error="continue" # "continue" or "abort" on step failure
    )
    .add_step("step_1", some_cleaner)
    .add_step("step_2", some_transformer)
    .add_step("custom", lambda df: df.dropna(subset=["id"]))
)

df_clean, report = pipeline.run(df_raw)
print(report.summary())
```

---

### TextCleaner

```python
from cleansly import TextCleaner

cleaner = TextCleaner(
    strip_whitespace=True,
    normalize_case="lower",      # "lower", "upper", "title"
    remove_html=True,
    remove_urls=True,
    normalize_unicode=True,
    redact_emails=True,
    redact_phones=True,
    redact_ssn=True,
    redact_credit_cards=True,
    redact_placeholder="[REDACTED]",
    custom_replacements=[
        (r"\bN/A\b", ""),        # (regex_pattern, replacement)
    ],
    min_length=2,
    max_length=200,
    null_if_empty=True,
)

df = cleaner.clean(df, columns=["name", "notes", "description"])
```

**PII Redaction** automatically scrubs emails, phone numbers, SSNs, and credit card numbers using production-grade regex patterns.

---

### NumericCleaner

```python
from cleansly import NumericCleaner

cleaner = NumericCleaner(
    coerce=True,               # Convert strings to numeric (coerce errors to NaN)
    clip_lower=0,              # Hard lower bound
    clip_upper=1_000_000,      # Hard upper bound
    outlier_method="iqr",      # "iqr", "zscore", or "none"
    outlier_action="clip",     # "clip" or "null"
    iqr_multiplier=1.5,        # IQR fence multiplier (Tukey's method)
    zscore_threshold=3.0,      # Z-score threshold for outlier detection
    round_decimals=2,
    fill_value=0.0,            # Fill remaining NaNs after cleaning
)

df = cleaner.clean(df, columns=["revenue", "age", "score"])
```

---

### DateTimeCleaner

```python
from cleansly import DateTimeCleaner

cleaner = DateTimeCleaner(
    input_formats=["%Y-%m-%d", "%d/%m/%Y", "%B %d %Y"],
    output_format="%Y-%m-%d",  # None = keep as datetime64
    timezone="UTC",
    min_date="2000-01-01",
    max_date="2030-12-31",
    null_out_of_range=True,    # Set out-of-range dates to NaT/null
    extract_components=True,   # Adds _year, _month, _day, _hour, _quarter columns
    dayfirst=False,
    yearfirst=False,
)

df = cleaner.clean(df, columns=["created_at", "updated_at"])
# New columns if extract_components=True:
# created_at_year, created_at_month, created_at_day, created_at_dayofweek,
# created_at_hour, created_at_quarter
```

---

### MissingValueHandler

```python
from cleansly import MissingValueHandler

handler = MissingValueHandler(
    strategy="fill_median",             # Default strategy
    column_strategies={                  # Per-column overrides
        "status":    "fill_mode",
        "notes":     "fill_constant",
        "ts_column": "fill_forward",
    },
    fill_values={"notes": "N/A"},        # Used with fill_constant
    drop_threshold=0.8,                  # Drop columns with >80% nulls
    row_drop_threshold=0.9,              # Drop rows with >90% nulls
    interpolate_method="linear",
)

df = handler.clean(df)
```

**Available strategies:** `fill_mean`, `fill_median`, `fill_mode`, `fill_constant`, `fill_forward`, `fill_backward`, `fill_interpolate`, `drop_rows`, `drop_cols`

---

### SchemaValidator

```python
from cleansly import SchemaValidator
from cleansly.validators.schema import ColumnSchema

schema = {
    "user_id":   ColumnSchema(dtype="int64",   nullable=False, unique=True),
    "email":     ColumnSchema(nullable=False,  regex=r"^[^@]+@[^@]+\.[^@]+$"),
    "age":       ColumnSchema(dtype="float64", min_value=0, max_value=150),
    "status":    ColumnSchema(allowed_values=["active", "inactive", "pending"]),
    "username":  ColumnSchema(min_length=3, max_length=50),
}

validator = SchemaValidator(
    schema,
    require_all_columns=True,
    allow_extra_columns=True,
    on_error="warn",   # "warn", "raise", or "return"
)

result = validator.validate(df)
print(result.summary())
# result.is_valid  → bool
# result.errors    → list of error messages
# result.warnings  → list of warning messages
```

---

### RuleValidator

For cross-column and business logic rules:

```python
from cleansly import RuleValidator
from cleansly.validators.rules import Rule

rules = [
    # Row-level rule
    Rule(
        name="end_after_start",
        condition=lambda row: row["end_date"] >= row["start_date"],
        is_row_level=True,
        severity="error",
        message="end_date must be >= start_date.",
    ),
    # Frame-level rule
    Rule(
        name="revenue_nonzero",
        condition=lambda df: df["revenue"].sum() > 0,
        is_row_level=False,
        severity="warning",
        message="Total revenue is zero — possible ingestion issue.",
    ),
]

validator = RuleValidator(rules)
result = validator.validate(df)
print(result.summary())
```

---

### Standardizer

```python
from cleansly import Standardizer

standardizer = Standardizer(
    column_name_style="snake_case",        # "snake_case", "lower", "upper"
    rename_columns={"cust_id": "customer_id", "rev": "revenue"},
    dtype_map={"customer_id": "int64", "revenue": "float64"},
    value_maps={
        "active_flag": {"Y": True, "N": False, "1": True, "0": False},
    },
    drop_duplicates=True,
    duplicate_subset=["customer_id"],
    duplicate_keep="first",
    reset_index=True,
)

df = standardizer.transform(df)
```

---

### Encoder

```python
from cleansly import Encoder

encoder = Encoder(
    label_columns=["status", "region"],    # Ordinal label encoding
    onehot_columns=["product_category"],   # One-hot encoding
    drop_first=True,                       # Drop first dummy (avoid multicollinearity)
)

df_encoded = encoder.transform(df)

# Inspect the learned mappings
print(encoder.label_mappings_)
# {"status": {"active": 0, "inactive": 1, "pending": 2}}

# Reverse label encoding
df_original = encoder.inverse_transform_labels(df_encoded, columns=["status"])
```

---

### DataProfiler

```python
from cleansly import DataProfiler

profiler = DataProfiler(top_n=10)  # Top N most frequent values per column
profile = profiler.profile(df)

# Print human-readable summary
print(profile.summary())

# Access structured data
profile.row_count          # int
profile.null_pct           # float
profile.duplicate_rows     # int
profile.memory_usage_mb    # float
profile.columns["age"].mean     # float
profile.columns["age"].q75      # float
profile.columns["name"].avg_length

# Serialize to dict / JSON
import json
print(json.dumps(profile.to_dict(), indent=2, default=str))
```

---

## Pipeline Reports

Every `pipeline.run()` call returns a `PipelineReport`:

```
=================================================================
PIPELINE EXECUTION REPORT
=================================================================
  Total Duration  : 0.042s
  Steps Executed  : 6
-----------------------------------------------------------------
   1. [✓] standardize                    0.003s  rows Δ-1  cols Δ0
   2. [✓] clean_text                     0.005s  rows Δ0   cols Δ0
   3. [✓] handle_missing                 0.004s  rows Δ0   cols Δ0
   4. [✓] clean_numerics                 0.003s  rows Δ0   cols Δ0
   5. [✓] validate_schema                0.008s  rows Δ0   cols Δ0
   6. [✓] validate_rules                 0.002s  rows Δ0   cols Δ0
-----------------------------------------------------------------
  DATA SHAPE CHANGE
    Rows  : 7 → 6
    Cols  : 9 → 9
    Nulls : 12.70% → 4.76%
    Dupes : 1 → 0
=================================================================
```

---

## Error Handling

Each component supports an `errors` parameter:

```python
# "coerce" — Log a warning and continue (default, safe for production)
cleaner = TextCleaner(errors="coerce")

# "raise" — Raise TransformationError / ValidationError / SchemaError on first failure
cleaner = TextCleaner(errors="raise")

# Pipeline-level abort
pipeline = CleaningPipeline(on_error="abort")  # Stops at first failed step
pipeline = CleaningPipeline(on_error="continue")  # Records error, continues (default)
```

Custom exception hierarchy:
```
CleanslyException
├── TransformationError  (step, column)
├── ValidationError      (field, value, rule)
├── SchemaError          (expected, found)
└── PipelineError
```

---

## Logging

```python
from cleansly.utils.logger import get_logger
import logging

# Human-readable console output (default)
logger = get_logger("myapp", level=logging.DEBUG)

# Structured JSON (for Datadog, Splunk, CloudWatch, etc.)
logger = get_logger("myapp", structured=True, log_file="/var/log/cleansly.log")
```

Sample structured JSON output:
```json
{
  "timestamp": "2024-11-15T10:23:41.012345+00:00",
  "level": "INFO",
  "logger": "cleansly.pipeline",
  "message": "Pipeline 'CustomerPipeline' completed in 0.042s. 6/6 steps succeeded.",
  "module": "pipeline",
  "function": "run",
  "line": 98
}
```

---

## Testing

```bash
# Run the full test suite (46 tests)
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=cleansly --cov-report=term-missing

# Run a specific test class
pytest tests/ -v -k TestTextCleaner
```

---

## Project Structure

```
cleansly/
├── cleansly/
│   ├── __init__.py              # Public API exports
│   ├── pipeline.py              # CleaningPipeline, PipelineReport
│   ├── exceptions.py            # Custom exception hierarchy
│   ├── cleaners/
│   │   ├── text.py              # TextCleaner
│   │   ├── numeric.py           # NumericCleaner
│   │   ├── datetime_cleaner.py  # DateTimeCleaner
│   │   └── missing.py           # MissingValueHandler
│   ├── validators/
│   │   ├── schema.py            # SchemaValidator, ColumnSchema
│   │   └── rules.py             # RuleValidator, Rule
│   ├── transformers/
│   │   ├── standardizer.py      # Standardizer
│   │   └── encoder.py           # Encoder
│   └── utils/
│       ├── logger.py            # Structured logging
│       └── profiler.py          # DataProfiler
├── tests/
│   └── test_cleansly.py        # 46 unit tests
├── examples/
│   └── end_to_end_example.py    # Full enterprise example
└── pyproject.toml
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
