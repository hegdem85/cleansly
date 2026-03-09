"""
end_to_end_example.py
=====================
Full enterprise data cleansing example using the Cleansly library.
"""

import pandas as pd
import numpy as np
from cleansly import (
    CleaningPipeline,
    TextCleaner,
    NumericCleaner,
    DateTimeCleaner,
    MissingValueHandler,
    SchemaValidator,
    RuleValidator,
    Standardizer,
    Encoder,
    DataProfiler,
)
from cleansly.validators.schema import ColumnSchema
from cleansly.validators.rules import Rule

# ── 1. Create sample "raw" enterprise data ───────────────────────────────────
raw_data = {
    "CustomerID": [1, 2, 2, 3, 4, None, 6],
    "First Name": ["  Alice ", "BOB", "BOB", "charlie", None, "Eve", "frank"],
    "Email": [
        "alice@example.com",
        "bob@EXAMPLE.COM",
        "bob@EXAMPLE.COM",
        "not-an-email",
        "dave@company.org",
        "eve@example.com",
        None,
    ],
    "Age": [29, 35, 35, -5, 999, 41, 22],
    "Revenue": ["$1,200.50", "$850", "$850", "$3000", None, "$720.00", "$480"],
    "SignupDate": [
        "2022-01-15", "01/22/2021", "01/22/2021",
        "2099-12-31", "2020-07-04", "March 5 2023", "2021-11-30"
    ],
    "Status": ["active", "ACTIVE", "active", "inactive", "pending", "INACTIVE", "active"],
    "Score": [88.5, None, None, 72.0, 95.0, 60.0, np.nan],
    "Notes": ["<b>VIP</b> customer", "  Regular  ", None, "New client", "", "Referral", None],
}

df_raw = pd.DataFrame(raw_data)
print("=== RAW DATA ===")
print(df_raw.to_string())
print()

# ── 2. Profile raw data ───────────────────────────────────────────────────────
profiler = DataProfiler()
profile = profiler.profile(df_raw)
print(profile.summary())

# ── 3. Pre-process: parse revenue strings → numeric ──────────────────────────
def parse_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Revenue"] = (
        df["Revenue"].str.replace(r"[$,]", "", regex=True).astype(float)
    )
    return df

# ── 4. Define schema ──────────────────────────────────────────────────────────
schema = {
    "customer_id": ColumnSchema(dtype="float64", nullable=False),
    "first_name":  ColumnSchema(nullable=False, min_length=2, max_length=50),
    "email":       ColumnSchema(regex=r"^[^@]+@[^@]+\.[^@]+$"),
    "age":         ColumnSchema(min_value=0, max_value=120),
    "revenue":     ColumnSchema(min_value=0),
    "status":      ColumnSchema(allowed_values=["active", "inactive", "pending"]),
    "score":       ColumnSchema(min_value=0, max_value=100),
}

# ── 5. Define business rules ──────────────────────────────────────────────────
rules = [
    Rule(
        name="score_range",
        condition=lambda row: 0 <= row["score"] <= 100 if pd.notna(row["score"]) else True,
        message="Score must be between 0 and 100.",
    ),
    Rule(
        name="revenue_positive",
        condition=lambda row: row["revenue"] >= 0 if pd.notna(row["revenue"]) else True,
        message="Revenue must be non-negative.",
    ),
    Rule(
        name="total_revenue_nonzero",
        condition=lambda df: df["revenue"].sum() > 0,
        is_row_level=False,
        severity="warning",
        message="Total revenue is zero — possible ingestion issue.",
    ),
]

# ── 6. Build & run pipeline ───────────────────────────────────────────────────
pipeline = (
    CleaningPipeline(name="CustomerDataPipeline", profile=True)
    .add_step("parse_revenue", parse_revenue)
    .add_step(
        "standardize",
        Standardizer(
            column_name_style="snake_case",
            drop_duplicates=True,
            duplicate_subset=["CustomerID"],
        ),
    )
    .add_step(
        "clean_text",
        TextCleaner(
            strip_whitespace=True,
            normalize_case="lower",
            remove_html=True,
            null_if_empty=True,
            columns=["first_name", "notes", "status"],
        ),
    )
    .add_step(
        "handle_missing",
        MissingValueHandler(
            strategy="fill_median",
            column_strategies={
                "status": "fill_mode",
                "notes": "fill_constant",
                "first_name": "fill_constant",
            },
            fill_values={"notes": "N/A", "first_name": "Unknown"},
        ),
    )
    .add_step(
        "clean_numerics",
        NumericCleaner(
            coerce=True,
            outlier_method="iqr",
            outlier_action="clip",
            clip_lower=0,
            round_decimals=2,
            columns=["age", "revenue", "score"],
        ),
    )
    .add_step(
        "parse_dates",
        DateTimeCleaner(
            input_formats=["%Y-%m-%d", "%d/%m/%Y"],
            output_format="%Y-%m-%d",
            min_date="2000-01-01",
            max_date="2030-12-31",
            extract_components=False,
            columns=["signup_date"],
        ),
    )
    .add_step(
        "encode",
        Encoder(label_columns=["status"]),
    )
    .add_step(
        "validate_schema",
        SchemaValidator(schema, on_error="warn"),
    )
    .add_step(
        "validate_rules",
        RuleValidator(rules, on_error="warn"),
    )
)

df_clean, report = pipeline.run(df_raw)

# ── 7. Display results ────────────────────────────────────────────────────────
print("\n=== CLEANED DATA ===")
print(df_clean.to_string())
print()
print(report.summary())
