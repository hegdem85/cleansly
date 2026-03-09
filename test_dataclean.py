"""
Tests for Cleansly library.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from cleansly import (
    CleaningPipeline,
    DateTimeCleaner,
    MissingValueHandler,
    NumericCleaner,
    SchemaValidator,
    RuleValidator,
    Standardizer,
    Encoder,
    TextCleaner,
)
from cleansly.validators.schema import ColumnSchema
from cleansly.validators.rules import Rule
from cleansly.utils.profiler import DataProfiler
from cleansly.exceptions import TransformationError, SchemaError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "name": ["  Alice ", "BOB", None, " charlie "],
        "age": [29, -5, 999, 35],
        "revenue": [1200.0, 850.0, None, 3000.0],
        "email": ["alice@example.com", "notanemail", "bob@test.com", None],
        "status": ["active", "ACTIVE", "inactive", "pending"],
        "signup_date": ["2022-01-15", "01/22/2021", "2099-12-31", "2020-07-04"],
        "score": [88.5, None, 72.0, 95.0],
    })


# ── TextCleaner ───────────────────────────────────────────────────────────────

class TestTextCleaner:
    def test_strip_whitespace(self):
        df = pd.DataFrame({"name": ["  Alice ", " Bob "]})
        cleaner = TextCleaner(strip_whitespace=True)
        result = cleaner.clean(df, columns=["name"])
        assert result["name"].tolist() == ["Alice", "Bob"]

    def test_normalize_case_lower(self):
        df = pd.DataFrame({"name": ["Alice", "BOB"]})
        cleaner = TextCleaner(normalize_case="lower")
        result = cleaner.clean(df, columns=["name"])
        assert result["name"].tolist() == ["alice", "bob"]

    def test_remove_html(self):
        df = pd.DataFrame({"notes": ["<b>VIP</b>", "<p>Regular</p>"]})
        cleaner = TextCleaner(remove_html=True)
        result = cleaner.clean(df, columns=["notes"])
        assert "VIP" in result["notes"][0]
        assert "<b>" not in result["notes"][0]

    def test_redact_email(self):
        df = pd.DataFrame({"notes": ["Contact alice@example.com for info"]})
        cleaner = TextCleaner(redact_emails=True, redact_placeholder="[EMAIL]")
        result = cleaner.clean(df, columns=["notes"])
        assert "[EMAIL]" in result["notes"][0]
        assert "alice@example.com" not in result["notes"][0]

    def test_null_if_empty(self):
        df = pd.DataFrame({"name": ["", "  ", "Alice"]})
        cleaner = TextCleaner(strip_whitespace=True, null_if_empty=True)
        result = cleaner.clean(df, columns=["name"])
        assert pd.isna(result["name"][0])
        assert pd.isna(result["name"][1])
        assert result["name"][2] == "Alice"

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"name": ["Alice"]})
        cleaner = TextCleaner()
        result = cleaner.clean(df, columns=["nonexistent"])
        assert "name" in result.columns  # original df unchanged


# ── NumericCleaner ────────────────────────────────────────────────────────────

class TestNumericCleaner:
    def test_coerce_string_numerics(self):
        df = pd.DataFrame({"val": ["1", "2", "abc", "4"]})
        cleaner = NumericCleaner(coerce=True)
        result = cleaner.clean(df, columns=["val"])
        assert result["val"][2] != result["val"][2]  # NaN

    def test_clip_range(self):
        df = pd.DataFrame({"age": [-5, 29, 999, 35]})
        cleaner = NumericCleaner(clip_lower=0, clip_upper=120)
        result = cleaner.clean(df, columns=["age"])
        assert result["age"].min() >= 0
        assert result["age"].max() <= 120

    def test_iqr_outlier_clip(self):
        data = [10, 12, 11, 13, 10, 12, 1000]  # 1000 is outlier
        df = pd.DataFrame({"val": data})
        cleaner = NumericCleaner(outlier_method="iqr", outlier_action="clip")
        result = cleaner.clean(df, columns=["val"])
        assert result["val"].max() < 1000

    def test_iqr_outlier_null(self):
        data = [10, 12, 11, 13, 10, 12, 1000]
        df = pd.DataFrame({"val": data})
        cleaner = NumericCleaner(outlier_method="iqr", outlier_action="null")
        result = cleaner.clean(df, columns=["val"])
        assert result["val"].isnull().any()

    def test_round_decimals(self):
        df = pd.DataFrame({"val": [1.23456, 2.98765]})
        cleaner = NumericCleaner(round_decimals=2)
        result = cleaner.clean(df, columns=["val"])
        assert result["val"][0] == pytest.approx(1.23)

    def test_fill_nulls(self):
        df = pd.DataFrame({"val": [1.0, None, 3.0]})
        cleaner = NumericCleaner(fill_value=0.0)
        result = cleaner.clean(df, columns=["val"])
        assert result["val"].isnull().sum() == 0


# ── DateTimeCleaner ───────────────────────────────────────────────────────────

class TestDateTimeCleaner:
    def test_parse_iso_format(self):
        df = pd.DataFrame({"dt": ["2022-01-15", "2021-06-30"]})
        cleaner = DateTimeCleaner()
        result = cleaner.clean(df, columns=["dt"])
        assert pd.api.types.is_datetime64_any_dtype(result["dt"])

    def test_output_format(self):
        df = pd.DataFrame({"dt": ["2022-01-15", "01/22/2021"]})
        cleaner = DateTimeCleaner(output_format="%Y-%m-%d")
        result = cleaner.clean(df, columns=["dt"])
        assert result["dt"][0] == "2022-01-15"

    def test_date_range_nullification(self):
        df = pd.DataFrame({"dt": ["2022-01-15", "2099-12-31", "2020-07-04"]})
        cleaner = DateTimeCleaner(max_date="2030-12-31", null_out_of_range=True)
        result = cleaner.clean(df, columns=["dt"])
        assert pd.isna(result["dt"][1])

    def test_extract_components(self):
        df = pd.DataFrame({"dt": ["2022-03-15"]})
        cleaner = DateTimeCleaner(extract_components=True)
        result = cleaner.clean(df, columns=["dt"])
        assert "dt_year" in result.columns
        assert result["dt_year"][0] == 2022
        assert result["dt_month"][0] == 3


# ── MissingValueHandler ───────────────────────────────────────────────────────

class TestMissingValueHandler:
    def test_fill_mean(self):
        df = pd.DataFrame({"val": [1.0, None, 3.0, None]})
        handler = MissingValueHandler(strategy="fill_mean")
        result = handler.clean(df, columns=["val"])
        assert result["val"].isnull().sum() == 0
        assert result["val"][1] == pytest.approx(2.0)

    def test_fill_median(self):
        df = pd.DataFrame({"val": [1.0, None, 3.0, 5.0]})
        handler = MissingValueHandler(strategy="fill_median")
        result = handler.clean(df, columns=["val"])
        assert result["val"].isnull().sum() == 0

    def test_fill_constant(self):
        df = pd.DataFrame({"notes": ["hello", None, None]})
        handler = MissingValueHandler(
            strategy="fill_constant",
            fill_values={"notes": "N/A"},
        )
        result = handler.clean(df, columns=["notes"])
        assert result["notes"][1] == "N/A"

    def test_fill_forward(self):
        df = pd.DataFrame({"val": [1.0, None, None, 4.0]})
        handler = MissingValueHandler(strategy="fill_forward")
        result = handler.clean(df, columns=["val"])
        assert result["val"][1] == 1.0

    def test_drop_high_null_columns(self):
        df = pd.DataFrame({
            "good": [1, 2, 3, 4],
            "mostly_null": [None, None, None, 1],
        })
        handler = MissingValueHandler(strategy="fill_mean", drop_threshold=0.6)
        result = handler.clean(df)
        assert "mostly_null" not in result.columns

    def test_column_strategy_override(self):
        df = pd.DataFrame({
            "a": [1.0, None],
            "b": ["x", None],
        })
        handler = MissingValueHandler(
            strategy="fill_mean",
            column_strategies={"b": "fill_constant"},
            fill_values={"b": "unknown"},
        )
        result = handler.clean(df)
        assert result["b"][1] == "unknown"


# ── Standardizer ──────────────────────────────────────────────────────────────

class TestStandardizer:
    def test_rename_columns(self):
        df = pd.DataFrame({"cust_id": [1, 2], "rev": [100, 200]})
        std = Standardizer(rename_columns={"cust_id": "customer_id", "rev": "revenue"})
        result = std.transform(df)
        assert "customer_id" in result.columns
        assert "revenue" in result.columns

    def test_snake_case_columns(self):
        df = pd.DataFrame({"FirstName": [1], "LastName": [2], "CustomerID": [3]})
        std = Standardizer(column_name_style="snake_case")
        result = std.transform(df)
        assert "first_name" in result.columns
        assert "last_name" in result.columns

    def test_drop_duplicates(self):
        df = pd.DataFrame({"id": [1, 1, 2], "val": ["a", "a", "b"]})
        std = Standardizer(drop_duplicates=True, duplicate_subset=["id"])
        result = std.transform(df)
        assert len(result) == 2

    def test_value_map(self):
        df = pd.DataFrame({"flag": ["Y", "N", "Y"]})
        std = Standardizer(value_maps={"flag": {"Y": True, "N": False}})
        result = std.transform(df)
        assert result["flag"].tolist() == [True, False, True]

    def test_dtype_cast(self):
        df = pd.DataFrame({"val": ["1", "2", "3"]})
        std = Standardizer(dtype_map={"val": "int64"})
        result = std.transform(df)
        assert result["val"].dtype == "int64"


# ── Encoder ───────────────────────────────────────────────────────────────────

class TestEncoder:
    def test_label_encode(self):
        df = pd.DataFrame({"status": ["active", "inactive", "pending", "active"]})
        encoder = Encoder(label_columns=["status"])
        result = encoder.transform(df)
        assert pd.api.types.is_numeric_dtype(result["status"])
        assert "status" in encoder.label_mappings_

    def test_label_inverse_transform(self):
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})
        encoder = Encoder(label_columns=["status"])
        encoded = encoder.transform(df)
        decoded = encoder.inverse_transform_labels(encoded, columns=["status"])
        assert decoded["status"].tolist() == ["active", "inactive", "active"]

    def test_onehot_encode(self):
        df = pd.DataFrame({"category": ["A", "B", "A", "C"]})
        encoder = Encoder(onehot_columns=["category"])
        result = encoder.transform(df)
        assert "category_A" in result.columns or any(
            c.startswith("category_") for c in result.columns
        )


# ── SchemaValidator ───────────────────────────────────────────────────────────

class TestSchemaValidator:
    def test_passes_valid_data(self):
        df = pd.DataFrame({"age": [25, 30, 45], "name": ["a", "b", "c"]})
        schema = {"age": ColumnSchema(min_value=0, max_value=120)}
        validator = SchemaValidator(schema, require_all_columns=False)
        result = validator.validate(df)
        assert result.is_valid

    def test_fails_null_in_nonnullable(self):
        df = pd.DataFrame({"id": [1, None, 3]})
        schema = {"id": ColumnSchema(nullable=False)}
        validator = SchemaValidator(schema, on_error="return")
        result = validator.validate(df)
        assert not result.is_valid
        assert any("null" in e.lower() for e in result.errors)

    def test_fails_out_of_range(self):
        df = pd.DataFrame({"age": [25, -5, 200]})
        schema = {"age": ColumnSchema(min_value=0, max_value=150)}
        validator = SchemaValidator(schema, on_error="return")
        result = validator.validate(df)
        assert not result.is_valid

    def test_fails_disallowed_values(self):
        df = pd.DataFrame({"status": ["active", "unknown", "inactive"]})
        schema = {"status": ColumnSchema(allowed_values=["active", "inactive", "pending"])}
        validator = SchemaValidator(schema, on_error="return")
        result = validator.validate(df)
        assert not result.is_valid

    def test_raises_on_error(self):
        df = pd.DataFrame({"id": [1, None]})
        schema = {"id": ColumnSchema(nullable=False)}
        validator = SchemaValidator(schema, on_error="raise")
        with pytest.raises(SchemaError):
            validator.validate(df)


# ── RuleValidator ─────────────────────────────────────────────────────────────

class TestRuleValidator:
    def test_passes_valid_rules(self):
        df = pd.DataFrame({"age": [25, 30], "score": [80, 90]})
        rules = [Rule("age_pos", lambda row: row["age"] > 0)]
        validator = RuleValidator(rules)
        result = validator.validate(df)
        assert result.is_valid

    def test_fails_violated_row_rule(self):
        df = pd.DataFrame({"age": [25, -5]})
        rules = [Rule("age_pos", lambda row: row["age"] > 0, message="Age must be positive.")]
        validator = RuleValidator(rules)
        result = validator.validate(df)
        assert not result.is_valid

    def test_frame_level_rule_warning(self):
        df = pd.DataFrame({"revenue": [0, 0, 0]})
        rules = [Rule(
            "nonzero_rev",
            lambda df: df["revenue"].sum() > 0,
            is_row_level=False,
            severity="warning",
            message="Revenue is zero.",
        )]
        validator = RuleValidator(rules)
        result = validator.validate(df)
        assert result.is_valid  # warnings don't fail
        assert len(result.warnings) > 0


# ── DataProfiler ──────────────────────────────────────────────────────────────

class TestDataProfiler:
    def test_profile_shape(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df)
        assert profile.row_count == len(sample_df)
        assert profile.col_count == len(sample_df.columns)

    def test_profile_nulls(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        profiler = DataProfiler()
        profile = profiler.profile(df)
        assert profile.columns["a"].null_count == 1
        assert profile.columns["a"].null_pct == pytest.approx(100 / 3, rel=1e-3)

    def test_profile_summary_output(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df)
        summary = profile.summary()
        assert "DATA PROFILE SUMMARY" in summary


# ── CleaningPipeline ──────────────────────────────────────────────────────────

class TestCleaningPipeline:
    def test_pipeline_runs_and_returns_df(self, sample_df):
        pipeline = (
            CleaningPipeline(name="test")
            .add_step("clean_text", TextCleaner(strip_whitespace=True))
            .add_step("handle_missing", MissingValueHandler(strategy="fill_median"))
        )
        result, report = pipeline.run(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(report.steps) == 2

    def test_pipeline_with_profile(self, sample_df):
        pipeline = CleaningPipeline(profile=True).add_step(
            "standardize", Standardizer()
        )
        _, report = pipeline.run(sample_df)
        assert report.profile_before is not None
        assert report.profile_after is not None

    def test_pipeline_custom_function(self, sample_df):
        def add_flag(df):
            df = df.copy()
            df["cleaned"] = True
            return df

        pipeline = CleaningPipeline().add_step("add_flag", add_flag)
        result, _ = pipeline.run(sample_df)
        assert "cleaned" in result.columns

    def test_pipeline_report_summary(self, sample_df):
        pipeline = CleaningPipeline().add_step(
            "numeric", NumericCleaner(clip_lower=0)
        )
        _, report = pipeline.run(sample_df)
        summary = report.summary()
        assert "PIPELINE EXECUTION REPORT" in summary

    def test_step_chaining_returns_self(self, sample_df):
        pipeline = CleaningPipeline()
        returned = pipeline.add_step("step1", TextCleaner())
        assert returned is pipeline
