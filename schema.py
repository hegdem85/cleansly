"""Schema validation for DataFrames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from ..exceptions import SchemaError, ValidationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""

    dtype: Optional[str] = None          # Expected pandas dtype string
    nullable: bool = True
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    regex: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    unique: bool = False


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = ["VALIDATION RESULT: " + ("PASS" if self.is_valid else "FAIL")]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    ✗ {e}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)


class SchemaValidator:
    """
    Validates a DataFrame against a declared schema.

    Example::

        schema = {
            "user_id": ColumnSchema(dtype="int64", nullable=False, unique=True),
            "email":   ColumnSchema(nullable=False, regex=r"^[^@]+@[^@]+\.[^@]+$"),
            "age":     ColumnSchema(dtype="float64", min_value=0, max_value=150),
            "status":  ColumnSchema(allowed_values=["active", "inactive", "pending"]),
        }
        validator = SchemaValidator(schema, on_error="warn")
        result = validator.validate(df)
        print(result.summary())
    """

    def __init__(
        self,
        schema: Dict[str, ColumnSchema],
        require_all_columns: bool = True,
        allow_extra_columns: bool = True,
        on_error: str = "warn",  # "warn", "raise", "return"
    ):
        self.schema = schema
        self.require_all_columns = require_all_columns
        self.allow_extra_columns = allow_extra_columns
        self.on_error = on_error

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate the DataFrame against the schema."""
        result = ValidationResult(is_valid=True)

        # Check required columns
        if self.require_all_columns:
            for col in self.schema:
                if col not in df.columns:
                    result.add_error(f"Required column '{col}' is missing.")

        # Check for unexpected extra columns
        if not self.allow_extra_columns:
            extra = set(df.columns) - set(self.schema.keys())
            if extra:
                result.add_warning(f"Unexpected columns found: {sorted(extra)}")

        # Per-column checks
        for col, col_schema in self.schema.items():
            if col not in df.columns:
                continue
            self._validate_column(df[col], col_schema, result)

        if not result.is_valid:
            msg = f"Schema validation failed with {len(result.errors)} error(s)."
            if self.on_error == "raise":
                raise SchemaError(msg)
            elif self.on_error == "warn":
                logger.warning(msg)

        return result

    def _validate_column(
        self, series: pd.Series, schema: ColumnSchema, result: ValidationResult
    ):
        col = series.name

        # Dtype check
        if schema.dtype:
            try:
                expected = pd.api.types.pandas_dtype(schema.dtype)
                if not pd.api.types.is_dtype_equal(series.dtype, expected):
                    result.add_error(
                        f"Column '{col}': expected dtype '{schema.dtype}', "
                        f"got '{series.dtype}'."
                    )
            except Exception:
                result.add_warning(f"Column '{col}': could not compare dtype '{schema.dtype}'.")

        # Nullability
        null_count = series.isnull().sum()
        if not schema.nullable and null_count > 0:
            result.add_error(
                f"Column '{col}': contains {null_count} null value(s) but is not nullable."
            )

        non_null = series.dropna()

        # Range checks
        if schema.min_value is not None:
            violations = (non_null < schema.min_value).sum()
            if violations:
                result.add_error(
                    f"Column '{col}': {violations} value(s) below minimum {schema.min_value}."
                )

        if schema.max_value is not None:
            violations = (non_null > schema.max_value).sum()
            if violations:
                result.add_error(
                    f"Column '{col}': {violations} value(s) above maximum {schema.max_value}."
                )

        # Allowed values
        if schema.allowed_values is not None:
            invalid = ~non_null.isin(schema.allowed_values)
            if invalid.any():
                bad = non_null[invalid].unique()[:5]
                result.add_error(
                    f"Column '{col}': {invalid.sum()} value(s) not in allowed set. "
                    f"Examples: {list(bad)}"
                )

        # Regex
        if schema.regex:
            import re
            str_series = non_null.astype(str)
            mismatches = ~str_series.str.match(schema.regex)
            if mismatches.any():
                result.add_error(
                    f"Column '{col}': {mismatches.sum()} value(s) don't match regex '{schema.regex}'."
                )

        # String length
        if schema.min_length is not None or schema.max_length is not None:
            str_series = non_null.astype(str)
            lengths = str_series.str.len()
            if schema.min_length is not None:
                too_short = (lengths < schema.min_length).sum()
                if too_short:
                    result.add_error(
                        f"Column '{col}': {too_short} value(s) shorter than {schema.min_length} chars."
                    )
            if schema.max_length is not None:
                too_long = (lengths > schema.max_length).sum()
                if too_long:
                    result.add_error(
                        f"Column '{col}': {too_long} value(s) longer than {schema.max_length} chars."
                    )

        # Uniqueness
        if schema.unique:
            dupes = series.duplicated().sum()
            if dupes:
                result.add_error(
                    f"Column '{col}': expected unique values but found {dupes} duplicate(s)."
                )
