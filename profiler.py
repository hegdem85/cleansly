"""Data profiling utilities for pre- and post-cleaning analysis."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any
from dataclasses import dataclass, field, asdict


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    total_count: int
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    # Numeric stats
    mean: float = None
    std: float = None
    min: float = None
    max: float = None
    q25: float = None
    q50: float = None
    q75: float = None
    # Text stats
    avg_length: float = None
    max_length: int = None
    min_length: int = None
    # Top values
    top_values: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataProfile:
    row_count: int
    col_count: int
    total_cells: int
    total_nulls: int
    null_pct: float
    duplicate_rows: int
    duplicate_pct: float
    memory_usage_mb: float
    columns: dict[str, ColumnProfile] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "DATA PROFILE SUMMARY",
            "=" * 60,
            f"  Rows            : {self.row_count:,}",
            f"  Columns         : {self.col_count}",
            f"  Total Nulls     : {self.total_nulls:,} ({self.null_pct:.2f}%)",
            f"  Duplicate Rows  : {self.duplicate_rows:,} ({self.duplicate_pct:.2f}%)",
            f"  Memory Usage    : {self.memory_usage_mb:.2f} MB",
            "-" * 60,
            "COLUMN DETAILS",
            "-" * 60,
        ]
        for col, profile in self.columns.items():
            lines.append(
                f"  {col:<30} dtype={profile.dtype:<12} "
                f"nulls={profile.null_pct:.1f}%  unique={profile.unique_pct:.1f}%"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class DataProfiler:
    """
    Profiles a DataFrame to understand data quality before and after cleaning.

    Example::

        profiler = DataProfiler()
        profile = profiler.profile(df)
        print(profile.summary())
    """

    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def profile(self, df: pd.DataFrame) -> DataProfile:
        """Generate a full data profile for the given DataFrame."""
        row_count, col_count = df.shape
        total_cells = row_count * col_count
        total_nulls = int(df.isnull().sum().sum())
        null_pct = round(total_nulls / total_cells * 100, 4) if total_cells else 0.0
        duplicate_rows = int(df.duplicated().sum())
        duplicate_pct = round(duplicate_rows / row_count * 100, 4) if row_count else 0.0
        memory_usage_mb = round(df.memory_usage(deep=True).sum() / 1_048_576, 4)

        columns = {}
        for col in df.columns:
            columns[col] = self._profile_column(df[col])

        return DataProfile(
            row_count=row_count,
            col_count=col_count,
            total_cells=total_cells,
            total_nulls=total_nulls,
            null_pct=null_pct,
            duplicate_rows=duplicate_rows,
            duplicate_pct=duplicate_pct,
            memory_usage_mb=memory_usage_mb,
            columns=columns,
        )

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        total = len(series)
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / total * 100, 4) if total else 0.0
        non_null = series.dropna()
        unique_count = int(series.nunique())
        unique_pct = round(unique_count / total * 100, 4) if total else 0.0
        top_values = (
            series.value_counts().head(self.top_n).to_dict()
        )

        profile = ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            total_count=total,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
            top_values=[{"value": str(k), "count": int(v)} for k, v in top_values.items()],
        )

        if pd.api.types.is_numeric_dtype(series):
            if len(non_null) > 0:
                profile.mean = round(float(non_null.mean()), 6)
                profile.std = round(float(non_null.std()), 6)
                profile.min = float(non_null.min())
                profile.max = float(non_null.max())
                profile.q25 = float(non_null.quantile(0.25))
                profile.q50 = float(non_null.quantile(0.50))
                profile.q75 = float(non_null.quantile(0.75))

        elif pd.api.types.is_string_dtype(series) or series.dtype == object:
            str_series = non_null.astype(str)
            lengths = str_series.str.len()
            if len(lengths) > 0:
                profile.avg_length = round(float(lengths.mean()), 2)
                profile.max_length = int(lengths.max())
                profile.min_length = int(lengths.min())

        return profile
