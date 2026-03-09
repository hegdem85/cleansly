"""Missing value detection and imputation strategies."""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd

from ..exceptions import TransformationError
from ..utils.logger import get_logger

logger = get_logger(__name__)

STRATEGIES = (
    "drop_rows",
    "drop_cols",
    "fill_constant",
    "fill_mean",
    "fill_median",
    "fill_mode",
    "fill_forward",
    "fill_backward",
    "fill_interpolate",
)


class MissingValueHandler:
    """
    Flexible missing value handler supporting multiple imputation strategies.

    Per-column strategy overrides are supported for fine-grained control.

    Example::

        handler = MissingValueHandler(
            strategy="fill_median",
            column_strategies={"status": "fill_mode", "notes": "fill_constant"},
            fill_values={"notes": "N/A"},
            drop_threshold=0.8,  # Drop columns with >80% missing
        )
        df = handler.clean(df)
    """

    def __init__(
        self,
        strategy: str = "fill_mean",
        column_strategies: Optional[Dict[str, str]] = None,
        fill_values: Optional[Dict[str, Any]] = None,
        drop_threshold: Optional[float] = None,  # 0.0–1.0, drop cols above this
        row_drop_threshold: Optional[float] = None,  # 0.0–1.0, drop rows above this
        interpolate_method: str = "linear",
        errors: str = "coerce",
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.strategy = strategy
        self.column_strategies = column_strategies or {}
        self.fill_values = fill_values or {}
        self.drop_threshold = drop_threshold
        self.row_drop_threshold = row_drop_threshold
        self.interpolate_method = interpolate_method
        self.errors = errors

    def clean(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        df = df.copy()

        # Drop high-null columns first
        if self.drop_threshold is not None:
            null_ratio = df.isnull().mean()
            cols_to_drop = null_ratio[null_ratio > self.drop_threshold].index.tolist()
            if cols_to_drop:
                logger.info(
                    "Dropping %d column(s) exceeding null threshold %.0f%%: %s",
                    len(cols_to_drop),
                    self.drop_threshold * 100,
                    cols_to_drop,
                )
                df = df.drop(columns=cols_to_drop)

        # Drop high-null rows
        if self.row_drop_threshold is not None:
            row_null_ratio = df.isnull().mean(axis=1)
            rows_to_drop = row_null_ratio[row_null_ratio > self.row_drop_threshold].index
            if len(rows_to_drop):
                logger.info("Dropping %d row(s) exceeding row null threshold.", len(rows_to_drop))
                df = df.drop(index=rows_to_drop)

        target_cols = columns or df.columns.tolist()

        for col in target_cols:
            if col not in df.columns:
                continue
            strategy = self.column_strategies.get(col, self.strategy)
            fill_val = self.fill_values.get(col)
            try:
                df[col] = self._apply_strategy(df[col], strategy, fill_val)
            except Exception as exc:
                msg = f"Missing value handling failed for '{col}': {exc}"
                if self.errors == "raise":
                    raise TransformationError(msg, step="MissingValueHandler", column=col) from exc
                logger.warning(msg)

        return df

    def _apply_strategy(
        self, series: pd.Series, strategy: str, fill_val: Any
    ) -> pd.Series:
        if strategy == "drop_rows":
            # Can't drop rows per-series; caller must handle
            return series

        if strategy == "fill_constant":
            val = fill_val if fill_val is not None else 0
            return series.fillna(val)

        if strategy == "fill_mean":
            if pd.api.types.is_numeric_dtype(series):
                return series.fillna(series.mean())
            return series.fillna(series.mode().iloc[0] if not series.mode().empty else None)

        if strategy == "fill_median":
            if pd.api.types.is_numeric_dtype(series):
                return series.fillna(series.median())
            return series

        if strategy == "fill_mode":
            mode = series.mode()
            if not mode.empty:
                return series.fillna(mode.iloc[0])
            return series

        if strategy == "fill_forward":
            return series.ffill()

        if strategy == "fill_backward":
            return series.bfill()

        if strategy == "fill_interpolate":
            if pd.api.types.is_numeric_dtype(series):
                return series.interpolate(method=self.interpolate_method)
            return series.ffill()

        return series
