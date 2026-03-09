"""Numeric data cleaning operations."""

from __future__ import annotations

from typing import Optional, List, Union

import numpy as np
import pandas as pd

from ..exceptions import TransformationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class NumericCleaner:
    """
    Enterprise-grade numeric data cleaner.

    Handles coercion, outlier treatment, range clamping,
    precision rounding, and scaling.

    Example::

        cleaner = NumericCleaner(
            coerce=True,
            clip_lower=0,
            clip_upper=1_000_000,
            outlier_method="iqr",
            round_decimals=2,
        )
        df = cleaner.clean(df, columns=["revenue", "age"])
    """

    OUTLIER_METHODS = ("iqr", "zscore", "none")

    def __init__(
        self,
        coerce: bool = True,
        clip_lower: Optional[float] = None,
        clip_upper: Optional[float] = None,
        outlier_method: str = "none",  # "iqr", "zscore", "none"
        outlier_action: str = "clip",  # "clip" or "null"
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        round_decimals: Optional[int] = None,
        fill_value: Optional[float] = None,  # Fill nulls after cleaning
        errors: str = "coerce",
    ):
        if outlier_method not in self.OUTLIER_METHODS:
            raise ValueError(f"outlier_method must be one of {self.OUTLIER_METHODS}")
        self.coerce = coerce
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        self.outlier_method = outlier_method
        self.outlier_action = outlier_action
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.round_decimals = round_decimals
        self.fill_value = fill_value
        self.errors = errors

    def clean(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Clean numeric columns in the DataFrame."""
        df = df.copy()
        target_cols = columns or df.select_dtypes(include="number").columns.tolist()

        for col in target_cols:
            if col not in df.columns:
                logger.warning("Column '%s' not found — skipping.", col)
                continue
            try:
                df[col] = self._clean_series(df[col])
                logger.debug("Cleaned numeric column '%s'.", col)
            except Exception as exc:
                msg = f"Failed to clean numeric column '{col}': {exc}"
                if self.errors == "raise":
                    raise TransformationError(msg, step="NumericCleaner", column=col) from exc
                logger.warning(msg)

        return df

    def _clean_series(self, series: pd.Series) -> pd.Series:
        if self.coerce:
            series = pd.to_numeric(series, errors="coerce")

        if self.clip_lower is not None or self.clip_upper is not None:
            series = series.clip(lower=self.clip_lower, upper=self.clip_upper)

        if self.outlier_method == "iqr":
            series = self._handle_iqr_outliers(series)
        elif self.outlier_method == "zscore":
            series = self._handle_zscore_outliers(series)

        if self.round_decimals is not None:
            series = series.round(self.round_decimals)

        if self.fill_value is not None:
            series = series.fillna(self.fill_value)

        return series

    def _handle_iqr_outliers(self, series: pd.Series) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr
        return self._apply_outlier_action(series, lower, upper)

    def _handle_zscore_outliers(self, series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std()
        if std == 0:
            return series
        z = (series - mean) / std
        mask = z.abs() > self.zscore_threshold
        lower = mean - self.zscore_threshold * std
        upper = mean + self.zscore_threshold * std
        return self._apply_outlier_action(series, lower, upper)

    def _apply_outlier_action(
        self, series: pd.Series, lower: float, upper: float
    ) -> pd.Series:
        if self.outlier_action == "clip":
            return series.clip(lower=lower, upper=upper)
        elif self.outlier_action == "null":
            return series.where((series >= lower) & (series <= upper), other=np.nan)
        return series
