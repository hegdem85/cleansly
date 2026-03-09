"""Date and time cleaning operations."""

from __future__ import annotations

from typing import Optional, List, Union

import pandas as pd

from ..exceptions import TransformationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DateTimeCleaner:
    """
    Parses, validates, and normalizes datetime columns.

    Supports multiple input formats, timezone handling,
    range validation, and component extraction.

    Example::

        cleaner = DateTimeCleaner(
            input_formats=["%Y-%m-%d", "%d/%m/%Y"],
            output_format="%Y-%m-%d",
            timezone="UTC",
            min_date="2000-01-01",
            max_date="2030-12-31",
            extract_components=True,
        )
        df = cleaner.clean(df, columns=["created_at", "updated_at"])
    """

    def __init__(
        self,
        input_formats: Optional[List[str]] = None,
        output_format: Optional[str] = None,
        timezone: Optional[str] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        null_out_of_range: bool = True,
        extract_components: bool = False,
        dayfirst: bool = False,
        yearfirst: bool = False,
        errors: str = "coerce",
    ):
        self.input_formats = input_formats or []
        self.output_format = output_format
        self.timezone = timezone
        self.min_date = pd.Timestamp(min_date) if min_date else None
        self.max_date = pd.Timestamp(max_date) if max_date else None
        self.null_out_of_range = null_out_of_range
        self.extract_components = extract_components
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self.errors = errors

    def clean(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Parse and clean datetime columns."""
        df = df.copy()
        target_cols = columns or df.select_dtypes(
            include=["datetime", "datetimetz", "object"]
        ).columns.tolist()

        for col in target_cols:
            if col not in df.columns:
                logger.warning("Column '%s' not found — skipping.", col)
                continue
            try:
                df = self._clean_column(df, col)
                logger.debug("Cleaned datetime column '%s'.", col)
            except Exception as exc:
                msg = f"Failed to clean datetime column '{col}': {exc}"
                if self.errors == "raise":
                    raise TransformationError(msg, step="DateTimeCleaner", column=col) from exc
                logger.warning(msg)

        return df

    def _clean_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        series = df[col]

        # Try format-by-format first for precision
        parsed = None
        if self.input_formats:
            for fmt in self.input_formats:
                try:
                    parsed = pd.to_datetime(series, format=fmt, errors="coerce")
                    # Use this format if it parsed at least some values
                    if parsed.notna().any():
                        break
                except Exception:
                    continue

        if parsed is None:
            parsed = pd.to_datetime(
                series,
                errors="coerce",
                dayfirst=self.dayfirst,
                yearfirst=self.yearfirst,
            )

        # Timezone localization / conversion
        if self.timezone:
            if parsed.dt.tz is None:
                parsed = parsed.dt.tz_localize(self.timezone)
            else:
                parsed = parsed.dt.tz_convert(self.timezone)

        # Range validation
        if self.min_date is not None or self.max_date is not None:
            mask = pd.Series(True, index=parsed.index)
            if self.min_date is not None:
                mask &= parsed >= self.min_date
            if self.max_date is not None:
                mask &= parsed <= self.max_date
            if self.null_out_of_range:
                parsed = parsed.where(mask)

        # Format output as string if requested
        if self.output_format:
            df[col] = parsed.dt.strftime(self.output_format)
        else:
            df[col] = parsed

        # Extract date components as new columns
        if self.extract_components:
            df[f"{col}_year"] = parsed.dt.year
            df[f"{col}_month"] = parsed.dt.month
            df[f"{col}_day"] = parsed.dt.day
            df[f"{col}_dayofweek"] = parsed.dt.dayofweek
            df[f"{col}_hour"] = parsed.dt.hour
            df[f"{col}_quarter"] = parsed.dt.quarter

        return df
