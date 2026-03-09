"""Standardization transformers for enterprise data normalization."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..exceptions import TransformationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Standardizer:
    """
    Standardizes values across columns: deduplication, column renaming,
    dtype casting, and value mapping.

    Example::

        standardizer = Standardizer(
            rename_columns={"cust_id": "customer_id", "rev": "revenue"},
            dtype_map={"customer_id": "int64", "revenue": "float64"},
            value_maps={"status": {"Y": True, "N": False, "1": True, "0": False}},
            drop_duplicates=True,
            duplicate_subset=["customer_id"],
            reset_index=True,
        )
        df = standardizer.transform(df)
    """

    def __init__(
        self,
        rename_columns: Optional[Dict[str, str]] = None,
        dtype_map: Optional[Dict[str, str]] = None,
        value_maps: Optional[Dict[str, Dict]] = None,
        drop_duplicates: bool = False,
        duplicate_subset: Optional[List[str]] = None,
        duplicate_keep: str = "first",
        reset_index: bool = True,
        column_name_style: Optional[str] = None,  # "snake_case", "lower", "upper"
        errors: str = "coerce",
    ):
        self.rename_columns = rename_columns or {}
        self.dtype_map = dtype_map or {}
        self.value_maps = value_maps or {}
        self.drop_duplicates = drop_duplicates
        self.duplicate_subset = duplicate_subset
        self.duplicate_keep = duplicate_keep
        self.reset_index = reset_index
        self.column_name_style = column_name_style
        self.errors = errors

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all standardization steps."""
        df = df.copy()

        # Column name normalization
        if self.column_name_style:
            df = self._normalize_column_names(df)

        # Rename columns
        if self.rename_columns:
            df = df.rename(columns=self.rename_columns)
            logger.debug("Renamed %d column(s).", len(self.rename_columns))

        # Dtype casting
        for col, dtype in self.dtype_map.items():
            if col not in df.columns:
                logger.warning("Column '%s' not found for dtype cast — skipping.", col)
                continue
            try:
                df[col] = df[col].astype(dtype)
            except Exception as exc:
                msg = f"Failed to cast '{col}' to '{dtype}': {exc}"
                if self.errors == "raise":
                    raise TransformationError(msg, step="Standardizer", column=col) from exc
                logger.warning(msg)

        # Value mapping
        for col, mapping in self.value_maps.items():
            if col not in df.columns:
                continue
            df[col] = df[col].map(lambda v: mapping.get(v, v))
            logger.debug("Applied value map to column '%s'.", col)

        # Deduplication
        if self.drop_duplicates:
            before = len(df)
            df = df.drop_duplicates(
                subset=self.duplicate_subset, keep=self.duplicate_keep
            )
            removed = before - len(df)
            if removed:
                logger.info("Removed %d duplicate row(s).", removed)

        if self.reset_index:
            df = df.reset_index(drop=True)

        return df

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        import re

        def to_snake(name: str) -> str:
            name = re.sub(r"[\s\-]+", "_", name)
            name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
            name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
            return name.lower()

        if self.column_name_style == "snake_case":
            df.columns = [to_snake(c) for c in df.columns]
        elif self.column_name_style == "lower":
            df.columns = [c.lower().strip() for c in df.columns]
        elif self.column_name_style == "upper":
            df.columns = [c.upper().strip() for c in df.columns]

        return df
