"""Categorical encoding transformers."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..exceptions import TransformationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Encoder:
    """
    Encodes categorical columns using label encoding or one-hot encoding.

    Example::

        encoder = Encoder(
            label_columns=["status", "region"],
            onehot_columns=["category"],
            drop_first=True,
        )
        df = encoder.transform(df)
        # Retrieve label mappings for inverse transform
        print(encoder.label_mappings_)
    """

    def __init__(
        self,
        label_columns: Optional[List[str]] = None,
        onehot_columns: Optional[List[str]] = None,
        drop_first: bool = False,
        handle_unknown: str = "ignore",  # "ignore" or "error"
        errors: str = "coerce",
    ):
        self.label_columns = label_columns or []
        self.onehot_columns = onehot_columns or []
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.errors = errors
        self.label_mappings_: Dict[str, Dict] = {}
        self.inverse_mappings_: Dict[str, Dict] = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns."""
        df = df.copy()

        for col in self.label_columns:
            if col not in df.columns:
                logger.warning("Label encode: column '%s' not found — skipping.", col)
                continue
            try:
                df, mapping = self._label_encode(df, col)
                self.label_mappings_[col] = mapping
                self.inverse_mappings_[col] = {v: k for k, v in mapping.items()}
                logger.debug("Label encoded column '%s'.", col)
            except Exception as exc:
                msg = f"Label encoding failed for '{col}': {exc}"
                if self.errors == "raise":
                    raise TransformationError(msg, step="Encoder", column=col) from exc
                logger.warning(msg)

        if self.onehot_columns:
            cols_present = [c for c in self.onehot_columns if c in df.columns]
            if cols_present:
                try:
                    df = pd.get_dummies(
                        df,
                        columns=cols_present,
                        drop_first=self.drop_first,
                        dummy_na=False,
                    )
                    logger.debug("One-hot encoded columns: %s", cols_present)
                except Exception as exc:
                    msg = f"One-hot encoding failed: {exc}"
                    if self.errors == "raise":
                        raise TransformationError(msg, step="Encoder") from exc
                    logger.warning(msg)

        return df

    def inverse_transform_labels(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Reverse label encoding using stored mappings."""
        df = df.copy()
        target = columns or list(self.inverse_mappings_.keys())
        for col in target:
            if col not in df.columns or col not in self.inverse_mappings_:
                continue
            df[col] = df[col].map(self.inverse_mappings_[col])
        return df

    @staticmethod
    def _label_encode(df: pd.DataFrame, col: str):
        categories = df[col].dropna().unique().tolist()
        categories.sort(key=lambda x: str(x))
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        df[col] = df[col].map(mapping)
        return df, mapping
