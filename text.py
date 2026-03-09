"""Text cleaning operations for string columns."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, List

import pandas as pd

from ..exceptions import TransformationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """
    Comprehensive text cleaning for enterprise string data.

    Supports trimming, casing, unicode normalization, PII redaction,
    special character removal, and custom regex replacements.

    Example::

        cleaner = TextCleaner(
            strip_whitespace=True,
            normalize_case="lower",
            remove_html=True,
            redact_emails=True,
        )
        df = cleaner.clean(df, columns=["name", "description"])
    """

    # Common PII patterns
    _EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    _PHONE_RE = re.compile(
        r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}"
    )
    _SSN_RE = re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b")
    _CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")
    _HTML_RE = re.compile(r"<[^>]+>")
    _MULTI_SPACE_RE = re.compile(r"\s+")
    _URL_RE = re.compile(r"https?://\S+|www\.\S+")

    def __init__(
        self,
        strip_whitespace: bool = True,
        normalize_case: Optional[str] = None,  # "lower", "upper", "title"
        remove_html: bool = False,
        remove_urls: bool = False,
        normalize_unicode: bool = True,
        unicode_form: str = "NFC",
        redact_emails: bool = False,
        redact_phones: bool = False,
        redact_ssn: bool = False,
        redact_credit_cards: bool = False,
        redact_placeholder: str = "[REDACTED]",
        allowed_chars: Optional[str] = None,  # regex pattern of allowed chars
        custom_replacements: Optional[List[tuple]] = None,  # [(pattern, replacement)]
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        null_if_empty: bool = True,
        errors: str = "coerce",  # "coerce" or "raise"
    ):
        self.strip_whitespace = strip_whitespace
        self.normalize_case = normalize_case
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.normalize_unicode = normalize_unicode
        self.unicode_form = unicode_form
        self.redact_emails = redact_emails
        self.redact_phones = redact_phones
        self.redact_ssn = redact_ssn
        self.redact_credit_cards = redact_credit_cards
        self.redact_placeholder = redact_placeholder
        self.allowed_chars = re.compile(allowed_chars) if allowed_chars else None
        self.custom_replacements = [
            (re.compile(p), r) for p, r in (custom_replacements or [])
        ]
        self.min_length = min_length
        self.max_length = max_length
        self.null_if_empty = null_if_empty
        self.errors = errors

    def clean(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply text cleaning to specified columns (or all object/string columns).

        Args:
            df: Input DataFrame.
            columns: List of column names to clean. Defaults to all string columns.

        Returns:
            Cleaned DataFrame (copy).
        """
        df = df.copy()
        target_cols = columns or df.select_dtypes(include=["object", "string"]).columns.tolist()

        for col in target_cols:
            if col not in df.columns:
                logger.warning("Column '%s' not found in DataFrame — skipping.", col)
                continue
            try:
                df[col] = df[col].apply(self._clean_value)
                logger.debug("Cleaned text column '%s'.", col)
            except Exception as exc:
                msg = f"Failed to clean column '{col}': {exc}"
                if self.errors == "raise":
                    raise TransformationError(msg, step="TextCleaner", column=col) from exc
                logger.warning(msg)

        return df

    def _clean_value(self, value) -> Optional[str]:
        if pd.isna(value):
            return value

        text = str(value)

        if self.normalize_unicode:
            text = unicodedata.normalize(self.unicode_form, text)

        if self.remove_html:
            text = self._HTML_RE.sub(" ", text)

        if self.remove_urls:
            text = self._URL_RE.sub(" ", text)

        if self.redact_ssn:
            text = self._SSN_RE.sub(self.redact_placeholder, text)

        if self.redact_credit_cards:
            text = self._CREDIT_CARD_RE.sub(self.redact_placeholder, text)

        if self.redact_emails:
            text = self._EMAIL_RE.sub(self.redact_placeholder, text)

        if self.redact_phones:
            text = self._PHONE_RE.sub(self.redact_placeholder, text)

        for pattern, replacement in self.custom_replacements:
            text = pattern.sub(replacement, text)

        if self.allowed_chars:
            text = "".join(c for c in text if self.allowed_chars.match(c) or c == " ")

        if self.strip_whitespace:
            text = self._MULTI_SPACE_RE.sub(" ", text).strip()

        if self.normalize_case == "lower":
            text = text.lower()
        elif self.normalize_case == "upper":
            text = text.upper()
        elif self.normalize_case == "title":
            text = text.title()

        if self.max_length is not None:
            text = text[: self.max_length]

        if self.min_length is not None and len(text) < self.min_length:
            return None

        if self.null_if_empty and text == "":
            return None

        return text
