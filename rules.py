"""Rule-based validation for cross-column and row-level business logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pandas as pd

from ..utils.logger import get_logger
from .schema import ValidationResult

logger = get_logger(__name__)


@dataclass
class Rule:
    """
    A single validation rule.

    Args:
        name: Human-readable rule name.
        condition: Callable accepting a DataFrame row (Series) or full DataFrame
                   and returning a boolean Series (row-level) or bool (frame-level).
        is_row_level: If True, condition receives each row as pd.Series.
                      If False, condition receives the full DataFrame.
        severity: "error" or "warning".
        message: Description of the violation.
    """
    name: str
    condition: Callable
    is_row_level: bool = True
    severity: str = "error"
    message: str = ""


class RuleValidator:
    """
    Validate a DataFrame against a set of custom business rules.

    Example::

        rules = [
            Rule(
                name="age_positive",
                condition=lambda row: row["age"] >= 0,
                message="Age must be non-negative.",
            ),
            Rule(
                name="end_after_start",
                condition=lambda row: row["end_date"] >= row["start_date"],
                message="end_date must be >= start_date.",
            ),
            Rule(
                name="revenue_not_all_zero",
                condition=lambda df: df["revenue"].sum() > 0,
                is_row_level=False,
                severity="warning",
                message="Total revenue is zero — possible data issue.",
            ),
        ]
        validator = RuleValidator(rules)
        result = validator.validate(df)
        print(result.summary())
    """

    def __init__(self, rules: List[Rule], on_error: str = "warn"):
        self.rules = rules
        self.on_error = on_error

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Apply all rules and return a consolidated ValidationResult."""
        result = ValidationResult(is_valid=True)

        for rule in self.rules:
            try:
                if rule.is_row_level:
                    self._apply_row_rule(df, rule, result)
                else:
                    self._apply_frame_rule(df, rule, result)
            except Exception as exc:
                msg = f"Rule '{rule.name}' raised an exception: {exc}"
                logger.error(msg)
                result.add_warning(msg)

        return result

    def _apply_row_rule(self, df: pd.DataFrame, rule: Rule, result: ValidationResult):
        try:
            mask = df.apply(rule.condition, axis=1)
        except Exception as exc:
            result.add_warning(f"Rule '{rule.name}' could not be applied: {exc}")
            return

        violations = (~mask).sum()
        if violations:
            msg = (
                f"Rule '{rule.name}' violated on {violations} row(s). "
                f"{rule.message}"
            )
            if rule.severity == "error":
                result.add_error(msg)
            else:
                result.add_warning(msg)
        else:
            logger.debug("Rule '%s' passed.", rule.name)

    def _apply_frame_rule(self, df: pd.DataFrame, rule: Rule, result: ValidationResult):
        passed = rule.condition(df)
        if not passed:
            msg = f"Rule '{rule.name}' failed. {rule.message}"
            if rule.severity == "error":
                result.add_error(msg)
            else:
                result.add_warning(msg)
        else:
            logger.debug("Rule '%s' passed.", rule.name)
