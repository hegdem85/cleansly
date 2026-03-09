"""
CleaningPipeline — Orchestrates a sequence of cleaning, validation,
and transformation steps with audit logging and metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .exceptions import PipelineError
from .utils.logger import get_logger
from .utils.profiler import DataProfiler, DataProfile

logger = get_logger(__name__)


@dataclass
class StepResult:
    step_name: str
    success: bool
    duration_seconds: float
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineReport:
    total_duration_seconds: float
    steps: List[StepResult] = field(default_factory=list)
    profile_before: Optional[DataProfile] = None
    profile_after: Optional[DataProfile] = None

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "PIPELINE EXECUTION REPORT",
            "=" * 65,
            f"  Total Duration  : {self.total_duration_seconds:.3f}s",
            f"  Steps Executed  : {len(self.steps)}",
            "-" * 65,
        ]
        for i, step in enumerate(self.steps, 1):
            status = "✓" if step.success else "✗"
            row_delta = step.rows_after - step.rows_before
            col_delta = step.cols_after - step.cols_before
            lines.append(
                f"  {i:>2}. [{status}] {step.step_name:<30} "
                f"{step.duration_seconds:.3f}s  "
                f"rows Δ{row_delta:+d}  cols Δ{col_delta:+d}"
            )
            if step.error:
                lines.append(f"       ERROR: {step.error}")

        if self.profile_before and self.profile_after:
            lines.extend([
                "-" * 65,
                "  DATA SHAPE CHANGE",
                f"    Rows  : {self.profile_before.row_count:,} → {self.profile_after.row_count:,}",
                f"    Cols  : {self.profile_before.col_count} → {self.profile_after.col_count}",
                f"    Nulls : {self.profile_before.null_pct:.2f}% → {self.profile_after.null_pct:.2f}%",
                f"    Dupes : {self.profile_before.duplicate_rows:,} → {self.profile_after.duplicate_rows:,}",
            ])
        lines.append("=" * 65)
        return "\n".join(lines)


class CleaningPipeline:
    """
    Composable, auditable data cleaning pipeline.

    Chains together any combination of cleaners, validators, transformers,
    and custom functions. Generates a full execution report.

    Example::

        pipeline = (
            CleaningPipeline(name="CustomerDataPipeline", profile=True)
            .add_step("standardize", Standardizer(column_name_style="snake_case"))
            .add_step("clean_text", TextCleaner(strip_whitespace=True, normalize_case="lower"))
            .add_step("handle_missing", MissingValueHandler(strategy="fill_median"))
            .add_step("clean_numerics", NumericCleaner(outlier_method="iqr"))
            .add_step("validate", SchemaValidator(schema))
        )

        df_clean, report = pipeline.run(df)
        print(report.summary())
    """

    def __init__(
        self,
        name: str = "CleaningPipeline",
        profile: bool = False,
        on_error: str = "continue",  # "continue" or "abort"
    ):
        self.name = name
        self.profile = profile
        self.on_error = on_error
        self._steps: List[Tuple[str, Any]] = []

    def add_step(self, name: str, step: Any) -> "CleaningPipeline":
        """
        Add a step to the pipeline.

        Args:
            name: Unique step name for logging and reporting.
            step: Any object with a `clean(df)`, `transform(df)`,
                  `validate(df)`, or a plain callable `(df) -> df`.

        Returns:
            Self (for method chaining).
        """
        self._steps.append((name, step))
        return self

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, PipelineReport]:
        """
        Execute the pipeline on the given DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (cleaned DataFrame, PipelineReport).

        Raises:
            PipelineError: If a step fails and on_error="abort".
        """
        logger.info("Starting pipeline '%s' with %d step(s).", self.name, len(self._steps))
        pipeline_start = time.perf_counter()

        profiler = DataProfiler() if self.profile else None
        profile_before = profiler.profile(df) if profiler else None

        step_results: List[StepResult] = []
        current_df = df.copy()

        for step_name, step in self._steps:
            rows_before, cols_before = current_df.shape
            step_start = time.perf_counter()
            error_msg = None
            success = True

            logger.info("Running step: %s", step_name)
            try:
                current_df = self._run_step(step, current_df)
            except Exception as exc:
                success = False
                error_msg = str(exc)
                logger.error("Step '%s' failed: %s", step_name, exc)
                if self.on_error == "abort":
                    raise PipelineError(
                        f"Pipeline aborted at step '{step_name}': {exc}"
                    ) from exc

            rows_after, cols_after = current_df.shape
            duration = time.perf_counter() - step_start

            step_results.append(
                StepResult(
                    step_name=step_name,
                    success=success,
                    duration_seconds=round(duration, 4),
                    rows_before=rows_before,
                    rows_after=rows_after,
                    cols_before=cols_before,
                    cols_after=cols_after,
                    error=error_msg,
                )
            )

        profile_after = profiler.profile(current_df) if profiler else None
        total_duration = round(time.perf_counter() - pipeline_start, 4)

        report = PipelineReport(
            total_duration_seconds=total_duration,
            steps=step_results,
            profile_before=profile_before,
            profile_after=profile_after,
        )
        logger.info(
            "Pipeline '%s' completed in %.3fs. %d/%d steps succeeded.",
            self.name,
            total_duration,
            sum(s.success for s in step_results),
            len(step_results),
        )
        return current_df, report

    @staticmethod
    def _run_step(step: Any, df: pd.DataFrame) -> pd.DataFrame:
        """Dispatch to the appropriate step interface."""
        if callable(step) and not hasattr(step, "clean") and not hasattr(step, "transform"):
            result = step(df)
            return result if isinstance(result, pd.DataFrame) else df

        if hasattr(step, "clean"):
            result = step.clean(df)
            return result if isinstance(result, pd.DataFrame) else df

        if hasattr(step, "transform"):
            result = step.transform(df)
            return result if isinstance(result, pd.DataFrame) else df

        if hasattr(step, "validate"):
            step.validate(df)  # Validation runs but doesn't modify df
            return df

        raise PipelineError(
            f"Step object {type(step).__name__} has no 'clean', 'transform', or 'validate' method."
        )
