"""Precision, recall, and F1 across decision thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def threshold_grid() -> np.ndarray:
    """Thresholds from 0.0 to 1.0 inclusive, step 0.01."""
    return np.round(np.linspace(0.0, 1.0, 101), 2)


def precision_recall_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[float, float]:
    """Precision and recall at ``threshold``. Empty slices are scored as 0."""
    predicted = y_score >= threshold
    actual = y_true == 1
    true_positive = int(np.sum(predicted & actual))
    false_positive = int(np.sum(predicted & ~actual))
    false_negative = int(np.sum(~predicted & actual))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    return precision, recall


def f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall. Both zero yields zero."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def threshold_table(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    """Precision, recall, F1, and |precision - recall| at every threshold.

    The manual counts are checked against scikit-learn so a definition drift
    fails immediately.
    """
    rows: list[dict[str, float]] = []
    for threshold in threshold_grid():
        precision, recall = precision_recall_at(y_true, y_score, float(threshold))
        predicted = (y_score >= threshold).astype(int)
        sklearn_precision = float(precision_score(y_true, predicted, zero_division=0))
        sklearn_recall = float(recall_score(y_true, predicted, zero_division=0))
        sklearn_f1 = float(f1_score(y_true, predicted, zero_division=0))
        manual_f1 = f1(precision, recall)
        if not (
            np.isclose(precision, sklearn_precision)
            and np.isclose(recall, sklearn_recall)
            and np.isclose(manual_f1, sklearn_f1)
        ):
            raise AssertionError(
                f"Metric mismatch at threshold {threshold}: "
                f"manual=({precision}, {recall}, {manual_f1}) "
                f"sklearn=({sklearn_precision}, {sklearn_recall}, {sklearn_f1})"
            )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": manual_f1,
                "abs_diff": abs(precision - recall),
                "both_zero": float(precision == 0.0 and recall == 0.0),
            }
        )
    return pd.DataFrame(rows)


def closest_precision_recall(table: pd.DataFrame) -> pd.Series:
    """Threshold with the smallest |precision - recall|.

    Thresholds where both metrics are zero are ignored. Ties keep the
    smallest threshold.
    """
    usable = table.loc[table["both_zero"] == 0.0]
    if usable.empty:
        raise RuntimeError("Every threshold has precision and recall equal to zero.")
    min_gap = usable["abs_diff"].min()
    candidates = usable.loc[np.isclose(usable["abs_diff"], min_gap)]
    return candidates.sort_values("threshold").iloc[0]


def maximum_f1(table: pd.DataFrame) -> pd.Series:
    """Threshold where F1 is maximal. Ties keep the smallest threshold."""
    max_score = table["f1"].max()
    candidates = table.loc[np.isclose(table["f1"], max_score)]
    return candidates.sort_values("threshold").iloc[0]
