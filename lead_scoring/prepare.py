"""Load the lead-scoring data, impute missing values, and split it.

The target is ``converted``: whether the client signed up. Categorical
gaps are filled with ``'NA'`` and numerical gaps with ``0.0``. The split
matches the homework calls exactly (60% train, 20% validation, 20% test).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from lead_scoring.paths import DATA_DIR

DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/course_lead_scoring_2026.csv"
)
DATA_PATH = DATA_DIR / "course_lead_scoring_2026.csv"

TARGET = "converted"
CATEGORICAL = [
    "lead_source",
    "industry",
    "employment_status",
    "location",
]
NUMERICAL = [
    "annual_income",
    "number_of_courses_viewed",
    "interaction_count",
    "lead_score",
]
Q1_CANDIDATES = [
    "lead_score",
    "number_of_courses_viewed",
    "interaction_count",
    "annual_income",
]
FEATURE_COLUMNS = CATEGORICAL + NUMERICAL


@dataclass(frozen=True)
class PreparedData:
    """Imputed frame plus the three homework splits. Every frame keeps the target."""

    missing_counts: pd.Series
    full: pd.DataFrame
    full_train: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the cached CSV, downloading it on the first run."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.read_csv(DATA_URL)
        frame.to_csv(path, index=False)
        return frame
    return pd.read_csv(path)


def missing_counts(frame: pd.DataFrame) -> pd.Series:
    """Count missing values in the features, leaving the target out."""
    return frame[FEATURE_COLUMNS].isna().sum().astype(int)


def impute(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace categorical gaps with 'NA' and numerical gaps with 0.0."""
    expected = set(FEATURE_COLUMNS + [TARGET])
    if set(frame.columns) != expected:
        missing = expected - set(frame.columns)
        extra = set(frame.columns) - expected
        raise ValueError(f"Unexpected columns. missing={sorted(missing)} extra={sorted(extra)}")

    prepared = frame.copy()
    for column in CATEGORICAL:
        prepared[column] = prepared[column].fillna("NA").astype(str).str.strip()
        prepared.loc[prepared[column] == "", column] = "NA"
    for column in NUMERICAL:
        prepared[column] = (
            pd.to_numeric(prepared[column], errors="coerce").fillna(0.0).astype(float)
        )
    prepared[TARGET] = prepared[TARGET].astype(int)
    return prepared


def split_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into full-train, train, validation, and test with the homework seeds."""
    df_full_train, df_test = train_test_split(frame, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1
    )
    return (
        df_full_train.reset_index(drop=True),
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def prepare() -> PreparedData:
    """Load, report missing values, impute, and split."""
    raw = load_raw()
    counts = missing_counts(raw)
    prepared = impute(raw)
    leftover = missing_counts(prepared)
    if int(leftover.sum()) != 0:
        raise RuntimeError(f"Imputation left missing values:\n{leftover}")
    full_train, train, validation, test = split_data(prepared)
    return PreparedData(
        missing_counts=counts,
        full=prepared,
        full_train=full_train,
        train=train,
        validation=validation,
        test=test,
    )


def format_report(data: PreparedData) -> str:
    """Text summary of missing values and split sizes."""
    lines = ["Missing values before imputation:"]
    for column, count in data.missing_counts.items():
        lines.append(f"  {column}: {int(count)}")
    lines.append("Missing values after imputation: 0")
    lines.append(
        "Split sizes: "
        f"full={len(data.full)}, "
        f"full_train={len(data.full_train)}, "
        f"train={len(data.train)}, "
        f"validation={len(data.validation)}, "
        f"test={len(data.test)}"
    )
    rate = float(data.full[TARGET].mean())
    lines.append(f"Positive rate (converted=1): {rate:.4f}")
    return "\n".join(lines)


def main() -> PreparedData:
    data = prepare()
    print(format_report(data))
    return data


if __name__ == "__main__":
    main()
