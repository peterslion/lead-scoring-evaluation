"""Question 4: threshold that maximizes F1 on the validation set.

F1 = 2 * P * R / (P + R), evaluated from 0.0 to 1.0 with step 0.01.
If several thresholds share the maximum, the smallest one is reported.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lead_scoring.metrics import maximum_f1, threshold_table
from lead_scoring.modeling import fit_and_score_validation
from lead_scoring.paths import PLOTS_DIR, RESULTS_DIR
from lead_scoring.prepare import prepare

OPTIONS = (0.21, 0.41, 0.61, 0.81)
PLOT_PATH = PLOTS_DIR / "question_4_f1.png"
TABLE_PATH = RESULTS_DIR / "question_4_f1.csv"


def solve() -> dict:
    data = prepare()
    y_val, scores, _vectorizer, _model = fit_and_score_validation(data.train, data.validation)
    table = threshold_table(y_val, scores)
    chosen = maximum_f1(table)
    threshold = float(chosen["threshold"])
    tied = table.loc[np.isclose(table["f1"], float(chosen["f1"])), "threshold"]
    return {
        "table": table,
        "threshold": threshold,
        "f1": float(chosen["f1"]),
        "precision": float(chosen["precision"]),
        "recall": float(chosen["recall"]),
        "tied_thresholds": [float(value) for value in tied.tolist()],
        "matches_option": threshold in OPTIONS,
        "answer": threshold,
    }


def save_plot(table: pd.DataFrame, threshold: float, path=PLOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=140)
    ax.plot(table["threshold"], table["f1"], color="#1d4e89", linewidth=2.0, label="F1")
    row = table.loc[table["threshold"] == threshold].iloc[0]
    ax.axvline(threshold, color="#1f2933", linestyle="--", linewidth=1.0, label=f"Maximum ({threshold:.2f})")
    ax.scatter([threshold], [row["f1"]], color="#1d4e89", zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("F1")
    ax.set_title("F1 score on the validation set")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> dict:
    result = solve()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result["table"].to_csv(TABLE_PATH, index=False)
    save_plot(result["table"], result["threshold"])
    top = result["table"].sort_values(["f1", "threshold"], ascending=[False, True]).head(5)
    print("Question 4: F1 score")
    print("  Highest F1 thresholds:")
    for _, row in top.iterrows():
        print(
            f"    t={row['threshold']:.2f}  F1={row['f1']:.4f}  "
            f"precision={row['precision']:.4f}  recall={row['recall']:.4f}"
        )
    if len(result["tied_thresholds"]) > 1:
        joined = ", ".join(f"{value:.2f}" for value in result["tied_thresholds"])
        print(f"  Tied maximum at: {joined}")
    print(f"ANSWER: {result['threshold']:.2f}")
    print(f"  Plot: {PLOT_PATH}")
    if not result["matches_option"]:
        print(f"  Note: {result['threshold']:.2f} is outside the listed options {OPTIONS}.")
    return result


if __name__ == "__main__":
    main()
