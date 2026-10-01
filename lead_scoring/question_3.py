"""Question 3: threshold where precision and recall are closest.

The model from question 2 is scored on every threshold from 0.0 to 1.0
with step 0.01. Thresholds at which both metrics are zero are ignored.
The answer is the smallest threshold among those with the smallest
absolute difference between precision and recall.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lead_scoring.metrics import closest_precision_recall, threshold_table
from lead_scoring.modeling import fit_and_score_validation
from lead_scoring.paths import PLOTS_DIR, RESULTS_DIR
from lead_scoring.prepare import prepare

OPTIONS = (0.43, 0.63, 0.73, 0.83)
PLOT_PATH = PLOTS_DIR / "question_3_precision_recall.png"
TABLE_PATH = RESULTS_DIR / "question_3_precision_recall.csv"


def solve() -> dict:
    data = prepare()
    y_val, scores, _vectorizer, _model = fit_and_score_validation(data.train, data.validation)
    table = threshold_table(y_val, scores)
    chosen = closest_precision_recall(table)
    threshold = float(chosen["threshold"])
    return {
        "table": table,
        "threshold": threshold,
        "precision": float(chosen["precision"]),
        "recall": float(chosen["recall"]),
        "abs_diff": float(chosen["abs_diff"]),
        "matches_option": threshold in OPTIONS,
        "answer": threshold,
    }


def save_plot(table: pd.DataFrame, threshold: float, path=PLOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = table.copy()
    undefined = display["both_zero"] == 1.0
    display.loc[undefined, ["precision", "recall"]] = float("nan")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=140)
    ax.plot(display["threshold"], display["precision"], color="#0f6e6e", linewidth=2.0, label="Precision")
    ax.plot(display["threshold"], display["recall"], color="#c4492c", linewidth=2.0, label="Recall")
    row = table.loc[table["threshold"] == threshold].iloc[0]
    ax.axvline(threshold, color="#1f2933", linestyle="--", linewidth=1.0, label=f"Closest pair ({threshold:.2f})")
    ax.scatter([threshold], [row["precision"]], color="#0f6e6e", zorder=3)
    ax.scatter([threshold], [row["recall"]], color="#c4492c", zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision and recall on the validation set")
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
    usable = result["table"].loc[result["table"]["both_zero"] == 0.0]
    ignored = result["table"].loc[result["table"]["both_zero"] == 1.0, "threshold"]
    nearest = usable.nsmallest(5, "abs_diff")
    print("Question 3: precision and recall intersection")
    ignored_text = ", ".join(f"{value:.2f}" for value in ignored.tolist())
    print(f"  Ignored thresholds where both metrics are zero: {ignored_text}")
    print("  Closest remaining thresholds:")
    for _, row in nearest.iterrows():
        print(
            f"    t={row['threshold']:.2f}  precision={row['precision']:.4f}  "
            f"recall={row['recall']:.4f}  |P-R|={row['abs_diff']:.4f}"
        )
    print(
        f"  At t={result['threshold']:.2f}: "
        f"precision={result['precision']:.4f}, recall={result['recall']:.4f}"
    )
    print(f"ANSWER: {result['threshold']:.2f}")
    print(f"  Plot: {PLOT_PATH}")
    if not result["matches_option"]:
        print(f"  Note: {result['threshold']:.2f} is outside the listed options {OPTIONS}.")
    return result


if __name__ == "__main__":
    main()
