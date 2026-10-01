"""Question 1: ROC AUC as univariate feature importance.

Each numerical feature is used as a score on the training set. An AUC below
0.5 is recomputed on the negated feature. The answer is the candidate with
the highest adjusted AUC.
"""

from __future__ import annotations

from sklearn.metrics import roc_auc_score

from lead_scoring.prepare import Q1_CANDIDATES, TARGET, prepare


def univariate_auc(y_true, values) -> tuple[float, float, bool]:
    """Return raw AUC, direction-adjusted AUC, and whether the feature was negated."""
    raw = float(roc_auc_score(y_true, values))
    if raw < 0.5:
        adjusted = float(roc_auc_score(y_true, -values))
        return raw, adjusted, True
    return raw, raw, False


def solve() -> dict:
    data = prepare()
    y_train = data.train[TARGET].to_numpy()
    rows = []
    for column in Q1_CANDIDATES:
        raw, adjusted, inverted = univariate_auc(y_train, data.train[column].to_numpy())
        rows.append(
            {
                "feature": column,
                "raw_auc": raw,
                "adjusted_auc": adjusted,
                "inverted": inverted,
            }
        )
    winner = max(rows, key=lambda row: row["adjusted_auc"])
    return {"scores": rows, "answer": winner["feature"], "auc": winner["adjusted_auc"]}


def main() -> dict:
    result = solve()
    print("Question 1: ROC AUC feature importance (training set)")
    for row in result["scores"]:
        direction = "negated" if row["inverted"] else "as-is"
        print(
            f"  {row['feature']:<28} raw={row['raw_auc']:.6f}  "
            f"adjusted={row['adjusted_auc']:.6f}  ({direction})"
        )
    print(f"ANSWER: {result['answer']}")
    return result


if __name__ == "__main__":
    main()
