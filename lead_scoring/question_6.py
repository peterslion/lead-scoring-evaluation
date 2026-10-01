"""Question 6: choose C by 5-fold mean ROC AUC.

The same KFold as question 5 is repeated for C in [0.000001, 0.001, 1].
Mean and standard deviation are rounded to 3 decimal digits. The best mean
wins. A tie on the rounded mean is broken by the lower rounded std, then by
the smaller C.
"""

from __future__ import annotations

import numpy as np

from lead_scoring.prepare import prepare
from lead_scoring.question_5 import fold_aucs

C_VALUES = [0.000001, 0.001, 1]


def solve() -> dict:
    data = prepare()
    rows = []
    for C in C_VALUES:
        scores = fold_aucs(data.full_train, C=C)
        mean_auc = float(np.mean(scores))
        std_auc = float(np.std(scores))
        rows.append(
            {
                "C": C,
                "fold_aucs": [float(score) for score in scores],
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_rounded": round(mean_auc, 3),
                "std_rounded": round(std_auc, 3),
            }
        )
    winner = sorted(rows, key=lambda row: (-row["mean_rounded"], row["std_rounded"], row["C"]))[0]
    return {"rows": rows, "answer": winner["C"], "best": winner}


def _format_c(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") if value >= 1 else f"{value:.6f}"


def main() -> dict:
    result = solve()
    print("Question 6: hyperparameter tuning for C")
    print(f"  {'C':<12} {'mean AUC':>10} {'std':>8}")
    for row in result["rows"]:
        print(
            f"  {_format_c(row['C']):<12} {row['mean_rounded']:>10.3f} {row['std_rounded']:>8.3f}"
            f"    (raw {row['mean_auc']:.6f} ± {row['std_auc']:.6f})"
        )
    print(f"ANSWER: {_format_c(result['answer'])}")
    return result


if __name__ == "__main__":
    main()
