"""Run data preparation and all six questions, then write ``results/answers.json``."""

from __future__ import annotations

import json
import platform

import numpy
import pandas
import sklearn

from lead_scoring.paths import RESULTS_DIR
from lead_scoring.prepare import format_report, prepare
from lead_scoring.question_1 import solve as solve_q1
from lead_scoring.question_2 import solve as solve_q2
from lead_scoring.question_3 import main as run_q3
from lead_scoring.question_4 import main as run_q4
from lead_scoring.question_5 import solve as solve_q5
from lead_scoring.question_6 import solve as solve_q6


def _without_frames(result: dict) -> dict:
    """Drop DataFrames so the payload can be serialized."""
    return {key: value for key, value in result.items() if key != "table"}


def main() -> dict:
    print(format_report(prepare()))
    print()
    q1 = solve_q1()
    print("Question 1")
    for row in q1["scores"]:
        direction = "negated" if row["inverted"] else "as-is"
        print(
            f"  {row['feature']:<28} raw={row['raw_auc']:.6f}  "
            f"adjusted={row['adjusted_auc']:.6f}  ({direction})"
        )
    print(f"ANSWER: {q1['answer']}")
    print()

    q2 = solve_q2()
    print("Question 2")
    print(f"  AUC: {q2['auc']:.6f}")
    print(f"ANSWER: {q2['auc_rounded']:.3f}")
    print()

    q3 = run_q3()
    print()
    q4 = run_q4()
    print()

    q5 = solve_q5()
    print("Question 5")
    for index, score in enumerate(q5["fold_aucs"], start=1):
        print(f"  Fold {index}: {score:.6f}")
    print(f"  Mean: {q5['mean_auc']:.6f}  std: {q5['std_auc']:.6f}")
    print(f"ANSWER: {q5['std_rounded']:.3f}")
    print()

    q6 = solve_q6()
    print("Question 6")
    for row in q6["rows"]:
        print(
            f"  C={row['C']:<10} mean={row['mean_rounded']:.3f}  std={row['std_rounded']:.3f}"
        )
    print(f"ANSWER: {q6['answer']}")

    payload = {
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "question_1": q1,
        "question_2": q2,
        "question_3": _without_frames(q3),
        "question_4": _without_frames(q4),
        "question_5": q5,
        "question_6": q6,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_DIR / "answers.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"Wrote {destination}")
    return payload


if __name__ == "__main__":
    main()
