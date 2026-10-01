"""Question 5: standard deviation of validation AUC across 5 folds.

``KFold(n_splits=5, shuffle=True, random_state=1)`` walks ``df_full_train``.
Each fold refits DictVectorizer and
``LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)``.
The standard deviation is ``numpy.std`` (population, ddof=0), which is the
convention used in the course notebooks.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from lead_scoring.modeling import predict_scores, target_array, train_model
from lead_scoring.prepare import prepare

OPTIONS = (0.001, 0.007, 0.013, 0.060)


def fold_aucs(df_full_train, C: float = 1.0) -> np.ndarray:
    """Validation ROC AUC on each fold of ``df_full_train``."""
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores: list[float] = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        vectorizer, model = train_model(df_train, target_array(df_train), C=C)
        y_pred = predict_scores(df_val, vectorizer, model)
        scores.append(float(roc_auc_score(target_array(df_val), y_pred)))
    return np.asarray(scores, dtype=float)


def solve() -> dict:
    data = prepare()
    scores = fold_aucs(data.full_train, C=1.0)
    std = float(np.std(scores))
    rounded = round(std, 3)
    return {
        "fold_aucs": [float(score) for score in scores],
        "mean_auc": float(np.mean(scores)),
        "std_auc": std,
        "std_rounded": rounded,
        "sample_std": float(np.std(scores, ddof=1)),
        "matches_option": rounded in OPTIONS,
        "answer": rounded,
    }


def main() -> dict:
    result = solve()
    print("Question 5: 5-fold cross-validation")
    for index, score in enumerate(result["fold_aucs"], start=1):
        print(f"  Fold {index}: {score:.6f}")
    print(f"  Mean AUC: {result['mean_auc']:.6f}")
    print(f"  Std (ddof=0): {result['std_auc']:.6f}")
    print(f"  Std (ddof=1): {result['sample_std']:.6f}")
    print(f"ANSWER: {result['std_rounded']:.3f}")
    if not result["matches_option"]:
        print(f"  Note: {result['std_rounded']:.3f} is outside the listed options {OPTIONS}.")
    return result


if __name__ == "__main__":
    main()
