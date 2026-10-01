"""Question 2: logistic regression AUC on the validation set.

One-hot encode the training features with DictVectorizer and fit
``LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)``.
The reported figure is the validation ROC AUC rounded to 3 digits.
"""

from __future__ import annotations

from sklearn.metrics import roc_auc_score

from lead_scoring.modeling import fit_and_score_validation
from lead_scoring.prepare import prepare

OPTIONS = (0.532, 0.632, 0.732, 0.832)


def solve() -> dict:
    data = prepare()
    y_val, scores, vectorizer, _model = fit_and_score_validation(data.train, data.validation)
    auc = float(roc_auc_score(y_val, scores))
    rounded = round(auc, 3)
    return {
        "auc": auc,
        "auc_rounded": rounded,
        "n_features": int(len(vectorizer.get_feature_names_out())),
        "matches_option": rounded in OPTIONS,
        "answer": rounded,
    }


def main() -> dict:
    result = solve()
    print("Question 2: validation ROC AUC of logistic regression")
    print(f"  Encoded features: {result['n_features']}")
    print(f"  AUC: {result['auc']:.6f}")
    print(f"ANSWER: {result['auc_rounded']:.3f}")
    if not result["matches_option"]:
        print(f"  Note: {result['auc_rounded']:.3f} is outside the listed options {OPTIONS}.")
    return result


if __name__ == "__main__":
    main()
