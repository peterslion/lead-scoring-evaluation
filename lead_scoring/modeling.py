"""One-hot encoding and logistic regression shared by later questions.

Each question module calls these helpers and keeps its own evaluation logic.
DictVectorizer is fit on the training fold only, so validation rows never
influence the encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from lead_scoring.prepare import FEATURE_COLUMNS, TARGET


def feature_dicts(frame: pd.DataFrame) -> list[dict]:
    """Records of model features. The target column is never included."""
    return frame[FEATURE_COLUMNS].to_dict(orient="records")


def target_array(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGET].to_numpy()


def train_model(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    C: float = 1.0,
) -> tuple[DictVectorizer, LogisticRegression]:
    """One-hot encode with DictVectorizer and fit liblinear logistic regression."""
    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform(feature_dicts(df_train))
    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
    model.fit(x_train, y_train)
    return vectorizer, model


def predict_scores(
    frame: pd.DataFrame,
    vectorizer: DictVectorizer,
    model: LogisticRegression,
) -> np.ndarray:
    """Positive-class probabilities for ``frame``."""
    matrix = vectorizer.transform(feature_dicts(frame))
    return model.predict_proba(matrix)[:, 1]


def fit_and_score_validation(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    C: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, DictVectorizer, LogisticRegression]:
    """Train on ``df_train`` and return validation labels and probabilities."""
    y_train = target_array(df_train)
    y_val = target_array(df_val)
    vectorizer, model = train_model(df_train, y_train, C=C)
    scores = predict_scores(df_val, vectorizer, model)
    return y_val, scores, vectorizer, model
