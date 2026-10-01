# Lead scoring evaluation

Python solution for the Machine Learning Zoomcamp 2026 evaluation homework. The task is to score course leads and predict `converted`: whether the client signed up.

The dataset is the pinned 2026 release:

https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv

Each question lives in its own module and can be run by itself.

## Setup

```bash
pip install -r requirements.txt
```

The first run downloads the CSV into `data/` (that directory is gitignored).

## Run

All six questions:

```bash
python -m lead_scoring
```

One question:

```bash
python -m lead_scoring.prepare
python -m lead_scoring.question_1
python -m lead_scoring.question_2
python -m lead_scoring.question_3
python -m lead_scoring.question_4
python -m lead_scoring.question_5
python -m lead_scoring.question_6
```

Run these from the repository root so Python can import the `lead_scoring` package.

## What the code does

Data preparation (`lead_scoring/prepare.py`):

- Reports missing feature values.
- Fills categorical gaps with `NA` and numerical gaps with `0.0`.
- Splits with the homework calls: 20% test, then 25% of the remainder as validation (`random_state=1`), which is a 60/20/20 split.

| Module | Question |
| --- | --- |
| `question_1.py` | Univariate ROC AUC of the four numerical features on the training set. AUC below 0.5 is recomputed on the negated feature. |
| `question_2.py` | DictVectorizer plus `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`. Validation AUC, rounded to 3 digits. |
| `question_3.py` | Precision and recall from threshold 0.0 to 1.0 step 0.01. Chooses the threshold with the smallest absolute gap, ignoring thresholds where both are zero. Saves a plot. |
| `question_4.py` | F1 at the same thresholds. Reports the threshold where F1 is maximal. Saves a plot. |
| `question_5.py` | 5-fold AUC on `df_full_train` with `KFold(n_splits=5, shuffle=True, random_state=1)`. Reports the population standard deviation (`numpy.std`). |
| `question_6.py` | Same cross-validation for `C` in `[0.000001, 0.001, 1]`. Mean and std are rounded to 3 digits. Ties go to the lower std, then the smaller `C`. |

`python -m lead_scoring` writes `results/answers.json` plus the question 3 and 4 tables and plots.

## Answers

Missing values are present. Categorical gaps: `lead_source` 148, `industry` 240, `employment_status` 194, `location` 208. Numerical gaps: `annual_income` 369, `lead_score` 35. `number_of_courses_viewed` and `interaction_count` are complete. After imputation the feature matrix has no missing values. The split is 3000 / 1000 / 1000 (train / validation / test).

| Question | Answer | Detail |
| --- | --- | --- |
| 1 | `lead_score` | Training AUC 0.788. Next is `interaction_count` at 0.765. None of the four features had AUC below 0.5, so none were negated. |
| 2 | 0.732 | Validation ROC AUC is 0.731822, rounded to 3 digits. |
| 3 | 0.63 | Precision 0.733 and recall 0.725. Thresholds 0.97–1.00 are ignored because both metrics are zero. |
| 4 | 0.41 | Maximum F1 is 0.758 (precision 0.619, recall 0.976). |
| 5 | 0.007 | Fold AUCs average 0.732. Population std is 0.006853, which rounds to 0.007. |
| 6 | 0.001 | Rounded mean AUC: `C=0.000001` → 0.617, `C=0.001` → 0.749, `C=1` → 0.732. |

Question 5 uses `numpy.std` (divide by N). The sample standard deviation is 0.0077 and is not the figure the comparison uses.
