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
