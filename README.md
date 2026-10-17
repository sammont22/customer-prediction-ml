# Customer Subscription Prediction

End-to-end machine learning project that predicts whether a bank customer will subscribe to a term deposit.

The project covers exploratory data analysis, preprocessing, model comparison, hyperparameter tuning, decision-threshold selection, final evaluation, reusable Python modules, and automated tests.

## Project Objective

The objective is to identify customers who are more likely to subscribe to a term deposit, using information available before the outcome of a marketing call is known.

This is an imbalanced binary classification problem:

- `0` / `no`: the customer did not subscribe.
- `1` / `yes`: the customer subscribed.

## Dataset

The project uses the [Bank Marketing dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing).

The `bank-full.csv` file contains:

- 45,211 observations
- 16 input variables
- 1 binary target variable named `y`
- 39,922 negative observations (`88.30%`)
- 5,289 positive observations (`11.70%`)

The raw dataset is not included in the repository. Download `bank-full.csv` and place it in:

```text
data/raw/bank-full.csv
```

## Exploratory Findings

The exploratory analysis identified the following characteristics:

- No duplicate rows were detected.
- Pandas detected no explicit null values.
- Several categorical columns contain the value `unknown`, especially `poutcome` and `contact`.
- The target is strongly imbalanced, so accuracy alone is not sufficient for model evaluation.
- Several numerical variables have right-skewed distributions and extreme values.
- `pdays = -1` indicates that the customer had not previously been contacted.

The `unknown` values are retained as explicit categories because the dataset does not provide enough evidence to treat them as ordinary missing values.

## Preprocessing

The reusable preprocessing pipeline performs the following operations:

- Removes the target column from the model inputs.
- Excludes `duration` to prevent data leakage.
- Creates the binary feature `previously_contacted`.
- Replaces `pdays = -1` with `0` after preserving its meaning in `previously_contacted`.
- Applies median imputation to numerical features.
- Applies most-frequent imputation and one-hot encoding to categorical features.
- Ignores previously unseen categories during inference.

The `duration` feature is excluded because it records the duration of the current call and is only known after that call has taken place.

## Model Development

The following model configurations were compared using stratified cross-validation on the training data:

- Standard Logistic Regression
- Class-balanced Logistic Regression
- Random Forest
- HistGradientBoosting

HistGradientBoosting produced the best overall cross-validation results, with a mean ROC-AUC of `0.797` and mean average precision of `0.443` before tuning.

Hyperparameters were then selected with randomized search using five-fold stratified cross-validation. Average precision was used as the primary selection metric.

The selected model uses:

```text
learning_rate:        0.05
max_iter:             300
max_leaf_nodes:       63
min_samples_leaf:     10
l2_regularization:    10.0
class_weight:         balanced
random_state:         42
```

The final decision threshold stored with the model is:

```text
0.645058
```

This threshold was selected from out-of-fold training predictions by maximizing the F1-score. The held-out test set was not used for model or threshold selection.

## Final Test Results

The final model was evaluated once on a held-out test set containing 9,043 observations, including 1,058 subscribers.

| Metric | Result |
|---|---:|
| Accuracy | 0.8750 |
| Balanced accuracy | 0.7267 |
| Precision | 0.4700 |
| Recall | 0.5331 |
| F1-score | 0.4996 |
| ROC-AUC | 0.8060 |
| Average precision | 0.4674 |

Confusion matrix at the selected threshold:

| Actual class | Predicted no | Predicted yes |
|---|---:|---:|
| No | 7,349 | 636 |
| Yes | 494 | 564 |

The model correctly identifies 564 of the 1,058 subscribers in the test set. The selected threshold improves precision and F1-score relative to the default threshold of `0.5`, while reducing recall.

The threshold represents an F1-oriented operating point. A real marketing campaign may require a different threshold depending on contact costs and the value of identifying additional subscribers.

## Repository Structure

```text
customer-prediction-ml/
├── data/
│   ├── raw/
│   │   └── bank-full.csv
│   └── processed/
├── models/
│   └── final_model.joblib
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_model_tuning.ipynb
│   └── 06_final_evaluation.ipynb
├── src/
│   └── customer_prediction/
│       ├── __init__.py
│       ├── evaluate.py
│       ├── preprocessing.py
│       └── train.py
├── tests/
│   ├── test_evaluate.py
│   ├── test_preprocessing.py
│   └── test_train.py
├── .gitignore
├── pyproject.toml
└── README.md
```

The dataset and trained model are intentionally excluded from version control.

## Notebook Workflow

The notebooks document the complete development process:

1. `01_exploratory_analysis.ipynb` — dataset structure, data quality, target distribution, and feature analysis.
2. `02_preprocessing.ipynb` — target encoding, leakage prevention, data splitting, and preprocessing pipeline.
3. `03_baseline_model.ipynb` — majority-class baseline and Logistic Regression.
4. `04_model_comparison.ipynb` — cross-validation comparison of linear and tree-based models.
5. `05_model_tuning.ipynb` — HistGradientBoosting tuning and decision-threshold selection.
6. `06_final_evaluation.ipynb` — final evaluation on the held-out test set.

## Installation

The project requires Python 3.11 or later.

Clone the repository:

```bash
git clone https://github.com/sammont22/customer-prediction-ml.git
cd customer-prediction-ml
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Training the Model

After placing the dataset in `data/raw/`, train the final model with:

```bash
train-customer-prediction \
    --data data/raw/bank-full.csv \
    --output models/final_model.joblib
```

The saved artifact contains:

- The fitted preprocessing and classification pipeline
- The selected decision threshold
- The expected feature names
- The target mapping
- The excluded features
- The model parameters
- The scikit-learn version

## Generating Predictions

```python
import pandas as pd

from customer_prediction.evaluate import (
    load_model_artifact,
    predict_subscriptions,
)

artifact = load_model_artifact(
    "models/final_model.joblib"
)

new_customers = pd.read_csv(
    "new_customers.csv",
    sep=";",
)

predictions = predict_subscriptions(
    new_customers,
    artifact,
)

print(predictions.head())
```

The prediction output contains:

- `subscription_probability`
- `prediction`
- `prediction_label`

## Tests and Code Quality

Run the complete automated test suite:

```bash
pytest -q
```

Check the source code and tests with Ruff:

```bash
ruff check src tests
```

## Limitations

- The model was trained on historical data from a specific bank marketing campaign.
- Its performance may change with different customers, campaigns, or economic conditions.
- The selected threshold optimizes F1-score rather than direct business value.
- Campaign costs and expected subscription value were not available.
- Production deployment, monitoring, and drift detection are outside the current project scope.