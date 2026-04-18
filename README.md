# CMPE 188 — Homework Repository

## Repository Structure

```
├── hw1/                         # HW1 — Original tasks (pytorch_task_v1 protocol)
│   ├── ml_tasks.json            # Task definitions
│   └── tasks/
│       ├── linreg_adam_optim/       # Univariate LinReg with Adam
│       ├── linreg_nn_linear/        # Multivariate LinReg with nn.Linear
│       ├── logreg_breast_cancer/    # Binary LogReg on Breast Cancer
│       └── logreg_iris_multiclass/  # Multiclass LogReg on Iris
│
├── hw1-ec/                      # HW1 Extra Credit — 4 new tasks
│   ├── ml_tasks.json            # Extended task definitions (+4 new entries)
│   └── tasks/
│       ├── linreg_lvl5_bq_housing/  # LinReg on Census data (BigQuery/sklearn)
│       ├── linreg_lvl6_adam_poly/   # Polynomial Reg: Adam vs SGD comparison
│       ├── logreg_lvl5_bq_income/   # LogReg on Census income classification
│       └── logreg_lvl6_multilabel/  # Multi-label classification (sigmoids)
│
├── hw2/                         # HW2 — Neural Network tasks (pytorch_task_v1 protocol)
│   └── tasks/
│       ├── mlp_diabetes_regression/  # 3-layer MLP on Diabetes dataset (regression)
│       ├── mlp_wine_multiclass/      # MLP + BatchNorm + Dropout on Wine (3-class)
│       ├── mlp_circles_binary/       # MLP on make_circles, SGD + ReduceLROnPlateau
│       └── mlp_digits_multiclass/    # MLP + AdamW + CosineAnnealingLR on Digits (10-class)
│
└── requirements.txt
```

All tasks follow the `pytorch_task_v1` protocol: each `task.py` implements 8 required functions and a `__main__` block that self-verifies with `sys.exit(0)` on pass or `sys.exit(1)` on failure.

---

## HW1 — Linear & Logistic Regression

Four tasks implementing Linear and Logistic Regression.

| Task ID | Algorithm | Dataset |
|---|---|---|
| `linreg_adam_optim` | Linear Regression | Synthetic (y = 3x + 7) |
| `linreg_nn_linear` | Multivariate LinReg | Synthetic (3 features) |
| `logreg_breast_cancer` | Binary Logistic Regression | sklearn Breast Cancer |
| `logreg_iris_multiclass` | Multiclass Logistic Regression | sklearn Iris |

Run:
```bash
python hw1/tasks/<task_id>/task.py
```

---

## HW1 Extra Credit — Four New Tasks

| Task ID | Algorithm | Dataset | Key Feature |
|---|---|---|---|
| `linreg_lvl5_bq_housing` | Linear Regression + L2 | Census Adult (BigQuery/sklearn) | BigQuery loading + sklearn comparison |
| `logreg_lvl5_bq_income` | Logistic Regression | Census Adult (BigQuery/sklearn) | Manual one-hot encoding + CrossEntropyLoss |
| `linreg_lvl6_adam_poly` | Polynomial Regression | Synthetic sin(x) | Adam + CosineAnnealingLR vs SGD convergence |
| `logreg_lvl6_multilabel` | Multi-Label Classification | Synthetic multi-label | Independent sigmoids + BCEWithLogitsLoss |

Extra credit tasks are Jupyter notebooks (`task.ipynb`).

Run:
```bash
jupyter nbconvert --to notebook --execute hw1-ec/tasks/<task_id>/task.ipynb
```

> BigQuery tasks fall back to `sklearn.datasets.fetch_openml('adult')` if `gcloud auth` is unavailable.

---

## HW2 — Neural Networks (MLP)

Four tasks applying multi-layer neural networks.

| Task ID | Algorithm | Dataset | New vs HW1 |
|---|---|---|---|
| `mlp_diabetes_regression` | 3-layer MLP Regression | sklearn Diabetes (442×10) | Deep MLP + Adam |
| `mlp_wine_multiclass` | MLP 3-class Classifier | sklearn Wine (178×13) | BatchNorm + Dropout |
| `mlp_circles_binary` | MLP Binary Classifier | make_circles (1000×2, non-linear) | SGD + ReduceLROnPlateau |
| `mlp_digits_multiclass` | MLP 10-class Classifier | sklearn Digits (1797×64) | AdamW + CosineAnnealingLR |

Run:
```bash
python hw2/tasks/<task_id>/task.py
```

---

## Setup

```bash
pip install -r requirements.txt
```
