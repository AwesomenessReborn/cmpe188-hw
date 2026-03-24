# CMPE 188 — HW1: Linear & Logistic Regression Tasks

## Repository Structure

```
├── hw1/                         # Original HW1 tasks (pytorch_task_v1 protocol)
│   ├── ml_tasks.json            # Task definitions (original 72 tasks)
│   └── tasks/
│       ├── linreg_adam_optim/       # Univariate LinReg with Adam
│       ├── linreg_nn_linear/        # Multivariate LinReg with nn.Linear
│       ├── logreg_breast_cancer/    # Binary LogReg on Breast Cancer
│       └── logreg_iris_multiclass/  # Multiclass LogReg on Iris
│
├── hw1-ec/                      # Extra Credit — 4 new tasks
│   ├── ml_tasks.json            # Extended task definitions (+4 new entries)
│   └── tasks/
│       ├── linreg_lvl5_bq_housing/  # LinReg on Census data (BigQuery/sklearn)
│       ├── linreg_lvl6_adam_poly/   # Polynomial Reg: Adam vs SGD comparison
│       ├── logreg_lvl5_bq_income/   # LogReg on Census income classification
│       └── logreg_lvl6_multilabel/  # Multi-label classification (sigmoids)
│
└── requirements.txt
```

## HW1 — Original Tasks

Four tasks implementing Linear and Logistic Regression following the `pytorch_task_v1` protocol. Each task is a self-contained `task.py` with all required functions (`get_task_metadata`, `set_seed`, `get_device`, `make_dataloaders`, `build_model`, `train`, `evaluate`, `predict`, `save_artifacts`).

## HW1 Extra Credit — Four New Tasks

| Task ID | Algorithm | Dataset | Key Feature |
|---|---|---|---|
| `linreg_lvl5_bq_housing` | Linear Regression + L2 | Census Adult (BigQuery/sklearn) | BigQuery loading + sklearn comparison |
| `logreg_lvl5_bq_income` | Logistic Regression | Census Adult (BigQuery/sklearn) | Manual one-hot encoding + CrossEntropyLoss |
| `linreg_lvl6_adam_poly` | Polynomial Regression | Synthetic sin(x) | Adam+CosineAnnealingLR vs SGD convergence |
| `logreg_lvl6_multilabel` | Multi-Label Classification | Synthetic multi-label | Independent sigmoids + BCEWithLogitsLoss |

Extra credit tasks are implemented as Jupyter notebooks (`task.ipynb`).

### Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run a notebook
jupyter nbconvert --to notebook --execute hw1-ec/tasks/<task_id>/task.ipynb

# Or open interactively
jupyter notebook hw1-ec/tasks/<task_id>/task.ipynb
```

Each notebook exits with status 0 on success (all quality assertions pass) or status 1 on failure.

### BigQuery Tasks

Tasks `linreg_lvl5_bq_housing` and `logreg_lvl5_bq_income` include BigQuery loading code that requires `gcloud auth application-default login`. If BigQuery is unavailable, they automatically fall back to loading the same UCI Census Adult dataset via `sklearn.datasets.fetch_openml('adult')`.
