# CLAUDE.md — HW2

## Running Tasks

Each task is a self-contained script. Run with the `torch-default` conda environment:

```bash
/Users/hareee234/miniconda3/envs/torch-default/bin/python tasks/<task_id>/task.py
```

Example:
```bash
/Users/hareee234/miniconda3/envs/torch-default/bin/python tasks/mlp_diabetes_regression/task.py
```

Success: prints metrics and exits with code 0. Failure: prints assertion error and exits with code 1.

## Task Inventory (hw2 — Neural Networks)

| Task ID | Algorithm | Dataset | Key Features | Quality Threshold |
|---|---|---|---|---|
| `mlp_diabetes_regression` | 3-layer MLP Regression | sklearn Diabetes (442×10) | ReLU, Adam | R² > 0.45, MSE < 3500 |
| `mlp_wine_multiclass` | MLP 3-class Classifier | sklearn Wine (178×13) | BatchNorm, Dropout, Adam | Accuracy > 0.90 |
| `mlp_circles_binary` | MLP Binary Classifier | make_circles (1000×2) | SGD+momentum, ReduceLROnPlateau | Accuracy > 0.92 |
| `mlp_digits_multiclass` | MLP 10-class Classifier | sklearn Digits (1797×64) | BatchNorm, Dropout, AdamW, CosineAnnealingLR | Accuracy > 0.95 |

## Architecture (all tasks follow pytorch_task_v1 protocol)

Each `task.py` implements exactly 8 functions + a `__main__` block:

```python
get_task_metadata() -> dict
set_seed(seed=42) -> None
get_device() -> torch.device
make_dataloaders(cfg) -> (DataLoader, DataLoader)
build_model(cfg) -> nn.Module
train(model, train_loader, cfg, device) -> list   # loss_history
evaluate(model, loader, device) -> dict           # metrics
predict(model, X, device) -> Tensor
save_artifacts(outputs, cfg) -> None
```

## What's New vs HW1

HW1 used single-layer linear/logistic regression models. HW2 introduces:
- **Multi-layer MLPs** with ReLU nonlinearities
- **BatchNorm1d** — stabilizes training of deeper networks
- **Dropout** — regularization to prevent overfitting
- **AdamW** — Adam with decoupled weight decay
- **Learning rate schedulers** — ReduceLROnPlateau and CosineAnnealingLR
- **Non-linear datasets** — make_circles (impossible for logistic regression)
- **New real datasets** — Diabetes, Wine, Digits (different from HW1's Breast Cancer, Iris)
