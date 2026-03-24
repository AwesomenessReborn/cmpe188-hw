"""
Task: Multivariate Linear Regression with nn.Linear
Series: Linear Regression
Level: 6

Math:
    h(X) = X @ W.T + b
    MSE = (1/N) * sum((h(X_i) - y_i)^2)

Dataset:
    Synthetic: y = 2*x1 + (-3)*x2 + 1*x3 + noise
    X has shape [N, 3] — three input features
    Split: 80% train / 20% validation

Goal:
    Use nn.Linear with multiple input features.
    Verify the model learns weights close to [2, -3, 1] and bias close to 0.
"""

import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_task_metadata():
    """Return a dict describing this task."""
    # TODO: return a dict with keys: id, series, level, algorithm, description
    pass


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    # TODO: set torch manual seed
    pass


def get_device():
    """Return the best available device (cuda, mps, or cpu)."""
    # TODO: return torch.device(...)
    pass


def make_dataloaders(cfg: dict):
    """
    Generate synthetic multivariate data and return train/val DataLoaders.

    cfg keys to use:
        cfg['data']['n_samples']    - total number of samples
        cfg['data']['n_features']   - number of input features
        cfg['data']['true_weights'] - list of true weights (one per feature)
        cfg['data']['true_bias']    - true bias term
        cfg['data']['train_ratio']  - fraction for training
        cfg['data']['batch_size']   - DataLoader batch size
        cfg['data']['noise_std']    - noise standard deviation

    Returns:
        train_loader, val_loader
    """
    # TODO:
    # 1. Generate X with shape [n_samples, n_features]
    # 2. Compute y = X @ true_weights + true_bias + noise
    # 3. Split and wrap in DataLoader
    pass


def build_model(cfg: dict):
    """
    Build a linear model for multivariate regression.

    cfg keys to use:
        cfg['model']['in_features']  - number of input features
        cfg['model']['out_features'] - 1 (predicting a scalar)

    Returns:
        model (nn.Linear)
    """
    # TODO: return nn.Linear(in_features, out_features)
    pass


def train(model, train_loader, cfg: dict, device):
    """
    Train using SGD optimizer and MSELoss.

    cfg keys to use:
        cfg['training']['epochs']    - number of epochs
        cfg['training']['lr']        - learning rate
        cfg['training']['momentum']  - SGD momentum

    Returns:
        loss_history (list of floats)
    """
    # TODO:
    # 1. Set up torch.optim.SGD with momentum
    # 2. Set up nn.MSELoss
    # 3. Training loop
    pass


def evaluate(model, loader, device):
    """
    Evaluate model on a DataLoader.

    Returns:
        dict with keys: 'mse', 'r2'
    """
    # TODO: compute MSE and R2 on loader data
    pass


def predict(model, X: torch.Tensor, device):
    """
    Run inference on raw tensor X.

    Returns:
        predictions as a torch.Tensor
    """
    # TODO
    pass


def save_artifacts(outputs: dict, cfg: dict):
    """
    Save any artifacts (plots, weights, etc).

    cfg keys to use:
        cfg['output']['dir']
    """
    # TODO (optional): save loss curve plot
    pass


# ---------------------------------------------------------------------------
# Main block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = {
        "data": {
            "n_samples": 300,
            "n_features": 3,
            "true_weights": [2.0, -3.0, 1.0],
            "true_bias": 0.0,
            "train_ratio": 0.8,
            "batch_size": 32,
            "noise_std": 0.3,
        },
        "model": {
            "in_features": 3,
            "out_features": 1,
        },
        "training": {
            "epochs": 200,
            "lr": 0.05,
            "momentum": 0.9,
        },
        "output": {
            "dir": "outputs/linreg_nn_linear",
        },
    }

    set_seed(42)
    device = get_device()

    train_loader, val_loader = make_dataloaders(cfg)
    model = build_model(cfg)
    model = model.to(device)

    loss_history = train(model, train_loader, cfg, device)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print(f"Train MSE: {train_metrics['mse']:.4f} | Train R2: {train_metrics['r2']:.4f}")
    print(f"Val   MSE: {val_metrics['mse']:.4f} | Val   R2: {val_metrics['r2']:.4f}")

    # TODO: print learned weights vs true weights [2.0, -3.0, 1.0]

    outputs = {
        "loss_history": loss_history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    save_artifacts(outputs, cfg)

    try:
        assert val_metrics["r2"] > 0.9, f"R2 too low: {val_metrics['r2']:.4f}"
        assert val_metrics["mse"] < 1.0, f"MSE too high: {val_metrics['mse']:.4f}"
        print("All assertions passed.")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
