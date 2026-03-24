"""
Task: Univariate Linear Regression with torch.optim (Adam)
Series: Linear Regression
Level: 5

Math:
    h(x) = theta_0 + theta_1 * x
    MSE = (1/N) * sum((h(x_i) - y_i)^2)

Dataset:
    Synthetic: y = 3x + 7 + noise
    Split: 80% train / 20% validation

Goal:
    Use torch.optim.Adam instead of manual gradient updates.
    Compare how many epochs it takes to converge vs manual GD.
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
    Generate synthetic data and return train/val DataLoaders.

    cfg keys to use:
        cfg['data']['n_samples']   - total number of samples
        cfg['data']['train_ratio'] - fraction for training (e.g. 0.8)
        cfg['data']['batch_size']  - batch size for DataLoader
        cfg['data']['noise_std']   - std of gaussian noise added to y

    Returns:
        train_loader, val_loader
    """
    # TODO:
    # 1. Generate X (random floats) and y = 3*X + 7 + noise
    # 2. Split into train/val
    # 3. Wrap in TensorDataset and DataLoader
    pass


def build_model(cfg: dict):
    """
    Build and return a simple linear model using nn.Linear.

    cfg keys to use:
        cfg['model']['in_features']  - number of input features (1 here)
        cfg['model']['out_features'] - number of outputs (1 here)

    Returns:
        model (nn.Module)
    """
    # TODO: return nn.Linear(in, out)
    pass


def train(model, train_loader, cfg: dict, device):
    """
    Train the model using Adam optimizer and MSELoss.

    cfg keys to use:
        cfg['training']['epochs']  - number of training epochs
        cfg['training']['lr']      - learning rate for Adam

    Returns:
        loss_history (list of floats, one per epoch)
    """
    # TODO:
    # 1. Set up torch.optim.Adam
    # 2. Set up nn.MSELoss
    # 3. Training loop: forward, loss, backward, optimizer step
    # 4. Track and return loss per epoch
    pass


def evaluate(model, loader, device):
    """
    Evaluate model on a DataLoader.

    Returns:
        dict with keys: 'mse', 'r2'
    """
    # TODO:
    # 1. Run inference (no grad)
    # 2. Compute MSE manually or with nn.MSELoss
    # 3. Compute R2 = 1 - SS_res / SS_tot
    pass


def predict(model, X: torch.Tensor, device):
    """
    Run inference on a raw tensor X.

    Returns:
        predictions as a torch.Tensor
    """
    # TODO: move X to device, run model in eval mode, return predictions
    pass


def save_artifacts(outputs: dict, cfg: dict):
    """
    Save any artifacts (loss curve plot, model weights, etc).

    outputs: dict returned by the main block
    cfg keys to use:
        cfg['output']['dir'] - directory to save files
    """
    # TODO (optional): save a loss curve plot using matplotlib
    pass


# ---------------------------------------------------------------------------
# Main block — trains, evaluates, asserts quality, and exits with status code
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = {
        "data": {
            "n_samples": 200,
            "train_ratio": 0.8,
            "batch_size": 32,
            "noise_std": 0.5,
        },
        "model": {
            "in_features": 1,
            "out_features": 1,
        },
        "training": {
            "epochs": 100,
            "lr": 0.01,
        },
        "output": {
            "dir": "outputs/linreg_adam_optim",
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

    # Print learned parameters
    # TODO: print model weight and bias, compare to true values (3.0, 7.0)

    outputs = {
        "loss_history": loss_history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    save_artifacts(outputs, cfg)

    # Quality thresholds — script exits non-zero if these fail
    try:
        assert val_metrics["r2"] > 0.9, f"R2 too low: {val_metrics['r2']:.4f}"
        assert val_metrics["mse"] < 1.0, f"MSE too high: {val_metrics['mse']:.4f}"
        print("All assertions passed.")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
