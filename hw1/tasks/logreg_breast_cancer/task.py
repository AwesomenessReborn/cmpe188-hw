"""
Task: Binary Logistic Regression on Breast Cancer Dataset
Series: Logistic Regression
Level: 5

Math:
    sigma(z) = 1 / (1 + exp(-z))
    z = X @ W.T + b
    Loss = BCE = -mean(y*log(p) + (1-y)*log(1-p))

Dataset:
    sklearn.datasets.load_breast_cancer()
    569 samples, 30 features, binary labels (malignant=0, benign=1)
    Split: 80% train / 20% validation
    Features must be standardized (zero mean, unit variance)

Goal:
    Train a logistic regression model using nn.Linear + Sigmoid + BCELoss.
    Achieve validation accuracy > 0.92.
"""

import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer


def get_task_metadata():
    """Return a dict describing this task."""
    # TODO: return a dict with keys: id, series, level, algorithm, description
    pass


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    # TODO
    pass


def get_device():
    """Return the best available device."""
    # TODO
    pass


def make_dataloaders(cfg: dict):
    """
    Load the breast cancer dataset, standardize features, and return DataLoaders.

    cfg keys to use:
        cfg['data']['train_ratio'] - fraction for training
        cfg['data']['batch_size']  - DataLoader batch size

    Steps:
        1. Load with load_breast_cancer()
        2. Standardize X: subtract mean, divide by std (use train stats only)
        3. Convert to float32 tensors
        4. Split into train/val
        5. Wrap in TensorDataset and DataLoader

    Returns:
        train_loader, val_loader
    """
    # TODO
    pass


def build_model(cfg: dict):
    """
    Build a binary logistic regression model.

    This is nn.Linear -> Sigmoid.
    Use nn.Sequential to combine them.

    cfg keys to use:
        cfg['model']['in_features'] - number of input features (30)

    Returns:
        model (nn.Module)
    """
    # TODO: nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())
    pass


def train(model, train_loader, cfg: dict, device):
    """
    Train using Adam and BCELoss.

    cfg keys to use:
        cfg['training']['epochs'] - number of epochs
        cfg['training']['lr']     - learning rate

    Returns:
        loss_history (list of floats)
    """
    # TODO:
    # 1. optimizer = torch.optim.Adam(...)
    # 2. criterion = nn.BCELoss()
    # 3. Training loop — remember labels need shape [N, 1] to match output
    pass


def evaluate(model, loader, device):
    """
    Evaluate model on a DataLoader.

    Returns:
        dict with keys: 'loss', 'accuracy'
    """
    # TODO:
    # 1. Run inference with torch.no_grad()
    # 2. Threshold predictions at 0.5 to get binary labels
    # 3. Compute accuracy = correct / total
    # 4. Compute BCE loss
    pass


def predict(model, X: torch.Tensor, device):
    """
    Run inference on raw tensor X.

    Returns:
        predicted probabilities as a torch.Tensor
    """
    # TODO
    pass


def save_artifacts(outputs: dict, cfg: dict):
    """
    Save any artifacts.

    cfg keys to use:
        cfg['output']['dir']
    """
    # TODO (optional): save loss curve
    pass


# ---------------------------------------------------------------------------
# Main block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = {
        "data": {
            "train_ratio": 0.8,
            "batch_size": 32,
        },
        "model": {
            "in_features": 30,
        },
        "training": {
            "epochs": 100,
            "lr": 0.001,
        },
        "output": {
            "dir": "outputs/logreg_breast_cancer",
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

    print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
    print(f"Val   Loss: {val_metrics['loss']:.4f} | Val   Acc: {val_metrics['accuracy']:.4f}")

    outputs = {
        "loss_history": loss_history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    save_artifacts(outputs, cfg)

    try:
        assert val_metrics["accuracy"] > 0.92, f"Accuracy too low: {val_metrics['accuracy']:.4f}"
        print("All assertions passed.")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
