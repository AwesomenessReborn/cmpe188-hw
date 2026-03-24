"""
Task: Multiclass Logistic Regression on Iris Dataset
Series: Logistic Regression
Level: 6

Math:
    Softmax: p_k = exp(z_k) / sum(exp(z_j))  for k in {0,1,2}
    Loss: CrossEntropyLoss (combines LogSoftmax + NLLLoss internally)

Dataset:
    sklearn.datasets.load_iris()
    150 samples, 4 features, 3 classes (setosa, versicolor, virginica)
    Split: 80% train / 20% validation
    Features must be standardized

Goal:
    Train a softmax regression model using nn.Linear(4, 3) + CrossEntropyLoss.
    Achieve validation accuracy > 0.90.
    Report per-class accuracy.
"""

import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris


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
    Load the Iris dataset, standardize features, and return DataLoaders.

    cfg keys to use:
        cfg['data']['train_ratio'] - fraction for training
        cfg['data']['batch_size']  - DataLoader batch size

    Steps:
        1. Load with load_iris()
        2. Standardize X using train set stats
        3. Labels should be torch.long (integer class indices)
        4. Split and wrap in DataLoader

    Returns:
        train_loader, val_loader
    """
    # TODO
    pass


def build_model(cfg: dict):
    """
    Build a multiclass logistic regression model.

    This is just nn.Linear(in_features, num_classes).
    CrossEntropyLoss handles the softmax internally, so NO softmax layer here.

    cfg keys to use:
        cfg['model']['in_features']  - 4 (iris features)
        cfg['model']['num_classes']  - 3 (iris classes)

    Returns:
        model (nn.Linear)
    """
    # TODO: return nn.Linear(in_features, num_classes)
    pass


def train(model, train_loader, cfg: dict, device):
    """
    Train using Adam and CrossEntropyLoss.

    cfg keys to use:
        cfg['training']['epochs'] - number of epochs
        cfg['training']['lr']     - learning rate

    Returns:
        loss_history (list of floats)
    """
    # TODO:
    # 1. optimizer = torch.optim.Adam(...)
    # 2. criterion = nn.CrossEntropyLoss()
    # 3. Training loop — labels must be torch.long
    pass


def evaluate(model, loader, device):
    """
    Evaluate model on a DataLoader.

    Returns:
        dict with keys: 'loss', 'accuracy', 'per_class_accuracy'
    """
    # TODO:
    # 1. Run inference with no_grad
    # 2. predictions = logits.argmax(dim=1)
    # 3. Overall accuracy + per-class accuracy (one value per class)
    pass


def predict(model, X: torch.Tensor, device):
    """
    Run inference and return predicted class indices.

    Returns:
        predicted class indices as a torch.Tensor (dtype=long)
    """
    # TODO: return logits.argmax(dim=1)
    pass


def save_artifacts(outputs: dict, cfg: dict):
    """
    Save any artifacts.

    cfg keys to use:
        cfg['output']['dir']
    """
    # TODO (optional): save loss curve or confusion matrix
    pass


# ---------------------------------------------------------------------------
# Main block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = {
        "data": {
            "train_ratio": 0.8,
            "batch_size": 16,
        },
        "model": {
            "in_features": 4,
            "num_classes": 3,
        },
        "training": {
            "epochs": 200,
            "lr": 0.01,
        },
        "output": {
            "dir": "outputs/logreg_iris_multiclass",
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
    print(f"Per-class Val Accuracy: {val_metrics['per_class_accuracy']}")

    outputs = {
        "loss_history": loss_history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    save_artifacts(outputs, cfg)

    try:
        assert val_metrics["accuracy"] > 0.90, f"Accuracy too low: {val_metrics['accuracy']:.4f}"
        print("All assertions passed.")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
