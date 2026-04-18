"""
Task: MLP Binary Classification on Non-Linearly Separable make_circles Dataset
Series: Neural Networks (MLP)
Level: 3

Math:
    h(x) = sigmoid(W3 * ReLU(W2 * ReLU(W1 * x + b1) + b2) + b3)
    Loss  = BCE = -mean(y * log(h) + (1 - y) * log(1 - h))

    A logistic regression model (linear decision boundary) scores ~50% on this
    dataset. The MLP with hidden ReLU layers learns a circular boundary.

Dataset:
    sklearn.datasets.make_circles(n_samples=1000, noise=0.1, factor=0.5)
    Two concentric circles — inner circle is class 0, outer is class 1.
    Non-linearly separable; requires a nonlinear decision boundary.
    Split: 80% train / 20% validation.

Goal:
    Train an MLP with SGD + momentum and a ReduceLROnPlateau scheduler.
    Demonstrates learning rate scheduling and neural network power on
    non-linear problems.
    Achieve: val accuracy > 0.92.
"""

import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_circles


def get_task_metadata():
    """Return a dict describing this task."""
    return {
        "id": "mlp_circles_binary",
        "series": "Neural Networks (MLP)",
        "level": 3,
        "algorithm": "MLP Binary Classifier with SGD + ReduceLROnPlateau",
        "description": (
            "3-layer MLP trained on sklearn make_circles (non-linearly separable). "
            "Uses SGD with momentum and ReduceLROnPlateau learning rate scheduling. "
            "Demonstrates neural networks solving problems impossible for logistic regression."
        ),
    }


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Return the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataloaders(cfg: dict):
    """
    Generate the make_circles dataset and return DataLoaders.

    cfg keys used:
        cfg['data']['n_samples']    - total samples to generate
        cfg['data']['noise']        - Gaussian noise added to data points
        cfg['data']['factor']       - scale factor between inner and outer circle
        cfg['data']['train_ratio']  - fraction for training (e.g. 0.8)
        cfg['data']['batch_size']   - DataLoader batch size

    Returns:
        train_loader, val_loader
    """
    X, y = make_circles(
        n_samples=cfg["data"]["n_samples"],
        noise=cfg["data"]["noise"],
        factor=cfg["data"]["factor"],
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    n = len(X)
    n_train = int(n * cfg["data"]["train_ratio"])

    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)  # [N, 1]
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    bs = cfg["data"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    return train_loader, val_loader


def build_model(cfg: dict):
    """
    Build a 3-layer MLP for binary classification.

    Architecture:
        Linear(2, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1) -> Sigmoid

    cfg keys used:
        cfg['model']['in_features']  - number of input features (2)
        cfg['model']['hidden1']      - first hidden layer size (32)
        cfg['model']['hidden2']      - second hidden layer size (16)

    Returns:
        model (nn.Module)
    """
    in_f = cfg["model"]["in_features"]
    h1 = cfg["model"]["hidden1"]
    h2 = cfg["model"]["hidden2"]
    return nn.Sequential(
        nn.Linear(in_f, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, 1),
        nn.Sigmoid(),
    )


def train(model, train_loader, cfg: dict, device):
    """
    Train using SGD with momentum and a ReduceLROnPlateau scheduler.

    cfg keys used:
        cfg['training']['epochs']    - number of epochs
        cfg['training']['lr']        - initial learning rate
        cfg['training']['momentum']  - SGD momentum

    Returns:
        loss_history (list of floats, one per epoch)
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )
    criterion = nn.BCELoss()

    loss_history = []
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
    return loss_history


def evaluate(model, loader, device):
    """
    Evaluate model on a DataLoader.

    Returns:
        dict with keys: 'loss', 'accuracy'
    """
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(y_batch)
            predicted = (preds >= 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += len(y_batch)
    return {"loss": total_loss / total, "accuracy": correct / total}


def predict(model, X: torch.Tensor, device):
    """
    Run inference on raw tensor X.

    Returns:
        predicted probabilities as a torch.Tensor of shape [N, 1]
    """
    model.eval()
    with torch.no_grad():
        return model(X.to(device)).cpu()


def save_artifacts(outputs: dict, cfg: dict):
    """
    Save any artifacts (optional).

    cfg keys used:
        cfg['output']['dir']
    """
    pass


# ---------------------------------------------------------------------------
# Main block — trains, evaluates, asserts quality, and exits with status code
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = {
        "data": {
            "n_samples": 1000,
            "noise": 0.1,
            "factor": 0.5,
            "train_ratio": 0.8,
            "batch_size": 32,
        },
        "model": {
            "in_features": 2,
            "hidden1": 32,
            "hidden2": 16,
        },
        "training": {
            "epochs": 300,
            "lr": 0.1,
            "momentum": 0.9,
        },
        "output": {
            "dir": "outputs/mlp_circles_binary",
        },
    }

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

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
        assert val_metrics["accuracy"] > 0.92, (
            f"Accuracy too low: {val_metrics['accuracy']:.4f}"
        )
        print("All assertions passed.")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
