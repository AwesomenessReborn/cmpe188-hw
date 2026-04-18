"""
Task: MLP Multiclass Classification on Digits Dataset with BatchNorm, AdamW, CosineAnnealingLR
Series: Neural Networks (MLP)
Level: 4

Math:
    h(x) = W3 * ReLU(BN(dropout(ReLU(BN(W1*x + b1)))) + ...) + b3
    Loss  = CrossEntropyLoss (softmax + NLL internally)
    LR(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T_max))

Dataset:
    sklearn.datasets.load_digits()
    1797 samples, 64 features (flattened 8x8 pixel images), 10 classes (0-9)
    Split: 80% train / 20% validation
    Features standardized (zero mean, unit variance) using train stats only.

Goal:
    Train an MLP with BatchNorm1d + Dropout using AdamW (weight decay regularization)
    and CosineAnnealingLR scheduler for 10-class handwritten digit recognition.
    Showcases: AdamW optimizer, cosine LR annealing, BatchNorm + Dropout together.
    Achieve: val accuracy > 0.95.
"""

import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits


def get_task_metadata():
    """Return a dict describing this task."""
    return {
        "id": "mlp_digits_multiclass",
        "series": "Neural Networks (MLP)",
        "level": 4,
        "algorithm": "MLP 10-class Classifier with BatchNorm, AdamW, CosineAnnealingLR",
        "description": (
            "MLP with BatchNorm1d + Dropout trained on sklearn Digits (8x8 images) "
            "using AdamW optimizer with CosineAnnealingLR scheduling. "
            "10-class handwritten digit recognition."
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
    Load the Digits dataset, standardize features, and return DataLoaders.

    cfg keys used:
        cfg['data']['train_ratio']  - fraction for training (e.g. 0.8)
        cfg['data']['batch_size']   - DataLoader batch size

    Returns:
        train_loader, val_loader
    """
    data = load_digits()
    X = data.data.astype(np.float32)    # (1797, 64)
    y = data.target.astype(np.int64)    # (1797,) — integer class labels 0-9

    n = len(X)
    n_train = int(n * cfg["data"]["train_ratio"])

    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    # Standardize using train statistics only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    bs = cfg["data"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    return train_loader, val_loader


def build_model(cfg: dict):
    """
    Build an MLP with BatchNorm1d and Dropout for 10-class output.

    Architecture:
        Linear(64, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.2)
        -> Linear(128, 64) -> BatchNorm1d(64) -> ReLU
        -> Linear(64, 10)

    cfg keys used:
        cfg['model']['in_features']  - number of input features (64)
        cfg['model']['hidden1']      - first hidden layer size (128)
        cfg['model']['hidden2']      - second hidden layer size (64)
        cfg['model']['n_classes']    - number of output classes (10)
        cfg['model']['dropout']      - dropout probability (0.2)

    Returns:
        model (nn.Module)
    """
    in_f = cfg["model"]["in_features"]
    h1 = cfg["model"]["hidden1"]
    h2 = cfg["model"]["hidden2"]
    n_cls = cfg["model"]["n_classes"]
    p = cfg["model"]["dropout"]
    return nn.Sequential(
        nn.Linear(in_f, h1),
        nn.BatchNorm1d(h1),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(h1, h2),
        nn.BatchNorm1d(h2),
        nn.ReLU(),
        nn.Linear(h2, n_cls),
    )


def train(model, train_loader, cfg: dict, device):
    """
    Train using AdamW optimizer and CrossEntropyLoss with CosineAnnealingLR.

    cfg keys used:
        cfg['training']['epochs']        - number of epochs
        cfg['training']['lr']            - initial learning rate
        cfg['training']['weight_decay']  - AdamW weight decay
        cfg['training']['t_max']         - CosineAnnealingLR T_max

    Returns:
        loss_history (list of floats, one per epoch)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["t_max"]
    )
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        loss_history.append(epoch_loss / n_batches)
        scheduler.step()
    return loss_history


def evaluate(model, loader, device):
    """
    Evaluate model on a DataLoader.

    Returns:
        dict with keys: 'loss', 'accuracy'
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    return {"loss": total_loss / total, "accuracy": correct / total}


def predict(model, X: torch.Tensor, device):
    """
    Run inference on raw tensor X.

    Returns:
        predicted class indices as a torch.Tensor of shape [N]
    """
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        return logits.argmax(dim=1).cpu()


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
            "train_ratio": 0.8,
            "batch_size": 32,
        },
        "model": {
            "in_features": 64,
            "hidden1": 128,
            "hidden2": 64,
            "n_classes": 10,
            "dropout": 0.2,
        },
        "training": {
            "epochs": 100,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "t_max": 100,
        },
        "output": {
            "dir": "outputs/mlp_digits_multiclass",
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
        assert val_metrics["accuracy"] > 0.95, (
            f"Accuracy too low: {val_metrics['accuracy']:.4f}"
        )
        print("All assertions passed.")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
