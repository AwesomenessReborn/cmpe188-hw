"""
Task: MLP Multiclass Classification on Wine Dataset with BatchNorm + Dropout
Series: Neural Networks (MLP)
Level: 2

Math:
    h(x) = W3 * ReLU(BN(W2 * ReLU(BN(W1 * x + b1)) + b2)) + b3
    Loss  = CrossEntropyLoss = -sum(y_true * log(softmax(logits)))
    (softmax is applied internally by CrossEntropyLoss)

Dataset:
    sklearn.datasets.load_wine()
    178 samples, 13 features, 3 classes (cultivars of Italian wine)
    Split: 80% train / 20% validation
    Features must be standardized (zero mean, unit variance) using train stats only.

Goal:
    Train an MLP with BatchNorm1d and Dropout(0.3) using Adam and CrossEntropyLoss.
    Introduces regularization techniques not present in hw1 logistic regression tasks.
    Achieve: val accuracy > 0.90.
"""

import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine


def get_task_metadata():
    """Return a dict describing this task."""
    return {
        "id": "mlp_wine_multiclass",
        "series": "Neural Networks (MLP)",
        "level": 2,
        "algorithm": "MLP Multiclass Classifier with BatchNorm + Dropout",
        "description": (
            "MLP with BatchNorm1d and Dropout regularization trained on the "
            "sklearn Wine dataset for 3-class classification. Uses Adam optimizer "
            "and CrossEntropyLoss."
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
    Load the Wine dataset, standardize features, and return DataLoaders.

    cfg keys used:
        cfg['data']['train_ratio']  - fraction for training (e.g. 0.8)
        cfg['data']['batch_size']   - DataLoader batch size

    Returns:
        train_loader, val_loader
    """
    data = load_wine()
    X = data.data.astype(np.float32)    # (178, 13)
    y = data.target.astype(np.int64)    # (178,) — integer class labels

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
    Build an MLP classifier with BatchNorm1d and Dropout for 3-class output.

    Architecture:
        Linear(13, 64) -> BatchNorm1d(64) -> ReLU -> Dropout(0.3)
        -> Linear(64, 32) -> ReLU -> Linear(32, 3)

    cfg keys used:
        cfg['model']['in_features']  - number of input features (13)
        cfg['model']['hidden1']      - first hidden layer size (64)
        cfg['model']['hidden2']      - second hidden layer size (32)
        cfg['model']['n_classes']    - number of output classes (3)
        cfg['model']['dropout']      - dropout probability (0.3)

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
        nn.ReLU(),
        nn.Linear(h2, n_cls),
    )


def train(model, train_loader, cfg: dict, device):
    """
    Train using Adam optimizer and CrossEntropyLoss.

    cfg keys used:
        cfg['training']['epochs']  - number of epochs
        cfg['training']['lr']      - learning rate

    Returns:
        loss_history (list of floats, one per epoch)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
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
            "batch_size": 16,
        },
        "model": {
            "in_features": 13,
            "hidden1": 64,
            "hidden2": 32,
            "n_classes": 3,
            "dropout": 0.3,
        },
        "training": {
            "epochs": 200,
            "lr": 0.01,
        },
        "output": {
            "dir": "outputs/mlp_wine_multiclass",
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
        assert val_metrics["accuracy"] > 0.90, (
            f"Accuracy too low: {val_metrics['accuracy']:.4f}"
        )
        print("All assertions passed.")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
