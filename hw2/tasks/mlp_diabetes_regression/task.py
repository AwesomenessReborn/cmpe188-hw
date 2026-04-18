"""
Task: MLP Regression on Diabetes Dataset
Series: Neural Networks (MLP)
Level: 1

Math:
    h(x) = W3 * ReLU(W2 * ReLU(W1 * x + b1) + b2) + b3
    MSE  = (1/N) * sum((h(x_i) - y_i)^2)
    R2   = 1 - SS_res / SS_tot

Dataset:
    sklearn.datasets.load_diabetes()
    442 samples, 10 features, continuous regression target
    (disease progression score one year after baseline)
    Split: 80% train / 20% validation
    Features must be standardized (zero mean, unit variance) using train stats only.

Goal:
    Train a 3-layer MLP [10 -> 64 -> 32 -> 1] with ReLU activations using Adam.
    Demonstrate that a multi-layer network outperforms simple linear regression on
    a real medical dataset.
    Achieve: val R2 > 0.45, val MSE < 3500.
"""

import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes


def get_task_metadata():
    """Return a dict describing this task."""
    return {
        "id": "mlp_diabetes_regression",
        "series": "Neural Networks (MLP)",
        "level": 1,
        "algorithm": "MLP Regression",
        "description": (
            "3-layer MLP with ReLU activations trained on the sklearn Diabetes "
            "dataset using Adam optimizer. Demonstrates deep regression beyond "
            "single linear layers."
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
    Load the diabetes dataset, standardize features, and return DataLoaders.

    cfg keys used:
        cfg['data']['train_ratio']  - fraction for training (e.g. 0.8)
        cfg['data']['batch_size']   - DataLoader batch size

    Returns:
        train_loader, val_loader
    """
    data = load_diabetes()
    X = data.data.astype(np.float32)   # (442, 10)
    y = data.target.astype(np.float32) # (442,)

    n = len(X)
    n_train = int(n * cfg["data"]["train_ratio"])

    # Shuffle with fixed seed before split
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
    Build a 3-layer MLP for regression.

    Architecture: Linear(10, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1)

    cfg keys used:
        cfg['model']['in_features']   - number of input features (10)
        cfg['model']['hidden1']       - first hidden layer size (64)
        cfg['model']['hidden2']       - second hidden layer size (32)

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
    )


def train(model, train_loader, cfg: dict, device):
    """
    Train using Adam optimizer and MSELoss.

    cfg keys used:
        cfg['training']['epochs']  - number of epochs
        cfg['training']['lr']      - learning rate

    Returns:
        loss_history (list of floats, one per epoch)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.MSELoss()
    model.train()

    loss_history = []
    for epoch in range(cfg["training"]["epochs"]):
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
        loss_history.append(epoch_loss / n_batches)
    return loss_history


def evaluate(model, loader, device):
    """
    Evaluate model on a DataLoader.

    Returns:
        dict with keys: 'mse', 'r2'
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu()
            all_preds.append(preds)
            all_targets.append(y_batch)

    preds_t = torch.cat(all_preds)       # [N, 1]
    targets_t = torch.cat(all_targets)   # [N, 1]

    mse = nn.functional.mse_loss(preds_t, targets_t).item()
    ss_res = ((targets_t - preds_t) ** 2).sum().item()
    ss_tot = ((targets_t - targets_t.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    return {"mse": mse, "r2": r2}


def predict(model, X: torch.Tensor, device):
    """
    Run inference on raw tensor X.

    Returns:
        predictions as a torch.Tensor of shape [N, 1]
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
    pass  # extend with matplotlib loss-curve plot if desired


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
            "in_features": 10,
            "hidden1": 64,
            "hidden2": 32,
        },
        "training": {
            "epochs": 200,
            "lr": 0.001,
        },
        "output": {
            "dir": "outputs/mlp_diabetes_regression",
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

    print(f"Train MSE: {train_metrics['mse']:.4f} | Train R2: {train_metrics['r2']:.4f}")
    print(f"Val   MSE: {val_metrics['mse']:.4f} | Val   R2: {val_metrics['r2']:.4f}")

    outputs = {
        "loss_history": loss_history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    save_artifacts(outputs, cfg)

    try:
        assert val_metrics["r2"] > 0.45, f"R2 too low: {val_metrics['r2']:.4f}"
        assert val_metrics["mse"] < 3500, f"MSE too high: {val_metrics['mse']:.4f}"
        print("All assertions passed.")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
