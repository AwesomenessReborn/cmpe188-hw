"""
Task: MLP Binary Classifier on HackerNews Comments — BigQuery Bigframe + Text Embedding
Series: Neural Networks (MLP) — BigQuery Extra Credit
Level: 5

Data Pipeline:
    bigframes.pandas loads HackerNews comment text + scores directly from BigQuery.
    BigQuery ML TextEmbeddingGenerator (text-embedding-004) converts each comment
    into a 768-dim dense vector. An MLP then classifies comments as high-quality
    (score >= median) or low-quality (score < median).

What's New vs hw2:
    - Data loaded from BigQuery via bigframes.pandas (not sklearn datasets).
    - Text embeddings as input features — 768-dim NLP representation.
    - First use of BigQuery ML remote embedding API (TextEmbeddingGenerator).
    - End-to-end pipeline: BigQuery → Embedding API → MLP → Evaluation.
    - Embeddings cached to disk so subsequent runs are instant.

Dataset:
    bigquery-public-data.hacker_news.comments (public BigQuery dataset)
    Columns used: text (string), score (int)
    Sample: 2000 rows (LIMIT for cost control)
    Embedding model: text-embedding-004 (output dim: 768)

Architecture:
    Linear(768, 256) → BatchNorm1d(256) → ReLU → Dropout(0.3)
    → Linear(256, 64) → BatchNorm1d(64) → ReLU
    → Linear(64, 1)
    Loss:      BCEWithLogitsLoss
    Optimizer: Adam(lr=0.001)

Goal:
    Predict whether a HackerNews comment is high-quality from its text embedding.
    Achieve: val accuracy > 0.60, val F1 > 0.55
"""

import os
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# GCP / BigQuery configuration
# ---------------------------------------------------------------------------
GCP_PROJECT_ID = "gen-lang-client-0916541599"
BQ_CONNECTION_NAME = "us.vertex-ai"  # TODO: set to your BigQuery ML connection name
                                      # Find it: bq ls --connection \
                                      #   --project_id=gen-lang-client-0916541599 --location=us


# ---------------------------------------------------------------------------
# pytorch_task_v1 protocol — 8 required functions
# ---------------------------------------------------------------------------

def get_task_metadata() -> dict:
    """Return a dict describing this task."""
    return {
        "id": "mlp_bq_text_embedding_hn",
        "series": "Neural Networks (MLP) — BigQuery Extra Credit",
        "level": 5,
        "algorithm": "MLP Binary Classifier on Text Embeddings (BigQuery Bigframe)",
        "description": (
            "Load HackerNews comments from BigQuery via bigframes.pandas. "
            "Generate 768-dim text embeddings using BigQuery ML TextEmbeddingGenerator "
            "(text-embedding-004). Train a 3-layer MLP with BatchNorm + Dropout to "
            "classify comment quality (high vs low score)."
        ),
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    """Return the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_or_generate_embeddings(cfg: dict) -> tuple:
    """
    Load BigQuery data via bigframes, generate text embeddings, and cache results.

    First call: queries BigQuery → calls TextEmbeddingGenerator → saves .npy files.
    Subsequent calls: loads cached .npy files directly (fast, no API cost).

    Returns:
        X_all (np.ndarray, float32): shape [N, 768] — one embedding per comment
        y_all (np.ndarray, int64):   shape [N]      — binary quality label
    """
    out_dir = cfg["output"]["dir"]
    emb_cache = os.path.join(out_dir, "embedding_cache.npy")
    lbl_cache = os.path.join(out_dir, "labels_cache.npy")

    if os.path.exists(emb_cache) and os.path.exists(lbl_cache):
        print("Loading embeddings from cache (skipping BigQuery API call)...")
        X_all = np.load(emb_cache)
        y_all = np.load(lbl_cache)
        print(f"  Loaded {len(X_all)} samples, embedding dim={X_all.shape[1]}")
        return X_all, y_all

    print("Step 1: Loading HackerNews comments from BigQuery via bigframes.pandas...")
    import bigframes.pandas as bpd
    from bigframes.ml.llm import TextEmbeddingGenerator

    bpd.options.bigquery.project = GCP_PROJECT_ID
    bpd.options.bigquery.location = "US"

    bdf = bpd.read_gbq(cfg["data"]["bq_query"])
    print(f"  Loaded {len(bdf)} rows from BigQuery: hacker_news.comments")

    # TextEmbeddingGenerator expects a column named 'content'
    bdf_content = bdf[["text"]].rename(columns={"text": "content"})

    print("Step 2: Generating text embeddings via BigQuery ML (text-embedding-004)...")
    print(f"  Using connection: {BQ_CONNECTION_NAME}")
    emb_model = TextEmbeddingGenerator(
        model_name="text-embedding-004",
        connection_name=BQ_CONNECTION_NAME,
    )
    emb_bdf = emb_model.predict(bdf_content)

    # Convert to pandas for numpy processing
    emb_pdf = emb_bdf.to_pandas()
    scores_pdf = bdf[["score"]].to_pandas()

    # Extract 768-dim embedding vectors
    X_all = np.array(emb_pdf["text_embedding"].tolist(), dtype=np.float32)

    # Binary label: score >= median → 1 (high quality), else → 0
    scores = scores_pdf["score"].values.astype(np.float32)
    threshold = float(np.median(scores))
    y_all = (scores >= threshold).astype(np.int64)

    label_counts = np.bincount(y_all)
    print(f"  Embedding shape: {X_all.shape}")
    print(f"  Label distribution: low={label_counts[0]}, high={label_counts[1]}, threshold={threshold:.1f}")

    # Cache to disk
    os.makedirs(out_dir, exist_ok=True)
    np.save(emb_cache, X_all)
    np.save(lbl_cache, y_all)
    print(f"  Embeddings cached to {out_dir}")

    return X_all, y_all


def make_dataloaders(cfg: dict):
    """
    Load BigQuery data, generate embeddings (or load from cache), split, and return DataLoaders.

    BigQuery Bigframe pipeline:
        1. bpd.read_gbq() → bigframes DataFrame of HN comments
        2. TextEmbeddingGenerator.predict() → 768-dim embeddings per comment
        3. Convert to numpy, binary label from score >= median
        4. Normalize embeddings using train-set mean/std
        5. Return (train_loader, val_loader)

    cfg keys used:
        cfg['data']['bq_query']     - SQL query string for BigQuery
        cfg['data']['train_ratio']  - fraction for training
        cfg['data']['batch_size']   - DataLoader batch size
        cfg['output']['dir']        - directory for embedding cache files

    Returns:
        train_loader, val_loader
    """
    X_all, y_all = _load_or_generate_embeddings(cfg)

    n = len(X_all)
    n_train = int(n * cfg["data"]["train_ratio"])

    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    X_all, y_all = X_all[idx], y_all[idx]

    X_train, X_val = X_all[:n_train], X_all[n_train:]
    y_train, y_val = y_all[:n_train], y_all[n_train:]

    # Normalize using train statistics only (standard practice)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # BCEWithLogitsLoss expects float targets
    train_ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val, dtype=torch.float32),
    )

    bs = cfg["data"]["batch_size"]
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Input dim: {X_train.shape[1]}")
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False),
    )


def build_model(cfg: dict) -> nn.Module:
    """
    Build an MLP for binary classification on 768-dim text embeddings.

    Architecture:
        Linear(768, 256) → BatchNorm1d(256) → ReLU → Dropout(0.3)
        → Linear(256, 64) → BatchNorm1d(64) → ReLU
        → Linear(64, 1)                         ← raw logit (no sigmoid here)

    The BCEWithLogitsLoss in train() applies sigmoid internally for numerical stability.

    cfg keys used:
        cfg['model']['in_features']  - embedding dimension (768)
        cfg['model']['hidden1']      - first hidden layer size (256)
        cfg['model']['hidden2']      - second hidden layer size (64)
        cfg['model']['dropout']      - dropout probability (0.3)

    Returns:
        model (nn.Module)
    """
    in_f = cfg["model"]["in_features"]
    h1 = cfg["model"]["hidden1"]
    h2 = cfg["model"]["hidden2"]
    p = cfg["model"]["dropout"]
    return nn.Sequential(
        nn.Linear(in_f, h1),
        nn.BatchNorm1d(h1),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(h1, h2),
        nn.BatchNorm1d(h2),
        nn.ReLU(),
        nn.Linear(h2, 1),
    )


def train(model, train_loader, cfg: dict, device) -> list:
    """
    Train using Adam + BCEWithLogitsLoss.

    cfg keys used:
        cfg['training']['epochs']  - number of epochs
        cfg['training']['lr']      - learning rate

    Returns:
        loss_history (list of floats, one per epoch)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.BCEWithLogitsLoss()

    loss_history = []
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)  # [B] → [B, 1]
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{cfg['training']['epochs']} | Loss: {avg_loss:.4f}")
    return loss_history


def evaluate(model, loader, device) -> dict:
    """
    Evaluate model on a DataLoader.

    Returns:
        dict with keys: 'loss', 'accuracy', 'f1'
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch_dev = y_batch.to(device).unsqueeze(1)
            logits = model(X_batch)
            loss = criterion(logits, y_batch_dev)
            total_loss += loss.item() * len(y_batch)

            preds = (torch.sigmoid(logits.squeeze()) >= 0.5).long().cpu()
            all_preds.append(preds)
            all_targets.append(y_batch.long())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    n = len(targets)

    accuracy = (preds == targets).float().mean().item()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "loss": total_loss / n,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def predict(model, X: torch.Tensor, device) -> torch.Tensor:
    """
    Run inference on raw tensor X (pre-normalized embeddings).

    Returns:
        predicted class indices as a torch.Tensor of shape [N] (0 or 1)
    """
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        return (torch.sigmoid(logits.squeeze()) >= 0.5).long().cpu()


def save_artifacts(outputs: dict, cfg: dict) -> None:
    """Save metrics JSON to output directory."""
    import json
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    saveable = {
        k: v
        for k, v in outputs.items()
        if isinstance(v, (int, float, str, dict, list))
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(saveable, f, indent=2, default=str)
    print(f"Metrics saved to {out_dir}/metrics.json")


# ---------------------------------------------------------------------------
# Main block — trains, evaluates, asserts quality, exits with status code
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = {
        "data": {
            "bq_query": (
                "SELECT text, score "
                "FROM `bigquery-public-data.hacker_news.comments` "
                "WHERE text IS NOT NULL "
                "  AND score IS NOT NULL "
                "  AND LENGTH(text) > 20 "
                "LIMIT 2000"
            ),
            "train_ratio": 0.8,
            "batch_size": 64,
        },
        "model": {
            "in_features": 768,   # text-embedding-004 output dimension
            "hidden1": 256,
            "hidden2": 64,
            "dropout": 0.3,
        },
        "training": {
            "epochs": 30,
            "lr": 0.001,
        },
        "output": {
            "dir": "output/mlp_bq_text_embedding_hn",
        },
    }

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print(f"Task: {get_task_metadata()['id']}")
    print()

    train_loader, val_loader = make_dataloaders(cfg)
    model = build_model(cfg).to(device)

    print("\n--- Training ---")
    loss_history = train(model, train_loader, cfg, device)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print(f"\nTrain | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")
    print(f"Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")

    outputs = {
        "loss_history": loss_history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    save_artifacts(outputs, cfg)

    print("\n--- Quality Assertions ---")
    try:
        assert val_metrics["accuracy"] > 0.60, (
            f"Accuracy too low: {val_metrics['accuracy']:.4f} (threshold: 0.60)"
        )
        print(f"PASS: val accuracy {val_metrics['accuracy']:.4f} > 0.60")

        assert val_metrics["f1"] > 0.55, (
            f"F1 too low: {val_metrics['f1']:.4f} (threshold: 0.55)"
        )
        print(f"PASS: val F1 {val_metrics['f1']:.4f} > 0.55")

        print("\nAll assertions passed.")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
