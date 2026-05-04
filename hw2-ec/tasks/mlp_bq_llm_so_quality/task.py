"""
Task: MLP Binary Classifier on Stack Overflow Questions — BigQuery Bigframe + Gemini LLM
Series: Neural Networks (MLP) — BigQuery Extra Credit
Level: 5

Data Pipeline:
    bigframes.pandas loads Stack Overflow question metadata from BigQuery.
    BigQuery ML GeminiTextGenerator (gemini-2.0-flash-001) analyzes each question
    title and rates its technical difficulty on a 1-5 scale. This LLM-derived
    difficulty score is combined with numeric metadata (answer count, view count,
    post score) to form a 4-feature input for an MLP that predicts whether a
    question received an accepted answer.

What's New vs hw2:
    - Data loaded from BigQuery via bigframes.pandas (not sklearn datasets).
    - LLM used as a feature extractor — Gemini scores question difficulty from text.
    - Hybrid feature space: LLM-derived structured feature + numeric metadata.
    - New dataset: Stack Overflow (first use in this series).
    - Demonstrates complete pipeline: BigQuery → LLM feature extraction → MLP → Evaluation.
    - LLM results cached to disk so subsequent runs are instant.

Dataset:
    bigquery-public-data.stackoverflow.posts_questions (public BigQuery dataset)
    Columns used: title (string), answer_count (int), view_count (int),
                  score (int), accepted_answer_id (int, nullable)
    Sample: 500 rows (LIMIT for cost control)

LLM Feature:
    GeminiTextGenerator prompt: "Rate difficulty 1-5: {title}"
    Parsed response → integer 1-5 (default=3 on parse failure)

Architecture:
    Linear(4, 64) → BatchNorm1d(64) → ReLU → Dropout(0.2)
    → Linear(64, 32) → BatchNorm1d(32) → ReLU
    → Linear(32, 1)
    Loss:      BCEWithLogitsLoss
    Optimizer: Adam(lr=0.001)

Goal:
    Predict whether a Stack Overflow question has an accepted answer.
    Achieve: val accuracy > 0.62
"""

import os
import re
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

# LLM prompt template — appended to each question title
_LLM_PROMPT_PREFIX = (
    "You are an expert software engineer. Rate the technical difficulty of this "
    "Stack Overflow question on a scale of 1 to 5, where 1 means beginner-level "
    "and 5 means expert-level. Reply with ONLY a single digit (1, 2, 3, 4, or 5).\n\n"
    "Question: "
)


# ---------------------------------------------------------------------------
# pytorch_task_v1 protocol — 8 required functions
# ---------------------------------------------------------------------------

def get_task_metadata() -> dict:
    """Return a dict describing this task."""
    return {
        "id": "mlp_bq_llm_so_quality",
        "series": "Neural Networks (MLP) — BigQuery Extra Credit",
        "level": 5,
        "algorithm": "MLP Binary Classifier with LLM Feature Extraction (BigQuery Bigframe + Gemini)",
        "description": (
            "Load Stack Overflow questions from BigQuery via bigframes.pandas. "
            "Use BigQuery ML GeminiTextGenerator to extract a difficulty score (1-5) "
            "from each question title. Combine LLM-derived score with numeric metadata "
            "(answer_count, log_view_count, post score) and train a 3-layer MLP with "
            "BatchNorm + Dropout to predict whether a question received an accepted answer."
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


def _parse_difficulty(response: str) -> float:
    """
    Parse a 1–5 integer from Gemini's response string.
    Falls back to 3 (neutral) if no valid digit is found.
    """
    match = re.search(r"[1-5]", str(response))
    return float(match.group()) if match else 3.0


def _load_or_generate_features(cfg: dict) -> tuple:
    """
    Load BigQuery data via bigframes, call Gemini for LLM difficulty scores,
    and cache results.

    First call: queries BigQuery → calls GeminiTextGenerator → saves .npy files.
    Subsequent calls: loads cached .npy files directly (fast, no API cost).

    Features (per question):
        0: answer_count          (numeric, standardized)
        1: log1p(view_count)     (numeric, log-scaled, standardized)
        2: score                 (numeric, standardized)
        3: llm_difficulty (1–5)  (LLM-derived, standardized)

    Label: 1 if accepted_answer_id IS NOT NULL, else 0

    Returns:
        X_all (np.ndarray, float32): shape [N, 4]
        y_all (np.ndarray, int64):   shape [N]
    """
    out_dir = cfg["output"]["dir"]
    feat_cache = os.path.join(out_dir, "features_cache.npy")
    lbl_cache = os.path.join(out_dir, "labels_cache.npy")

    if os.path.exists(feat_cache) and os.path.exists(lbl_cache):
        print("Loading features from cache (skipping BigQuery API call)...")
        X_all = np.load(feat_cache)
        y_all = np.load(lbl_cache)
        print(f"  Loaded {len(X_all)} samples, {X_all.shape[1]} features")
        return X_all, y_all

    print("Step 1: Loading Stack Overflow questions from BigQuery via bigframes.pandas...")
    import bigframes.pandas as bpd
    from bigframes.ml.llm import GeminiTextGenerator

    bpd.options.bigquery.project = GCP_PROJECT_ID
    bpd.options.bigquery.location = "US"

    bdf = bpd.read_gbq(cfg["data"]["bq_query"])
    print(f"  Loaded {len(bdf)} rows from BigQuery: stackoverflow.posts_questions")

    # Build LLM prompt: prefix + title
    bdf = bdf.assign(prompt=_LLM_PROMPT_PREFIX + bdf["title"])

    print("Step 2: Extracting difficulty scores via BigQuery ML (gemini-2.0-flash-001)...")
    print(f"  Using connection: {BQ_CONNECTION_NAME}")
    llm_model = GeminiTextGenerator(
        model_name="gemini-2.0-flash-001",
        connection_name=BQ_CONNECTION_NAME,
    )
    llm_bdf = llm_model.predict(bdf[["prompt"]])

    # Convert to pandas
    llm_pdf = llm_bdf.to_pandas()
    meta_pdf = bdf[["answer_count", "view_count", "score", "accepted_answer_id"]].to_pandas()

    # Parse LLM responses → difficulty score 1–5
    llm_difficulty = llm_pdf["ml_generate_text_llm_result"].map(_parse_difficulty).values.astype(np.float32)
    parse_failures = int((llm_pdf["ml_generate_text_llm_result"].map(
        lambda r: re.search(r"[1-5]", str(r)) is None
    )).sum())
    if parse_failures:
        print(f"  Warning: {parse_failures}/{len(llm_pdf)} LLM responses defaulted to 3 (parse failure)")

    # Assemble feature matrix
    answer_count = meta_pdf["answer_count"].values.astype(np.float32)
    log_view = np.log1p(meta_pdf["view_count"].values.astype(np.float32))
    score = meta_pdf["score"].values.astype(np.float32)

    X_all = np.stack([answer_count, log_view, score, llm_difficulty], axis=1)

    # Binary label: has accepted answer → 1, else → 0
    y_all = meta_pdf["accepted_answer_id"].notna().astype(np.int64).values

    label_counts = np.bincount(y_all)
    print(f"  Feature shape: {X_all.shape}")
    print(f"  Label distribution: no_accepted={label_counts[0]}, accepted={label_counts[1]}")
    print(f"  LLM difficulty scores — mean: {llm_difficulty.mean():.2f}, std: {llm_difficulty.std():.2f}")

    # Cache to disk
    os.makedirs(out_dir, exist_ok=True)
    np.save(feat_cache, X_all)
    np.save(lbl_cache, y_all)
    print(f"  Features cached to {out_dir}")

    return X_all, y_all


def make_dataloaders(cfg: dict):
    """
    Load BigQuery data + Gemini LLM features (or cache), split, return DataLoaders.

    BigQuery Bigframe + LLM pipeline:
        1. bpd.read_gbq() → bigframes DataFrame of SO questions
        2. GeminiTextGenerator.predict() → difficulty score per question title
        3. Combine: [answer_count, log_view_count, post_score, llm_difficulty]
        4. Binary label: accepted_answer_id IS NOT NULL
        5. Standardize features using train-set mean/std
        6. Return (train_loader, val_loader)

    cfg keys used:
        cfg['data']['bq_query']     - SQL query string for BigQuery
        cfg['data']['train_ratio']  - fraction for training
        cfg['data']['batch_size']   - DataLoader batch size
        cfg['output']['dir']        - directory for feature cache files

    Returns:
        train_loader, val_loader
    """
    X_all, y_all = _load_or_generate_features(cfg)

    n = len(X_all)
    n_train = int(n * cfg["data"]["train_ratio"])

    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    X_all, y_all = X_all[idx], y_all[idx]

    X_train, X_val = X_all[:n_train], X_all[n_train:]
    y_train, y_val = y_all[:n_train], y_all[n_train:]

    # Standardize using train statistics only
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
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Features: {X_train.shape[1]}")
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False),
    )


def build_model(cfg: dict) -> nn.Module:
    """
    Build an MLP for binary classification on LLM-augmented numeric features.

    Architecture:
        Linear(4, 64) → BatchNorm1d(64) → ReLU → Dropout(0.2)
        → Linear(64, 32) → BatchNorm1d(32) → ReLU
        → Linear(32, 1)                         ← raw logit (no sigmoid here)

    The BCEWithLogitsLoss in train() applies sigmoid internally.

    cfg keys used:
        cfg['model']['in_features']  - number of features (4)
        cfg['model']['hidden1']      - first hidden layer size (64)
        cfg['model']['hidden2']      - second hidden layer size (32)
        cfg['model']['dropout']      - dropout probability (0.2)

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
        dict with keys: 'loss', 'accuracy', 'f1', 'precision', 'recall'
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
    Run inference on raw tensor X (pre-normalized features).

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
                "SELECT title, answer_count, view_count, score, accepted_answer_id "
                "FROM `bigquery-public-data.stackoverflow.posts_questions` "
                "WHERE title IS NOT NULL "
                "  AND answer_count IS NOT NULL "
                "  AND view_count IS NOT NULL "
                "  AND score IS NOT NULL "
                "  AND EXTRACT(YEAR FROM creation_date) >= 2020 "
                "LIMIT 500"
            ),
            "train_ratio": 0.8,
            "batch_size": 32,
        },
        "model": {
            "in_features": 4,   # [answer_count, log_view_count, score, llm_difficulty]
            "hidden1": 64,
            "hidden2": 32,
            "dropout": 0.2,
        },
        "training": {
            "epochs": 50,
            "lr": 0.001,
        },
        "output": {
            "dir": "output/mlp_bq_llm_so_quality",
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
        assert val_metrics["accuracy"] > 0.62, (
            f"Accuracy too low: {val_metrics['accuracy']:.4f} (threshold: 0.62)"
        )
        print(f"PASS: val accuracy {val_metrics['accuracy']:.4f} > 0.62")

        print("\nAll assertions passed.")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
