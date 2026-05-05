"""
Homework 3: LLM Reasoning Data Classification
==============================================
Builds, evaluates, and compares 6 ML pipelines for text classification
on the Nvidia Nemotron Model Reasoning dataset.

Runs in TWO modes:
  - Easy (prompt):   classify task_type from the prompt text
  - Hard (answer):   classify task_type from the answer text

Target variable: task_type (6 balanced classes)
  bit_manipulation, gravity, unit_conversion, cipher_text, roman, symbol_transform

Pipelines per mode:
  P1: TF-IDF         + Multinomial Naive Bayes
  P2: TF-IDF         + Linear SVM (SGDClassifier)
  P3: TF-IDF         + k-Nearest Neighbors
  P4: Word2Vec       + Random Forest
  P5: Word2Vec       + MLP Neural Network (PyTorch / MPS)
  P6: DistilBERT emb + Logistic Regression

Outputs saved to hw3/:
  easy_fig1_metrics_bar.png
  easy_fig2_speed_tradeoff_scatter.png
  hard_fig1_metrics_bar.png
  hard_fig2_speed_tradeoff_scatter.png
  combined_comparison.png
  results_summary.csv

Run:
  python hw3/main.py
"""

import os
import sys
import time
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from transformers import DistilBertTokenizer, DistilBertModel

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using device: CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using device: CPU")


# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------
def preprocess_text_to_tokens(text):
    """Lower-case, keep alphanumeric, split into tokens of len > 1."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def tokens_to_avg_embedding(tokens_list, w2v_model, dim=100):
    """Average Word2Vec embedding over all tokens in each document."""
    vecs = []
    for tokens in tokens_list:
        valid = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
        vecs.append(np.mean(valid, axis=0) if valid else np.zeros(dim))
    return np.array(vecs)


def get_bert_embeddings(texts, tokenizer, model, device, batch_size=32):
    """Extract mean-pooled DistilBERT embeddings for a list of texts."""
    all_embeds = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch = list(texts[i * batch_size : (i + 1) * batch_size])
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            mask_exp = attention_mask.unsqueeze(-1).float()
            emb = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)

        all_embeds.append(emb.cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"  BERT batch {i+1}/{n_batches}")

    return np.vstack(all_embeds)


def evaluate_pipeline(name, y_true, y_pred, train_time, inf_time, label_names):
    """Compute metrics, print report, return result dict."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Pipeline: {name}")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.4f}  Precision: {prec:.4f}")
    print(f"  Recall:   {rec:.4f}  F1-Score:  {f1:.4f}")
    print(f"  Train: {train_time:.3f}s  Inference: {inf_time:.3f}s")
    print(classification_report(y_true, y_pred, target_names=label_names))

    return {
        "Pipeline": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Train Time (s)": round(train_time, 3),
        "Inference Time (s)": round(inf_time, 3),
    }


# ---------------------------------------------------------------------------
# 2. MLP definition
# ---------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. Run all pipelines for one mode
# ---------------------------------------------------------------------------
def run_all_pipelines(
    texts_train, texts_test,
    tokens_train, tokens_test,
    y_train_enc, y_test_enc,
    label_names, n_classes, device,
    mode_label,
):
    """
    Execute all 6 pipelines on the given train/test split.
    Returns list of result dicts (each with a 'Mode' key added).
    """
    results = []

    # ---- TF-IDF ----
    print(f"\n{'#'*70}")
    print(f"# TF-IDF Pipelines  —  {mode_label}")
    print(f"{'#'*70}")

    tfidf = TfidfVectorizer(
        max_features=4000, ngram_range=(1, 2),
        min_df=2, max_df=0.95, sublinear_tf=True,
    )
    X_train_tf = tfidf.fit_transform(texts_train)
    X_test_tf  = tfidf.transform(texts_test)
    print(f"TF-IDF shape: train {X_train_tf.shape}, test {X_test_tf.shape}")

    # P1: Naive Bayes
    print("\n--- P1: TF-IDF + Naive Bayes ---")
    t0 = time.perf_counter()
    nb = MultinomialNB(alpha=0.1).fit(X_train_tf, y_train_enc)
    tr = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = nb.predict(X_test_tf)
    inf = time.perf_counter() - t0
    r = evaluate_pipeline("P1: TF-IDF + Naive Bayes", y_test_enc, pred, tr, inf, label_names)
    r["Mode"] = mode_label
    results.append(r)

    # P2: SVM
    print("\n--- P2: TF-IDF + SVM ---")
    t0 = time.perf_counter()
    svm = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4,
                         max_iter=1000, tol=1e-3, random_state=42, n_jobs=-1
                         ).fit(X_train_tf, y_train_enc)
    tr = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = svm.predict(X_test_tf)
    inf = time.perf_counter() - t0
    r = evaluate_pipeline("P2: TF-IDF + SVM", y_test_enc, pred, tr, inf, label_names)
    r["Mode"] = mode_label
    results.append(r)

    # P3: k-NN
    print("\n--- P3: TF-IDF + k-NN ---")
    t0 = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1
                               ).fit(X_train_tf, y_train_enc)
    tr = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = knn.predict(X_test_tf)
    inf = time.perf_counter() - t0
    r = evaluate_pipeline("P3: TF-IDF + k-NN", y_test_enc, pred, tr, inf, label_names)
    r["Mode"] = mode_label
    results.append(r)

    # ---- Word2Vec ----
    print(f"\n{'#'*70}")
    print(f"# Word2Vec Pipelines  —  {mode_label}")
    print(f"{'#'*70}")

    w2v = Word2Vec(
        sentences=list(tokens_train), vector_size=100,
        window=5, min_count=2, workers=4, epochs=20, seed=42,
    )
    print(f"Word2Vec vocab size: {len(w2v.wv)}")

    X_train_w2v = tokens_to_avg_embedding(tokens_train, w2v, dim=100)
    X_test_w2v  = tokens_to_avg_embedding(tokens_test, w2v, dim=100)
    print(f"Word2Vec shape: train {X_train_w2v.shape}, test {X_test_w2v.shape}")

    # P4: Random Forest
    print("\n--- P4: Word2Vec + Random Forest ---")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1
                                ).fit(X_train_w2v, y_train_enc)
    tr = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = rf.predict(X_test_w2v)
    inf = time.perf_counter() - t0
    r = evaluate_pipeline("P4: Word2Vec + Random Forest", y_test_enc, pred, tr, inf, label_names)
    r["Mode"] = mode_label
    results.append(r)

    # P5: MLP
    print("\n--- P5: Word2Vec + MLP ---")
    X_tr_t = torch.tensor(X_train_w2v, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_enc, dtype=torch.long)
    X_te_t = torch.tensor(X_test_w2v, dtype=torch.float32)
    y_te_t = torch.tensor(y_test_enc, dtype=torch.long)

    train_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)
    test_dl  = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=256, shuffle=False)

    mlp = SimpleMLP(input_dim=100, hidden_dim=256, num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

    t0 = time.perf_counter()
    mlp.train()
    for epoch in range(30):
        epoch_loss = 0.0
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(mlp(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/30  loss: {epoch_loss/len(train_dl):.4f}")
    tr = time.perf_counter() - t0
    print(f"MLP training done in {tr:.3f}s")

    t0 = time.perf_counter()
    mlp.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in test_dl:
            preds.extend(mlp(bx.to(device)).argmax(dim=1).cpu().numpy())
    inf = time.perf_counter() - t0
    r = evaluate_pipeline("P5: Word2Vec + MLP", y_test_enc, np.array(preds), tr, inf, label_names)
    r["Mode"] = mode_label
    results.append(r)

    # ---- DistilBERT ----
    print(f"\n{'#'*70}")
    print(f"# DistilBERT Pipeline  —  {mode_label}")
    print(f"{'#'*70}")

    print("Loading DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    bert_model.eval()

    print("Extracting training embeddings...")
    t0 = time.perf_counter()
    X_train_bert = get_bert_embeddings(texts_train, tokenizer, bert_model, device, batch_size=32)
    bert_train_time = time.perf_counter() - t0
    print(f"  Done ({bert_train_time:.1f}s), shape: {X_train_bert.shape}")

    print("Extracting test embeddings...")
    t0_inf = time.perf_counter()
    X_test_bert = get_bert_embeddings(texts_test, tokenizer, bert_model, device, batch_size=32)
    bert_inf_time = time.perf_counter() - t0_inf
    print(f"  Done ({bert_inf_time:.1f}s), shape: {X_test_bert.shape}")

    # P6: Logistic Regression on DistilBERT
    print("Training Logistic Regression head...")
    t0 = time.perf_counter()
    lr = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
    lr.fit(X_train_bert, y_train_enc)
    lr_train_time = time.perf_counter() - t0

    total_train = bert_train_time + lr_train_time

    t0 = time.perf_counter()
    pred = lr.predict(X_test_bert)
    lr_inf_time = time.perf_counter() - t0
    total_inf = bert_inf_time + lr_inf_time

    r = evaluate_pipeline("P6: DistilBERT + LogReg", y_test_enc, pred,
                          total_train, total_inf, label_names)
    r["Mode"] = mode_label
    results.append(r)

    # Free GPU memory
    del bert_model, tokenizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# 4. Visualization helpers
# ---------------------------------------------------------------------------
def plot_metrics_bar(results_df, mode_label, prefix, output_dir):
    """Grouped bar chart of Accuracy / Precision / Recall / F1 per pipeline."""
    pipelines = results_df["Pipeline"].values
    x = np.arange(len(pipelines))
    w = 0.18

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (col, label, color) in enumerate([
        ("Accuracy",  "Accuracy",  "#2196F3"),
        ("Precision", "Precision", "#4CAF50"),
        ("Recall",    "Recall",    "#FF9800"),
        ("F1-Score",  "F1-Score",  "#F44336"),
    ]):
        vals = results_df[col].values
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Score")
    ax.set_title(f"Performance Metrics — {mode_label}", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{prefix}_fig1_metrics_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_speed_scatter(results_df, mode_label, prefix, output_dir):
    """Two-panel scatter: Accuracy vs Train Time, Accuracy vs Inference Time."""
    pipelines = results_df["Pipeline"].values
    acc = results_df["Accuracy"].values
    tr  = results_df["Train Time (s)"].values
    inf = results_df["Inference Time (s)"].values

    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]
    markers = ["o", "s", "^", "D", "P", "*"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, pl in enumerate(pipelines):
        ax1.scatter(tr[i], acc[i], c=colors[i], marker=markers[i], s=120,
                    edgecolors="black", linewidths=0.5, zorder=3)
        ax1.annotate(pl.replace(" ", "\n"), (tr[i], acc[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=7.5)
    ax1.set_xlabel("Training Time (s)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Accuracy vs Training Time — {mode_label}", fontweight="bold")
    ax1.grid(alpha=0.3)

    for i, pl in enumerate(pipelines):
        ax2.scatter(inf[i], acc[i], c=colors[i], marker=markers[i], s=120,
                    edgecolors="black", linewidths=0.5, zorder=3)
        ax2.annotate(pl.replace(" ", "\n"), (inf[i], acc[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=7.5)
    ax2.set_xlabel("Inference Time (s)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy vs Inference Time — {mode_label}", fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_fig2_speed_tradeoff_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_combined_comparison(all_df, output_dir):
    """Side-by-side bar chart comparing Easy vs Hard accuracy per pipeline."""
    pivot_acc = all_df.pivot(index="Pipeline", columns="Mode", values="Accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: Accuracy comparison
    pipelines = pivot_acc.index.tolist()
    x = np.arange(len(pipelines))
    w = 0.35
    modes = pivot_acc.columns.tolist()
    mode_colors = {"Easy (prompt)": "#4CAF50", "Hard (answer)": "#F44336"}

    for j, mode in enumerate(modes):
        vals = pivot_acc[mode].values
        axes[0].bar(x + (j - 0.5) * w, vals, w, label=mode, color=mode_colors.get(mode, f"C{j}"))
        for i, v in enumerate(vals):
            axes[0].text(x[i] + (j - 0.5) * w, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)

    axes[0].set_xlabel("Pipeline")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Easy vs Hard — Accuracy", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pipelines, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.08)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Right panel: Train time comparison (log scale)
    pivot_time = all_df.pivot(index="Pipeline", columns="Mode", values="Train Time (s)")
    for j, mode in enumerate(modes):
        vals = pivot_time[mode].values
        axes[1].bar(x + (j - 0.5) * w, vals, w, label=mode, color=mode_colors.get(mode, f"C{j}"))
        for i, v in enumerate(vals):
            axes[1].text(x[i] + (j - 0.5) * w, v + 0.5, f"{v:.2f}", ha="center", fontsize=7)

    axes[1].set_xlabel("Pipeline")
    axes[1].set_ylabel("Training Time (s)")
    axes[1].set_title("Easy vs Hard — Training Time", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pipelines, rotation=20, ha="right", fontsize=8)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "combined_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    data_paths = [
        os.path.join(SCRIPT_DIR, "train_with_task_type.csv"),
        os.path.join(SCRIPT_DIR, "..", "train_with_task_type.csv"),
        "/tmp/train_with_task_type.csv",
    ]
    csv_path = None
    for p in data_paths:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        sys.exit("ERROR: train_with_task_type.csv not found. "
                 "Download from https://github.com/lkk688/CoderGym/tree/main/Nemotron")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path}")

    # Fill NaN in answer column with empty string
    df["answer"] = df["answer"].fillna("")

    print("\nTask type distribution:")
    for label, cnt in df["task_type"].value_counts().items():
        print(f"  {label:30s} {cnt:5d}  ({cnt/len(df)*100:.1f}%)")

    # Label encoding (shared across modes)
    le = LabelEncoder()
    le.fit(df["task_type"])
    label_names = le.classes_.tolist()
    n_classes = len(label_names)
    print(f"\nLabel mapping: {dict(zip(label_names, range(n_classes)))}")

    # ------------------------------------------------------------------
    # Run both modes
    # ------------------------------------------------------------------
    all_results = []

    modes = [
        ("Easy (prompt)", "prompt"),
        ("Hard (answer)", "answer"),
    ]

    for mode_label, text_col in modes:
        print(f"\n{'=' * 70}")
        print(f"MODE: {mode_label}  (using column: '{text_col}')")
        print(f"{'=' * 70}")

        # Preprocess
        df["tokens"]     = df[text_col].apply(preprocess_text_to_tokens)
        df["clean_text"] = df["tokens"].apply(lambda t: " ".join(t))

        # Sample check
        print(f"\nSample ({mode_label}):")
        print(f"  Original: {df[text_col].iloc[0][:120]}...")
        print(f"  Tokens:   {df['tokens'].iloc[0][:12]}...")

        # Split
        X_text  = df["clean_text"].values
        tokens  = df["tokens"].values
        y_enc   = le.transform(df["task_type"].values)

        X_tr_txt, X_te_txt, tok_tr, tok_te, y_tr, y_te = train_test_split(
            X_text, tokens, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
        )
        print(f"Train: {len(X_tr_txt):,}  Test: {len(X_te_txt):,}")

        # Run pipelines
        results = run_all_pipelines(
            X_tr_txt, X_te_txt,
            list(tok_tr), list(tok_te),
            y_tr, y_te,
            label_names, n_classes, DEVICE,
            mode_label,
        )
        all_results.extend(results)

        # Per-mode figures
        mode_df = pd.DataFrame(results)
        prefix = mode_label.split()[0].lower()   # "easy" or "hard"
        plot_metrics_bar(mode_df, mode_label, prefix, SCRIPT_DIR)
        plot_speed_scatter(mode_df, mode_label, prefix, SCRIPT_DIR)

    # ------------------------------------------------------------------
    # Combined results + comparison
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("COMBINED RESULTS")
    print(f"{'=' * 70}")

    all_df = pd.DataFrame(all_results)
    # Reorder columns
    all_df = all_df[["Mode", "Pipeline", "Accuracy", "Precision", "Recall",
                      "F1-Score", "Train Time (s)", "Inference Time (s)"]]
    print(all_df.to_string(index=False))

    csv_out = os.path.join(SCRIPT_DIR, "results_summary.csv")
    all_df.to_csv(csv_out, index=False)
    print(f"\nResults CSV saved: {csv_out}")

    # Combined comparison figure
    plot_combined_comparison(all_df, SCRIPT_DIR)

    print(f"\n{'=' * 70}")
    print("ALL DONE — outputs saved to:", SCRIPT_DIR)
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
