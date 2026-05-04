"""
Homework 3: LLM Reasoning Data Classification
==============================================
This script builds, evaluates, and compares 6 machine learning pipelines for
text classification on the Nvidia Nemotron Model Reasoning dataset.

Target variable: task_type (6 classes: bit_manipulation, gravity,
                  unit_conversion, cipher_text, roman, symbol_transform)

Pipelines:
  P1: TF-IDF         + Multinomial Naive Bayes
  P2: TF-IDF         + Linear SVM (SGDClassifier with hinge loss)
  P3: TF-IDF         + k-Nearest Neighbors
  P4: Word2Vec (100d, trained on this data) + Random Forest
  P5: Word2Vec (100d, trained on this data) + MLP Neural Network (PyTorch/MPS)
  P6: DistilBERT embeddings                 + Logistic Regression

Metrics recorded per pipeline:
  - Accuracy, Precision, Recall, F1-Score
  - Training time (seconds)
  - Inference time (seconds)

Generated outputs (saved to hw3/):
  - fig1_metrics_bar.png  : Grouped bar chart of all 4 metrics
  - fig2_speed tradeoff_scatter.png : Accuracy vs training time + inference time
  - results_summary.csv   : Numeric results table

Run:
  python hw3/main.py
"""

import os
import sys
import time
import re
import string
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Gensim for Word2Vec
from gensim.models import Word2Vec

# PyTorch for MLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Transformers for DistilBERT
from transformers import DistilBertTokenizer, DistilBertModel

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Setup and paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = SCRIPT_DIR          # data lives alongside the script (or one level up)
OUTPUT_DIR = SCRIPT_DIR

# Device: prefer MPS (Apple Silicon GPU), fall back to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

# ---------------------------------------------------------------------------
# 1. Load and inspect data
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("SECTION 1: Loading Data")
print("="*70)

# Try to find train_with_task_type.csv
data_paths = [
    os.path.join(DATA_DIR, "train_with_task_type.csv"),
    os.path.join(DATA_DIR, "..", "train_with_task_type.csv"),
    "/tmp/train_with_task_type.csv",
]
csv_path = None
for p in data_paths:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    print("ERROR: train_with_task_type.csv not found.")
    print("Please download from: https://github.com/lkk688/CoderGym/tree/main/Nemotron")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} rows from {csv_path}")
print(f"Columns: {list(df.columns)}")

# Target variable
print(f"\nTask type distribution:")
for label, count in df["task_type"].value_counts().items():
    print(f"  {label:30s} {count:5d}  ({count/len(df)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("SECTION 2: Text Preprocessing")
print("="*70)

def preprocess_text(text: str) -> list[str]:
    """
    Lightweight preprocessing:
      - lower-case
      - keep alphanumeric + spaces
      - split on whitespace
    Returns list of tokens.
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    # Keep letters, digits, spaces; collapse everything else
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    return tokens

# Apply to prompts
df["tokens"] = df["prompt"].apply(preprocess_text)
df["clean_text"] = df["tokens"].apply(lambda t: " ".join(t))

print(f"Sample cleaned prompt (first row):")
print(f"  Original: {df['prompt'].iloc[0][:120]}...")
print(f"  Tokens:   {df['tokens'].iloc[0][:15]}...")

# ---------------------------------------------------------------------------
# 3. Train / test split (stratified)
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("SECTION 3: Train/Test Split")
print("="*70)

X_text = df["clean_text"].values
y      = df["task_type"].values
tokens  = df["tokens"].values

X_train_text, X_test_text, y_train, y_test, tokens_train, tokens_test = \
    train_test_split(
        X_text, y, tokens,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

print(f"Training set: {len(X_train_text):,} samples")
print(f"Test set:     {len(X_test_text):,} samples")

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)                                    # fit on all known labels
y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

label_names = le.classes_.tolist()
n_classes   = len(label_names)
print(f"Label mapping: {dict(zip(label_names, range(n_classes)))}")

# ---------------------------------------------------------------------------
# 4. Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_pipeline(name: str,
                     y_true, y_pred,
                     train_time: float,
                     inference_time: float) -> dict:
    """
    Compute standard metrics and bundle results into a dict.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    result = {
        "Pipeline":       name,
        "Accuracy":       round(acc, 4),
        "Precision":      round(prec, 4),
        "Recall":         round(rec, 4),
        "F1-Score":       round(f1, 4),
        "Train Time (s)": round(train_time, 3),
        "Inference Time (s)": round(inference_time, 3),
    }
    print(f"\n{'='*60}")
    print(f"  Pipeline: {name}")
    print(f"{'='*60}")
    print(f"  Accuracy:    {acc:.4f}   Precision: {prec:.4f}")
    print(f"  Recall:      {rec:.4f}   F1-Score:  {f1:.4f}")
    print(f"  Train time:  {train_time:.3f}s   Inference: {inference_time:.3f}s")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))
    return result

# Storage for all results
all_results = []

# ---------------------------------------------------------------------------
# 5. PIPELINES 1-3: TF-IDF + (NB | SVM | kNN)
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("SECTION 5: Pipelines 1-3  (TF-IDF base)")
print("="*70)

# Build TF-IDF vectorizer once — share across P1-P3
tfidf = TfidfVectorizer(
    max_features=4000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf  = tfidf.transform(X_test_text)
print(f"TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
print(f"TF-IDF train shape:    {X_train_tfidf.shape}")
print(f"TF-IDF test shape:     {X_test_tfidf.shape}")

# ---------- P1: Multinomial Naive Bayes ----------
print("\n--- P1: TF-IDF + Multinomial Naive Bayes ---")
t0 = time.perf_counter()
nb_clf = MultinomialNB(alpha=0.1)
nb_clf.fit(X_train_tfidf, y_train_enc)
train_time = time.perf_counter() - t0

t0 = time.perf_counter()
y_pred = nb_clf.predict(X_test_tfidf)
inf_time = time.perf_counter() - t0

r = evaluate_pipeline("P1: TF-IDF + Naive Bayes",
                      y_test_enc, y_pred, train_time, inf_time)
all_results.append(r)

# ---------- P2: Linear SVM (SGDClassifier) ----------
print("\n--- P2: TF-IDF + Linear SVM (SGD) ---")
t0 = time.perf_counter()
svm_clf = SGDClassifier(
    loss="hinge",
    penalty="l2",
    alpha=1e-4,
    max_iter=1000,
    tol=1e-3,
    random_state=42,
    n_jobs=-1,
)
svm_clf.fit(X_train_tfidf, y_train_enc)
train_time = time.perf_counter() - t0

t0 = time.perf_counter()
y_pred = svm_clf.predict(X_test_tfidf)
inf_time = time.perf_counter() - t0

r = evaluate_pipeline("P2: TF-IDF + SVM",
                      y_test_enc, y_pred, train_time, inf_time)
all_results.append(r)

# ---------- P3: k-Nearest Neighbors ----------
print("\n--- P3: TF-IDF + k-Nearest Neighbors ---")
t0 = time.perf_counter()
knn_clf = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
knn_clf.fit(X_train_tfidf, y_train_enc)
train_time = time.perf_counter() - t0

t0 = time.perf_counter()
y_pred = knn_clf.predict(X_test_tfidf)
inf_time = time.perf_counter() - t0

r = evaluate_pipeline("P3: TF-IDF + k-NN",
                      y_test_enc, y_pred, train_time, inf_time)
all_results.append(r)

# ---------------------------------------------------------------------------
# 6. PIPELINES 4-5: Word2Vec embeddings
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("SECTION 6: Pipelines 4-5  (Word2Vec base)")
print("="*70)

# Train Word2Vec on training tokens only (no data leakage)
w2v_model = Word2Vec(
    sentences=tokens_train.tolist(),
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=20,
    seed=42,
)
print(f"Word2Vec vocabulary size: {len(w2v_model.wv)}")

def tokens_to_avg_embedding(tokens_list, w2v, dim=100):
    """
    Average Word2Vec embedding over all tokens in a document.
    OOV tokens are silently skipped.
    """
    vectors = []
    for tokens in tokens_list:
        valid = [w2v.wv[t] for t in tokens if t in w2v.wv]
        if valid:
            vectors.append(np.mean(valid, axis=0))
        else:
            vectors.append(np.zeros(dim))
    return np.array(vectors)

print("Building Word2Vec embeddings for training set...")
X_train_w2v = tokens_to_avg_embedding(tokens_train, w2v_model)
print("Building Word2Vec embeddings for test set...")
X_test_w2v  = tokens_to_avg_embedding(tokens_test,  w2v_model)
print(f"Word2Vec train shape: {X_train_w2v.shape}")
print(f"Word2Vec test shape:  {X_test_w2v.shape}")

# ---------- P4: Random Forest ----------
print("\n--- P4: Word2Vec + Random Forest ---")
t0 = time.perf_counter()
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
)
rf_clf.fit(X_train_w2v, y_train_enc)
train_time = time.perf_counter() - t0

t0 = time.perf_counter()
y_pred = rf_clf.predict(X_test_w2v)
inf_time = time.perf_counter() - t0

r = evaluate_pipeline("P4: Word2Vec + Random Forest",
                      y_test_enc, y_pred, train_time, inf_time)
all_results.append(r)

# ---------- P5: MLP (PyTorch) ----------
print("\n--- P5: Word2Vec + MLP (PyTorch) ---")

# Simple 3-layer MLP
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

# Prepare PyTorch datasets
X_train_t_w2v = torch.tensor(X_train_w2v, dtype=torch.float32)
y_train_t     = torch.tensor(y_train_enc, dtype=torch.long)
X_test_t_w2v  = torch.tensor(X_test_w2v,  dtype=torch.float32)
y_test_t      = torch.tensor(y_test_enc,  dtype=torch.long)

train_ds = TensorDataset(X_train_t_w2v, y_train_t)
test_ds  = TensorDataset(X_test_t_w2v,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

mlp_model = SimpleMLP(input_dim=100, hidden_dim=256, num_classes=n_classes).to(device)
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(mlp_model.parameters(), lr=1e-3)

# Training
epochs = 30
t0 = time.perf_counter()
mlp_model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = mlp_model(batch_x)
        loss    = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{epochs}  loss: {epoch_loss/len(train_loader):.4f}")

train_time = time.perf_counter() - t0
print(f"MLP training completed in {train_time:.3f}s ({epochs} epochs)")

# Inference
t0 = time.perf_counter()
mlp_model.eval()
all_preds = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        outputs = mlp_model(batch_x)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
inf_time = time.perf_counter() - t0

y_pred_np = np.array(all_preds)
r = evaluate_pipeline("P5: Word2Vec + MLP",
                      y_test_enc, y_pred_np, train_time, inf_time)
all_results.append(r)

# ---------------------------------------------------------------------------
# 7. PIPELINE 6: DistilBERT embeddings + Logistic Regression
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("SECTION 7: Pipeline 6  (DistilBERT + Logistic Regression)")
print("="*70)

print("Loading DistilBERT tokenizer and model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
bert_model.eval()
print(f"DistilBERT loaded on {device}")

def get_bert_embeddings(texts, batch_size=32):
    """
    Extract [CLS] token embedding (768-dim) from DistilBERT for each text.
    Uses mean pooling if texts are longer than 512 tokens.
    """
    all_embeds = []
    n_batches  = (len(texts) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch_texts = texts[i*batch_size : (i+1)*batch_size]
        encoded = tokenizer(
            batch_texts.tolist() if hasattr(batch_texts, "tolist") else list(batch_texts),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        input_ids  = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pooling over token dimension
            hidden   = outputs.last_hidden_state                    # (batch, seq_len, 768)
            mask_exp = attention_mask.unsqueeze(-1).float()          # (batch, seq_len, 1)
            sum_emb  = (hidden * mask_exp).sum(dim=1)
            count    = mask_exp.sum(dim=1).clamp(min=1e-9)
            emb      = sum_emb / count                              # (batch, 768)
        all_embeds.append(emb.cpu().numpy())

        if (i + 1) % 50 == 0:
            print(f"  Processed batch {i+1}/{n_batches}")

    return np.vstack(all_embeds)

print("\nExtracting DistilBERT embeddings for training set...")
t0 = time.perf_counter()
X_train_bert = get_bert_embeddings(X_train_text, batch_size=32)
train_time = time.perf_counter() - t0
print(f"Training embeddings done in {train_time:.3f}s  shape: {X_train_bert.shape}")

print("Extracting DistilBERT embeddings for test set...")
t0_inf = time.perf_counter()
X_test_bert = get_bert_embeddings(X_test_text, batch_size=32)
inf_time = time.perf_counter() - t0_inf
print(f"Test embeddings done in {inf_time:.3f}s  shape: {X_test_bert.shape}")

# Combine BERT embedding extraction + classifier training time
total_train_time = train_time
total_inf_time   = inf_time

print("\nTraining Logistic Regression head on DistilBERT embeddings...")
t0 = time.perf_counter()
lr_clf = LogisticRegression(max_iter=500, solver="lbfgs", multi_class="multinomial", random_state=42)
lr_clf.fit(X_train_bert, y_train_enc)
lr_train_time = time.perf_counter() - t0
total_train_time += lr_train_time
print(f"Logistic Regression trained in {lr_train_time:.3f}s")

y_pred = lr_clf.predict(X_test_bert)
total_inf_time += time.perf_counter() - t0_inf   # already measured BERT inference above

r = evaluate_pipeline("P6: DistilBERT + Logistic Regression",
                      y_test_enc, y_pred, total_train_time, total_inf_time)
all_results.append(r)

# Clean up GPU memory
del bert_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ---------------------------------------------------------------------------
# 8. Compile results and save CSV
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("SECTION 8: Results Summary")
print("="*70)

results_df = pd.DataFrame(all_results)
print("\n", results_df.to_string(index=False))

csv_out = os.path.join(OUTPUT_DIR, "results_summary.csv")
results_df.to_csv(csv_out, index=False)
print(f"\nResults saved to: {csv_out}")

# ---------------------------------------------------------------------------
# 9. Visualizations
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("SECTION 9: Generating Visualizations")
print("="*70)

pipelines   = results_df["Pipeline"].values
acc_vals    = results_df["Accuracy"].values
prec_vals   = results_df["Precision"].values
rec_vals    = results_df["Recall"].values
f1_vals     = results_df["F1-Score"].values
train_times = results_df["Train Time (s)"].values
inf_times   = results_df["Inference Time (s)"].values

x = np.arange(len(pipelines))
width = 0.18

# ----- Figure 1: Grouped bar chart of metrics -----
fig1, ax1 = plt.subplots(figsize=(14, 6))
bars1 = ax1.bar(x - 1.5*width, acc_vals,  width, label="Accuracy",  color="#2196F3")
bars2 = ax1.bar(x - 0.5*width, prec_vals,  width, label="Precision", color="#4CAF50")
bars3 = ax1.bar(x + 0.5*width, rec_vals,   width, label="Recall",    color="#FF9800")
bars4 = ax1.bar(x + 1.5*width, f1_vals,     width, label="F1-Score",  color="#F44336")

ax1.set_xlabel("Pipeline", fontsize=12)
ax1.set_ylabel("Score", fontsize=12)
ax1.set_title("Figure 1: Performance Metrics Comparison Across Pipelines", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(pipelines, rotation=15, ha="right", fontsize=9)
ax1.set_ylim(0, 1.08)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(axis="y", alpha=0.3)

# Attach value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "fig1_metrics_bar.png")
fig1.savefig(fig1_path, dpi=150)
print(f"Figure 1 saved: {fig1_path}")
plt.close(fig1)

# ----- Figure 2: Accuracy vs Speed trade-off scatter -----
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))

colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]
markers = ["o", "s", "^", "D", "P", "*"]

# Left: Accuracy vs Training Time
for i, (pl, acc, tt) in enumerate(zip(pipelines, acc_vals, train_times)):
    ax2a.scatter(tt, acc, c=colors[i], marker=markers[i], s=120, zorder=3,
                 edgecolors="black", linewidths=0.5)
    ax2a.annotate(pl.replace(" ", "\n"), (tt, acc),
                  textcoords="offset points", xytext=(8, 5),
                  fontsize=7.5, ha="left")
ax2a.set_xlabel("Training Time (s)", fontsize=12)
ax2a.set_ylabel("Accuracy", fontsize=12)
ax2a.set_title("Figure 2a: Accuracy vs Training Time", fontsize=13, fontweight="bold")
ax2a.grid(alpha=0.3)

# Right: Accuracy vs Inference Time
for i, (pl, acc, it) in enumerate(zip(pipelines, acc_vals, inf_times)):
    ax2b.scatter(it, acc, c=colors[i], marker=markers[i], s=120, zorder=3,
                 edgecolors="black", linewidths=0.5)
    ax2b.annotate(pl.replace(" ", "\n"), (it, acc),
                  textcoords="offset points", xytext=(8, 5),
                  fontsize=7.5, ha="left")
ax2b.set_xlabel("Inference Time (s)", fontsize=12)
ax2b.set_ylabel("Accuracy", fontsize=12)
ax2b.set_title("Figure 2b: Accuracy vs Inference Time", fontsize=13, fontweight="bold")
ax2b.grid(alpha=0.3)

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "fig2_speed_tradeoff_scatter.png")
fig2.savefig(fig2_path, dpi=150)
print(f"Figure 2 saved: {fig2_path}")
plt.close(fig2)

# ----- Bonus: Heatmap-style metric table as image -----
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.axis("off")
col_labels = ["Pipeline", "Accuracy", "Precision", "Recall", "F1-Score", "Train(s)", "Inf.(s)"]
table_data = results_df[["Pipeline","Accuracy","Precision","Recall","F1-Score",
                           "Train Time (s)","Inference Time (s)"]].values.tolist()
table = ax3.table(cellText=table_data, colLabels=col_labels,
                  loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.6)
# Highlight best values
for j, col in enumerate(["Accuracy","Precision","Recall","F1-Score"]):
    col_idx = results_df.columns.get_loc(col) - 1
    max_val = results_df[col].max()
    for i, row in enumerate(results_df[col].values):
        if row == max_val:
            table[(i+1, col_idx)].set_facecolor("#C8E6C9")
fig3_path = os.path.join(OUTPUT_DIR, "fig3_results_table.png")
fig3.savefig(fig3_path, dpi=150)
print(f"Figure 3 saved: {fig3_path}")
plt.close(fig3)

print("\n" + "="*70)
print("ALL DONE — outputs saved to:", OUTPUT_DIR)
print("="*70)
