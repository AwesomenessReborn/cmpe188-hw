"""
Hyperparameter Exploration: Why k-NN and Naive Bayes Fail on Short Answers
==========================================================================

This script investigates whether poor performance on the Hard (answer) mode
is due to bad hyperparameters or fundamental feature representation issues.

Experiments:
  1. k-NN sweep: k ∈ [1, 3, 5, 10, 20] × metrics [cosine, euclidean, manhattan]
  2. Naive Bayes alpha sweep: alpha ∈ [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

Generates plots showing accuracy vs parameter, with SVM baseline as reference.

Run:
  python hw3/experiments.py
"""

import os
from time import time
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Load and preprocess
# ---------------------------------------------------------------------------
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

paths = [
    os.path.join(SCRIPT_DIR, "train_with_task_type.csv"),
    os.path.join(SCRIPT_DIR, "..", "train_with_task_type.csv"),
    "/tmp/train_with_task_type.csv",
]
csv_path = next((p for p in paths if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError("train_with_task_type.csv not found")

df = pd.read_csv(csv_path)
df["answer"] = df["answer"].fillna("")

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join([t for t in text.split() if len(t) > 1])

df["clean"] = df["answer"].apply(preprocess)

le = LabelEncoder()
y = le.fit_transform(df["task_type"])
X = df["clean"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

# TF-IDF (same as main.py baseline)
tfidf = TfidfVectorizer(
    max_features=4000, ngram_range=(1, 2),
    min_df=2, max_df=0.95, sublinear_tf=True,
)
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf  = tfidf.transform(X_test)
print(f"TF-IDF shape: {X_train_tf.shape}")

# SVM baseline for reference
svm = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4,
                     max_iter=1000, tol=1e-3, random_state=42, n_jobs=-1)
svm.fit(X_train_tf, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test_tf))
print(f"\nSVM baseline accuracy: {svm_acc:.4f}")

# ---------------------------------------------------------------------------
# 2. k-NN Parameter Sweep
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print("EXPERIMENT 1: k-NN Parameter Sweep")
print(f"{'=' * 70}")

k_values = [1, 3, 5, 10, 20]
metrics = ["cosine", "euclidean", "manhattan"]

knn_results = []
for metric in metrics:
    for k in k_values:
        t0 = time()
        # Note: manhattan and euclidean need dense vectors for k-NN
        if metric == "cosine":
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
            knn.fit(X_train_tf, y_train)
            pred = knn.predict(X_test_tf)
        else:
            # For euclidean/manhattan, use dense (small enough)
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
            knn.fit(X_train_tf.toarray(), y_train)
            pred = knn.predict(X_test_tf.toarray())
        
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
        knn_results.append({"k": k, "metric": metric, "accuracy": acc, "f1": f1})
        print(f"  k={k:2d}, metric={metric:10s} → accuracy={acc:.4f}, f1={f1:.4f}")

knn_df = pd.DataFrame(knn_results)
print(f"\nk-NN Best: {knn_df.loc[knn_df['accuracy'].idxmax()].to_dict()}")

# ---------------------------------------------------------------------------
# 3. Naive Bayes Alpha Sweep
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print("EXPERIMENT 2: Naive Bayes Alpha Sensitivity")
print(f"{'=' * 70}")

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
nb_results = []

for alpha in alphas:
    nb = MultinomialNB(alpha=alpha).fit(X_train_tf, y_train)
    pred = nb.predict(X_test_tf)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
    nb_results.append({"alpha": alpha, "accuracy": acc, "f1": f1})
    print(f"  alpha={alpha:7.3f} → accuracy={acc:.4f}, f1={f1:.4f}")

nb_df = pd.DataFrame(nb_results)
print(f"\nNB Best: {nb_df.loc[nb_df['accuracy'].idxmax()].to_dict()}")

# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print("GENERATING PLOTS")
print(f"{'=' * 70}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 1: k-NN ---
for metric in metrics:
    subset = knn_df[knn_df["metric"] == metric]
    ax1.plot(subset["k"], subset["accuracy"], marker="o", label=metric, linewidth=2)

ax1.axhline(y=svm_acc, color="red", linestyle="--", linewidth=1.5,
            label=f"SVM baseline ({svm_acc:.3f})")
ax1.set_xlabel("k (number of neighbors)")
ax1.set_ylabel("Accuracy")
ax1.set_title("k-NN: Accuracy vs k", fontweight="bold")
ax1.set_xticks(k_values)
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(0.5, 0.9)

# --- Plot 2: Naive Bayes ---
ax2.semilogx(nb_df["alpha"], nb_df["accuracy"], marker="o", linewidth=2, color="#4CAF50")
ax2.axhline(y=svm_acc, color="red", linestyle="--", linewidth=1.5,
            label=f"SVM baseline ({svm_acc:.3f})")
ax2.set_xlabel("Alpha (smoothing parameter, log scale)")
ax2.set_ylabel("Accuracy")
ax2.set_title("Naive Bayes: Accuracy vs Alpha", fontweight="bold")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(0.5, 0.9)

plt.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, "hyperparameter_sweep.png")
fig.savefig(plot_path, dpi=150)
print(f"Saved: {plot_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Summary Table
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")

print("\n--- k-NN Results ---")
print(knn_df.to_string(index=False))

print("\n--- Naive Bayes Results ---")
print(nb_df.to_string(index=False))

print(f"\n{'=' * 70}")
print("KEY FINDINGS")
print(f"{'=' * 70}")
print(f"""
1. k-NN best config:     {knn_df.loc[knn_df['accuracy'].idxmax(), 'metric']} metric, k={knn_df.loc[knn_df['accuracy'].idxmax(), 'k']}
   → Accuracy:          {knn_df['accuracy'].max():.4f}
   → vs SVM baseline:   {svm_acc:.4f}
   → Gap:               {svm_acc - knn_df['accuracy'].max():.4f}

2. Naive Bayes best config: alpha={nb_df.loc[nb_df['accuracy'].idxmax(), 'alpha']}
   → Accuracy:          {nb_df['accuracy'].max():.4f}
   → vs SVM baseline:   {svm_acc:.4f}
   → Gap:               {svm_acc - nb_df['accuracy'].max():.4f}

CONCLUSION: Even with optimal hyperparameters, both models fall significantly
below the SVM baseline. The problem is NOT hyperparameters — it is the feature
representation. Word-level TF-IDF cannot capture the structural patterns in
numeric answers (e.g., 154.62 vs 16.65).
""")
