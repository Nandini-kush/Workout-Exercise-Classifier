# ============================================================
#  visualize.py  –  All matplotlib/seaborn visualisations
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")       # non-interactive backend (safe for all OS)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import PLOT_DIR, IMG_SIZE

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "#0f0f1a",
    "axes.facecolor"   : "#1a1a2e",
    "axes.edgecolor"   : "#444466",
    "axes.labelcolor"  : "#ccccee",
    "xtick.color"      : "#ccccee",
    "ytick.color"      : "#ccccee",
    "text.color"       : "#eeeeff",
    "grid.color"       : "#333355",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,
    "font.family"      : "monospace",
})

ACCENT1 = "#4fc3f7"   # cyan-blue
ACCENT2 = "#f06292"   # pink
ACCENT3 = "#81c784"   # green
ACCENT4 = "#ffb74d"   # amber


# ── 1. Training history ───────────────────────────────────────

def plot_training_history(history: dict, model_name: str):
    """
    Two-panel plot: Accuracy & Loss over epochs.
    """
    epochs = range(1, len(history["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.grid(True)

    # Accuracy
    ax1.plot(epochs, history["accuracy"],     color=ACCENT1, lw=2, label="Train Acc")
    ax1.plot(epochs, history["val_accuracy"], color=ACCENT2, lw=2, label="Val Acc", linestyle="--")
    ax1.set_title(f"{model_name} – Accuracy", color="#eeeeff", fontsize=13)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(facecolor="#1a1a2e", labelcolor="#eeeeff")
    ax1.set_ylim(0, 1.05)

    # Loss
    ax2.plot(epochs, history["loss"],     color=ACCENT3, lw=2, label="Train Loss")
    ax2.plot(epochs, history["val_loss"], color=ACCENT4, lw=2, label="Val Loss", linestyle="--")
    ax2.set_title(f"{model_name} – Loss", color="#eeeeff", fontsize=13)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(facecolor="#1a1a2e", labelcolor="#eeeeff")

    fig.suptitle(f"Training History – {model_name}", color="#eeeeff", fontsize=15, y=1.01)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"{model_name}_history.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Saved: {out_path}")


# ── 2. Model comparison bar chart ─────────────────────────────

def plot_model_comparison(acc_dict: dict):
    """
    acc_dict: {model_name: test_accuracy}
    """
    names  = list(acc_dict.keys())
    accs   = [acc_dict[n] * 100 for n in names]
    colors = [ACCENT1, ACCENT2, ACCENT3, ACCENT4][:len(names)]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    bars = ax.bar(names, accs, color=colors, width=0.45, edgecolor="#0f0f1a", linewidth=1.5)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.2f}%",
            ha="center", va="bottom", color="#eeeeff", fontsize=12, fontweight="bold"
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Test Accuracy (%)", color="#ccccee")
    ax.set_title("Model Comparison – Test Accuracy", color="#eeeeff", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", labelsize=11)

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, "model_comparison.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Saved: {out_path}")


# ── 3. Confusion matrix ───────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """
    Normalised confusion matrix heatmap.
    """
    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig_size = max(10, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    fig.patch.set_facecolor("#0f0f1a")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.3,
        linecolor="#0f0f1a",
        cbar_kws={"shrink": 0.7}
    )

    ax.set_facecolor("#1a1a2e")
    ax.set_title(f"Confusion Matrix – {model_name}", color="#eeeeff", fontsize=13, pad=15)
    ax.set_xlabel("Predicted", color="#ccccee")
    ax.set_ylabel("True", color="#ccccee")
    plt.xticks(rotation=45, ha="right", color="#ccccee", fontsize=8)
    plt.yticks(rotation=0, color="#ccccee", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"{model_name}_confusion_matrix.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Saved: {out_path}")


# ── 4. Sample predictions grid ────────────────────────────────

def plot_sample_predictions(model, test_ds, class_names, model_name, n=16):
    """
    Grid of n sample images with true vs predicted labels.
    Green border = correct, Red border = wrong.
    """
    images, y_true_batch, y_pred_batch = [], [], []

    for imgs, lbls in test_ds:
        preds = model.predict(imgs, verbose=0)
        images.extend(imgs.numpy())
        y_true_batch.extend(np.argmax(lbls.numpy(), axis=1))
        y_pred_batch.extend(np.argmax(preds, axis=1))
        if len(images) >= n:
            break

    images      = images[:n]
    y_true_batch = y_true_batch[:n]
    y_pred_batch = y_pred_batch[:n]

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.patch.set_facecolor("#0f0f1a")

    for idx, ax in enumerate(axes.flat):
        if idx >= len(images):
            ax.axis("off")
            continue

        ax.imshow(images[idx])
        true_cls = class_names[y_true_batch[idx]]
        pred_cls = class_names[y_pred_batch[idx]]
        correct  = true_cls == pred_cls

        color = ACCENT3 if correct else ACCENT2
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

        ax.set_title(
            f"T: {true_cls}\nP: {pred_cls}",
            fontsize=8, color=color, pad=3
        )
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Sample Predictions – {model_name}", color="#eeeeff", fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"{model_name}_sample_predictions.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Saved: {out_path}")
