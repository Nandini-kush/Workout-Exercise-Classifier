# ============================================================
#  evaluate.py  –  Model evaluation helpers
# ============================================================

import os
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import tensorflow as tf
from config import REPORT_DIR


def evaluate_model(model, test_ds, class_names, model_name):
    """
    Run full evaluation on the test dataset.
    Prints accuracy, classification report, per-class accuracy.
    Returns a dict with y_true, y_pred, test_accuracy.
    """
    print(f"\n[Eval] Evaluating {model_name} …")

    y_true, y_pred = [], []

    for batch_images, batch_labels in test_ds:
        preds = model.predict(batch_images, verbose=0)
        y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    per_class = {}
    for i, cls in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            per_class[cls] = float(np.mean(y_pred[mask] == i))
        else:
            per_class[cls] = 0.0

    # Print
    print(f"\n{'─'*50}")
    print(f"  {model_name.upper()} – Test Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'─'*50}")
    print("\nClassification Report:")
    print(report)

    print("\nPer-class Accuracy:")
    for cls, a in sorted(per_class.items(), key=lambda x: x[1]):
        bar = "█" * int(a * 20)
        print(f"  {cls:<25} {a:.4f}  {bar}")

    # Save report
    report_path = os.path.join(REPORT_DIR, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nPer-class Accuracy:\n")
        for cls, a in per_class.items():
            f.write(f"  {cls}: {a:.4f}\n")

    print(f"\n[Eval] Report saved: {report_path}")

    return {
        "test_accuracy": acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "cm": cm,
        "per_class": per_class,
        "report": report
    }
