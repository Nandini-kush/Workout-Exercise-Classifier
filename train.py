# ============================================================
#  train.py  –  Train MobileNetV2 & EfficientNetB0
#  Run this FIRST: python train.py
# ============================================================

import os
import json
import time
import tensorflow as tf

from config import (
    MODEL_DIR,
    PLOT_DIR,
    REPORT_DIR,
    EPOCHS_FROZEN,
    EPOCHS_FINE,
    FINE_LR,
    MOBILENET_NAME,
    EFFICIENTNET_NAME,
    BEST_MODEL_PATH,
    ES_PATIENCE,
    LR_PATIENCE,
)
from data_loader import get_datasets
from models import (
    build_mobilenetv2,
    unfreeze_mobilenetv2,
    build_efficientnetb0,
    unfreeze_efficientnetb0,
    get_callbacks,
)
from visualize import (
    plot_training_history,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_sample_predictions,
)
from evaluate import evaluate_model


print("=" * 60)
print("  Workout Exercise Classifier – Training Script")
print("=" * 60)


# ── Load datasets ────────────────────────────────────────────
train_ds, val_ds, test_ds, class_names, num_classes, steps = get_datasets()
print(f"\n[Info] Classes ({num_classes}): {class_names}\n")

# Save class names for GUI / prediction use
with open(os.path.join(REPORT_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=2)


# ── Helper: combine two Keras history objects safely ────────
def combine_histories(hist1, hist2):
    combined = {}
    all_keys = set(hist1.history.keys()) | set(hist2.history.keys())

    for key in all_keys:
        part1 = hist1.history.get(key, [])
        part2 = hist2.history.get(key, [])
        combined[key] = part1 + part2

    return combined


# ── Helper: save history to json ─────────────────────────────
def save_history(history_dict, model_name):
    history_path = os.path.join(REPORT_DIR, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[Info] History saved: {history_path}")


# ── Helper: train one model in two phases ───────────────────
def train_model(name, build_fn, unfreeze_fn, unfreeze_kwargs):
    print(f"\n{'─' * 55}")
    print(f"  Training: {name.upper()}")
    print(f"{'─' * 55}")

    model, base = build_fn(num_classes)
    callbacks, ckpt_path = get_callbacks(name, MODEL_DIR, ES_PATIENCE, LR_PATIENCE)

    # Phase 1: frozen base
    print(f"\n[Phase 1] Frozen base – {EPOCHS_FROZEN} epochs")
    t0 = time.time()

    hist1 = model.fit(
        train_ds,
        epochs=EPOCHS_FROZEN,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"[Phase 1] Done in {(time.time() - t0) / 60:.1f} min")

    # Phase 2: fine-tuning
    if EPOCHS_FINE > 0:
        print(f"\n[Phase 2] Fine-tuning – {EPOCHS_FINE} more epochs")

        model = unfreeze_fn(model, base, fine_lr=FINE_LR, **unfreeze_kwargs)
        callbacks2, _ = get_callbacks(name, MODEL_DIR, ES_PATIENCE, LR_PATIENCE)

        t1 = time.time()

        hist2 = model.fit(
            train_ds,
            initial_epoch=EPOCHS_FROZEN,
            epochs=EPOCHS_FROZEN + EPOCHS_FINE,
            validation_data=val_ds,
            callbacks=callbacks2,
            verbose=1,
        )

        print(f"[Phase 2] Done in {(time.time() - t1) / 60:.1f} min")
        combined_history = combine_histories(hist1, hist2)

    else:
        print("\n[Phase 2] Skipped because EPOCHS_FINE = 0")
        combined_history = hist1.history

    # Load best checkpoint before evaluation
    if os.path.exists(ckpt_path):
        model = tf.keras.models.load_model(ckpt_path)
        print(f"[Info] Loaded best checkpoint: {ckpt_path}")

    save_history(combined_history, name)

    return model, combined_history, ckpt_path


# ── Train MobileNetV2 ────────────────────────────────────────
mob_model, mob_hist, mob_ckpt = train_model(
    MOBILENET_NAME,
    build_mobilenetv2,
    unfreeze_mobilenetv2,
    {"unfreeze_from": 100},
)

# ── Train EfficientNetB0 ─────────────────────────────────────
eff_model, eff_hist, eff_ckpt = train_model(
    EFFICIENTNET_NAME,
    build_efficientnetb0,
    unfreeze_efficientnetb0,
    {"unfreeze_from": 200},
)

# ── Evaluate both models ─────────────────────────────────────
print("\n" + "=" * 55)
print("  EVALUATION")
print("=" * 55)

mob_results = evaluate_model(mob_model, test_ds, class_names, MOBILENET_NAME)
eff_results = evaluate_model(eff_model, test_ds, class_names, EFFICIENTNET_NAME)

mob_acc = mob_results["test_accuracy"]
eff_acc = eff_results["test_accuracy"]

print(f"\n  MobileNetV2     test accuracy : {mob_acc:.4f}")
print(f"  EfficientNetB0 test accuracy : {eff_acc:.4f}")

# ── Select best model ────────────────────────────────────────
if eff_acc >= mob_acc:
    best_model = eff_model
    best_name = EFFICIENTNET_NAME
    best_acc = eff_acc
else:
    best_model = mob_model
    best_name = MOBILENET_NAME
    best_acc = mob_acc

print(f"\n  ★ Best model: {best_name.upper()} (acc={best_acc:.4f})")
best_model.save(BEST_MODEL_PATH)
print(f"  Saved to: {BEST_MODEL_PATH}")

# Save training summary
summary = {
    "best_model": best_name,
    "best_test_accuracy": float(best_acc),
    "mobilenetv2_test_accuracy": float(mob_acc),
    "efficientnetb0_test_accuracy": float(eff_acc),
    "num_classes": num_classes,
    "class_names": class_names,
}

summary_path = os.path.join(REPORT_DIR, "training_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[Info] Summary saved: {summary_path}")

# ── Visualisations ───────────────────────────────────────────
print("\n[Plots] Generating visualisations …")

plot_training_history(mob_hist, MOBILENET_NAME)
plot_training_history(eff_hist, EFFICIENTNET_NAME)

plot_model_comparison({
    MOBILENET_NAME: mob_acc,
    EFFICIENTNET_NAME: eff_acc,
})

plot_confusion_matrix(
    mob_results["y_true"],
    mob_results["y_pred"],
    class_names,
    MOBILENET_NAME,
)

plot_confusion_matrix(
    eff_results["y_true"],
    eff_results["y_pred"],
    class_names,
    EFFICIENTNET_NAME,
)

# Sample predictions using best model
plot_sample_predictions(best_model, test_ds, class_names, best_name)

print("\n" + "=" * 55)
print("  Training Complete!")
print(f"  Plots saved to  : {PLOT_DIR}")
print(f"  Reports saved to: {REPORT_DIR}")
print(f"  Best model saved: {BEST_MODEL_PATH}")
print("=" * 55)
print("\nNext step → Run the GUI: python gui.py")