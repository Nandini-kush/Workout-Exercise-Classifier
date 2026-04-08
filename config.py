# ============================================================
#  config.py  –  Central configuration for Workout Classifier
#  Edit the paths and hyper-parameters here before training.
# ============================================================

import os

# ── Dataset ──────────────────────────────────────────────────
DATASET_DIR   = os.path.join(os.path.dirname(__file__), "archive (4)")  # folder with class sub-folders
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "outputs")       # graphs, models, reports
MODEL_DIR     = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR      = os.path.join(OUTPUT_DIR, "plots")
REPORT_DIR    = os.path.join(OUTPUT_DIR, "reports")

# ── Image settings ───────────────────────────────────────────
IMG_SIZE      = (224,224)#MbileNetV2 & EfficientNetB0 default input
CHANNELS      = 3
BATCH_SIZE    = 16

# ── Split ratios ─────────────────────────────────────────────
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15   # remainder goes to test

# ── Training ─────────────────────────────────────────────────
EPOCHS_FROZEN = 10  # epochs while base layers are frozen
EPOCHS_FINE   = 10 # additional fine-tuning epochs
LEARNING_RATE = 1e-3
FINE_LR       = 1e-5
SEED          = 42

# ── Early stopping ───────────────────────────────────────────
ES_PATIENCE   = 7      # epochs with no improvement before stopping
LR_PATIENCE   = 3      # epochs before reducing LR on plateau

# ── Model names (used as save-file prefixes) ─────────────────
MOBILENET_NAME    = "mobilenetv2"
EFFICIENTNET_NAME = "efficientnetb0"
BEST_MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.keras")

# ── Create output directories ────────────────────────────────
for _dir in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, REPORT_DIR]:
    os.makedirs(_dir, exist_ok=True)
