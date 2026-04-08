# 🏋️ Workout Exercise Classifier

A complete deep-learning project for classifying workout exercises from images/videos,
using **MobileNetV2** and **EfficientNetB0** with transfer learning, built with TensorFlow/Keras.

---

## 📁 Project Structure

```
workout_project/
│
├── archive (4)/              ← ✅ PLACE YOUR DATASET HERE
│   ├── barbell biceps curl/
│   ├── bench press/
│   ├── deadlift/
│   └── ...                   (20+ class folders, each with images/videos)
│
├── outputs/                  ← Auto-created on training
│   ├── models/               ← Saved .keras model files
│   ├── plots/                ← Accuracy/loss/confusion graphs
│   └── reports/              ← Text reports + class_names.json
│
├── config.py                 ← ⚙️  Edit paths & hyperparameters here
├── data_loader.py            ← Dataset loading & tf.data pipeline
├── models.py                 ← MobileNetV2 & EfficientNetB0 builders
├── train.py                  ← 🚀 Run this FIRST to train both models
├── evaluate.py               ← Evaluation metrics & reports
├── visualize.py              ← All matplotlib/seaborn visualisations
├── predict.py                ← CLI single-image prediction
├── gui.py                    ← 🖥️  Desktop GUI (run after training)
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users** (optional, recommended for speed):
> ```bash
> pip install tensorflow[and-cuda]
> ```

### 2. Place your dataset

Copy (or symlink) your dataset folder into the project root so the structure looks like:
```
workout_project/
└── archive (4)/
    ├── barbell biceps curl/
    ├── bench press/
    └── ...
```

If your folder name differs, edit `DATASET_DIR` in **config.py**.

---

## 🚀 How to Run

### Step 1 — Train the models

```bash
python train.py
```

This will:
- Auto-detect all classes from your dataset folders
- Split data into train / val / test (70 / 15 / 15 %)
- Train **MobileNetV2** (Phase 1: frozen + Phase 2: fine-tune)
- Train **EfficientNetB0** (Phase 1: frozen + Phase 2: fine-tune)
- Compare both on the test set
- Save the **best model** to `outputs/models/best_model.keras`
- Generate all graphs in `outputs/plots/`
- Print classification report to terminal + save to `outputs/reports/`

Training time estimate (CPU-only laptop):
- MobileNetV2 : ~30–60 min (faster, lighter)
- EfficientNetB0: ~45–90 min (heavier, usually more accurate)

> **Tip**: Reduce `EPOCHS_FROZEN` and `EPOCHS_FINE` in `config.py` for a quick test run.

---

### Step 2 — Launch the GUI

```bash
python gui.py
```

Features:
- Browse and open any image (JPG, PNG, BMP) or video (MP4, MOV, AVI)
- See a live preview of the selected file
- Click **Predict** to classify the exercise
- Displays top predicted label, confidence %, and Top-3 results
- Colour-coded confidence bar (green = high, amber = medium, pink = low)

---

### (Optional) Command-line prediction

```bash
python predict.py path/to/your/image.jpg
```

---

## 📊 Outputs

| Output | Location |
|---|---|
| Best model | `outputs/models/best_model.keras` |
| Training history plots | `outputs/plots/{model}_history.png` |
| Model comparison chart | `outputs/plots/model_comparison.png` |
| Confusion matrices | `outputs/plots/{model}_confusion_matrix.png` |
| Sample prediction grid | `outputs/plots/{model}_sample_predictions.png` |
| Classification reports | `outputs/reports/{model}_classification_report.txt` |
| Class names (for GUI) | `outputs/reports/class_names.json` |
| Training summary | `outputs/reports/training_summary.json` |

---

## 🏆 Which model is best?

After training, check `outputs/reports/training_summary.json` for the winner.

**Expected results (typical):**

| Model | Params | Speed (CPU) | Expected Accuracy |
|---|---|---|---|
| MobileNetV2 | ~2.2M | ⚡ Fast | 80–90% |
| EfficientNetB0 | ~4.0M | 🔥 Moderate | 85–93% |

**Why EfficientNetB0 often wins:**
- Compound scaling (depth + width + resolution simultaneously)
- Better feature extraction with similar parameter budget
- Generally outperforms MobileNetV2 on fine-grained classification

**Why MobileNetV2 might be preferred:**
- Much faster inference (great for real-time use)
- Smaller model file
- Still very competitive accuracy

---

## ⚙️ Configuration (config.py)

| Parameter | Default | Description |
|---|---|---|
| `DATASET_DIR` | `./archive (4)` | Path to dataset root |
| `IMG_SIZE` | `(224, 224)` | Input image size |
| `BATCH_SIZE` | `32` | Training batch size |
| `EPOCHS_FROZEN` | `20` | Phase 1 epochs (frozen base) |
| `EPOCHS_FINE` | `20` | Phase 2 epochs (fine-tuning) |
| `LEARNING_RATE` | `1e-3` | Initial learning rate |
| `FINE_LR` | `1e-5` | Fine-tuning learning rate |
| `ES_PATIENCE` | `7` | Early stopping patience |

**Quick test run** — edit `config.py`:
```python
EPOCHS_FROZEN = 5
EPOCHS_FINE   = 5
```

---

## 🛡️ Anti-Overfitting Measures

- ✅ Data augmentation (flip, brightness, contrast, saturation, zoom)
- ✅ Dropout (0.4 → 0.2 cascade)
- ✅ Batch Normalization after each dense layer
- ✅ L2 weight regularisation
- ✅ Early Stopping (restores best weights)
- ✅ ReduceLROnPlateau (halves LR when stuck)
- ✅ Stratified train/val/test split
- ✅ Two-phase training (frozen → fine-tune)

---

## 🛠️ Troubleshooting

**"No sub-folders found"** → Check `DATASET_DIR` in config.py matches your actual folder name.

**Out of memory** → Reduce `BATCH_SIZE` to 16 in config.py.

**Slow training** → Reduce epochs for a test run, or use a GPU.

**GUI won't open** → Make sure you've run `train.py` first so the model exists.
