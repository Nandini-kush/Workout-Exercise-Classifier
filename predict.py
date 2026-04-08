# ============================================================
#  predict.py  –  Command-line image/video prediction
#  Usage:
#     python predict.py path/to/image.jpg
#     python predict.py path/to/video.mp4
# ============================================================

import sys
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

from config import BEST_MODEL_PATH, REPORT_DIR, IMG_SIZE

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Number of frames to sample from a video
NUM_VIDEO_FRAMES = 5


def load_class_names():
    class_path = os.path.join(REPORT_DIR, "class_names.json")
    if not os.path.exists(class_path):
        raise FileNotFoundError(
            f"class_names.json not found at {class_path}. Run train.py first."
        )

    with open(class_path, "r") as f:
        return json.load(f)


def load_model():
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {BEST_MODEL_PATH}. Run train.py first."
        )

    return tf.keras.models.load_model(BEST_MODEL_PATH)


def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def predict_image(model, image_path):
    img = Image.open(image_path)
    arr = preprocess_pil_image(img)
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    return probs


def sample_video_frames(video_path, num_frames=NUM_VIDEO_FRAMES):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError(f"Could not read frame count for video: {video_path}")

    # Evenly spaced frame indices
    indices = np.linspace(0, max(frame_count - 1, 0), num=num_frames, dtype=int)

    frames = []
    used_indices = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        arr = preprocess_pil_image(pil_img)

        frames.append(arr)
        used_indices.append(int(idx))

    cap.release()

    if not frames:
        raise ValueError(f"No valid frames could be extracted from: {video_path}")

    return np.array(frames, dtype=np.float32), used_indices


def predict_video(model, video_path, num_frames=NUM_VIDEO_FRAMES):
    frame_batch, used_indices = sample_video_frames(video_path, num_frames=num_frames)

    # Predict each sampled frame
    frame_probs = model.predict(frame_batch, verbose=0)

    # Average probabilities across frames
    avg_probs = np.mean(frame_probs, axis=0)

    return avg_probs, frame_probs, used_indices


def print_top_predictions(probs, class_names, top_k=5):
    top_idx = np.argsort(probs)[::-1][:top_k]

    print(f"\nTop-{top_k} Predictions:")
    print(f"{'Rank':<6} {'Exercise':<28} {'Confidence':>12}")
    print("─" * 52)

    for rank, idx in enumerate(top_idx, 1):
        print(f"{rank:<6} {class_names[idx]:<28} {probs[idx] * 100:>10.2f}%")

    best_idx = top_idx[0]
    print(f"\n→ Predicted: {class_names[best_idx].upper()} ({probs[best_idx] * 100:.1f}%)")


def predict_path(path):
    class_names = load_class_names()
    model = load_model()

    ext = os.path.splitext(path)[1].lower()

    if ext in IMAGE_EXTS:
        probs = predict_image(model, path)

        print(f"\nInput Type: Image")
        print(f"File: {path}")
        print_top_predictions(probs, class_names, top_k=5)

    elif ext in VIDEO_EXTS:
        avg_probs, frame_probs, used_indices = predict_video(model, path, num_frames=NUM_VIDEO_FRAMES)

        print(f"\nInput Type: Video")
        print(f"File: {path}")
        print(f"Sampled Frames: {used_indices}")
        print(f"Prediction Method: Averaged probabilities across {len(used_indices)} frames")

        print_top_predictions(avg_probs, class_names, top_k=5)

        print("\nPer-frame Top Prediction:")
        print(f"{'Frame':<10} {'Predicted Class':<28} {'Confidence':>12}")
        print("─" * 55)

        for frame_idx, probs in zip(used_indices, frame_probs):
            best_idx = int(np.argmax(probs))
            print(f"{frame_idx:<10} {class_names[best_idx]:<28} {probs[best_idx] * 100:>10.2f}%")

    else:
        raise ValueError(
            f"Unsupported file type: {ext}\n"
            f"Supported image types: {sorted(IMAGE_EXTS)}\n"
            f"Supported video types: {sorted(VIDEO_EXTS)}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_or_video_path>")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    try:
        predict_path(path)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)