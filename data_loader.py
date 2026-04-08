# ============================================================
#  data_loader.py  –  Dataset loading, splitting, augmentation
# ============================================================

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import (
    DATASET_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    SEED,
)

AUTOTUNE = tf.data.AUTOTUNE

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
VALID_EXTS = IMAGE_EXTS | VIDEO_EXTS


# ── 1. Collect file paths & labels ───────────────────────────
def collect_paths_and_labels(dataset_dir=DATASET_DIR):
    """
    Walk the dataset directory.
    Each sub-folder name becomes a class label.
    Returns:
        file_paths   : list of file paths
        labels       : list of integer labels
        class_names  : sorted class names
    """
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    class_names = sorted(
        [
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]
    )

    if not class_names:
        raise ValueError(f"No class sub-folders found in: {dataset_dir}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    file_paths, labels = [], []

    for cls in class_names:
        cls_dir = os.path.join(dataset_dir, cls)
        for fname in os.listdir(cls_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTS:
                file_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])

    if not file_paths:
        raise ValueError(f"No valid image/video files found in: {dataset_dir}")

    print(f"[Data] Found {len(file_paths)} files across {len(class_names)} classes.")
    return file_paths, labels, class_names


# ── 2. Train / Val / Test split ──────────────────────────────
def split_dataset(file_paths, labels):
    """
    Stratified split into train / val / test.
    """
    test_frac = 1.0 - TRAIN_SPLIT

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        file_paths,
        labels,
        test_size=test_frac,
        stratify=labels,
        random_state=SEED,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.5,
        stratify=y_tmp,
        random_state=SEED,
    )

    print(f"[Split] Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 3. File-type helpers ─────────────────────────────────────
def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


# ── 4. Load single image or first video frame ────────────────
def load_image_from_path(path: str) -> np.ndarray:
    """
    Load one RGB image.
    If input is a video, extract the first readable frame.
    Returns float32 array in [0,1] with shape (H, W, 3).
    """
    from PIL import Image
    import cv2

    if _is_video(path):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise IOError(f"Cannot read video: {path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
    else:
        img = Image.open(path).convert("RGB")

    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


# ── 5. TensorFlow image parsing ──────────────────────────────
def _parse_image_tf(path, label, num_classes):
    """
    Graph-mode decode for image files only.
    """
    raw = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, num_classes)
    return image, label


# ── 6. Data augmentation ─────────────────────────────────────
def _augment(image, label):
    """
    Augmentation only for training data.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.12)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.03)

    # Mild zoom/crop
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE[0] + 20, IMG_SIZE[1] + 20)
    image = tf.image.random_crop(image, size=[IMG_SIZE[0], IMG_SIZE[1], 3])

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


# ── 7. Build tf.data dataset ─────────────────────────────────
def build_tf_dataset(paths, labels, num_classes, augment=False, shuffle=False):
    """
    Build a tf.data.Dataset from file paths and integer labels.
    Image files are processed in TensorFlow graph mode.
    Video files are loaded as first-frame numpy arrays.
    """
    img_paths, img_labels = [], []
    vid_paths, vid_labels = [], []

    for p, l in zip(paths, labels):
        if _is_video(p):
            vid_paths.append(p)
            vid_labels.append(l)
        elif _is_image(p):
            img_paths.append(p)
            img_labels.append(l)

    datasets = []

    # Image branch
    if img_paths:
        img_ds = tf.data.Dataset.from_tensor_slices((img_paths, img_labels))
        img_ds = img_ds.map(
            lambda p, l: _parse_image_tf(p, l, num_classes),
            num_parallel_calls=AUTOTUNE,
        )
        if augment:
            img_ds = img_ds.map(_augment, num_parallel_calls=AUTOTUNE)
        datasets.append(img_ds)

    # Video branch
    if vid_paths:
        valid_video_arrays = []
        valid_video_labels = []

        print(f"[Data] Pre-loading {len(vid_paths)} video frames …")

        for vp, vl in zip(vid_paths, vid_labels):
            try:
                arr = load_image_from_path(vp)
                valid_video_arrays.append(arr)
                valid_video_labels.append(vl)
            except Exception as e:
                print(f"  [WARN] Skipping video {vp}: {e}")

        if valid_video_arrays:
            vid_arr_np = np.stack(valid_video_arrays).astype(np.float32)
            vid_lbl_oh = tf.one_hot(valid_video_labels, num_classes).numpy()

            vid_ds = tf.data.Dataset.from_tensor_slices((vid_arr_np, vid_lbl_oh))
            if augment:
                vid_ds = vid_ds.map(_augment, num_parallel_calls=AUTOTUNE)
            datasets.append(vid_ds)

    if not datasets:
        raise ValueError("No valid image/video samples found to build dataset.")

    ds = datasets[0]
    for extra_ds in datasets[1:]:
        ds = ds.concatenate(extra_ds)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 2000), seed=SEED)

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


# ── 8. Main entry point ──────────────────────────────────────
def get_datasets():
    """
    Returns:
        train_ds, val_ds, test_ds, class_names, num_classes, steps
    """
    paths, labels, class_names = collect_paths_and_labels()
    num_classes = len(class_names)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(paths, labels)

    train_ds = build_tf_dataset(
        X_train, y_train, num_classes, augment=True, shuffle=True
    )
    val_ds = build_tf_dataset(
        X_val, y_val, num_classes, augment=False, shuffle=False
    )
    test_ds = build_tf_dataset(
        X_test, y_test, num_classes, augment=False, shuffle=False
    )

    steps = {
        "train": max(1, len(X_train) // BATCH_SIZE),
        "val": max(1, len(X_val) // BATCH_SIZE),
        "test": max(1, len(X_test) // BATCH_SIZE),
        "X_test": X_test,
        "y_test": y_test,
    }

    return train_ds, val_ds, test_ds, class_names, num_classes, steps