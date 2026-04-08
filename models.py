# ============================================================
#  models.py  –  MobileNetV2 & EfficientNetB0 model builders
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from config import IMG_SIZE, CHANNELS, LEARNING_RATE


# ── Shared classification head ────────────────────────────────

def _classification_head(base_output, num_classes, dropout_rate=0.4):
    """
    Attaches a compact, regularised head on top of a feature extractor.
    """
    x = layers.GlobalAveragePooling2D()(base_output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return outputs


# ── MobileNetV2 ───────────────────────────────────────────────

def build_mobilenetv2(num_classes, input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)):
    """
    MobileNetV2 with ImageNet weights, frozen base, custom head.
    Very lightweight – good for laptops.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # freeze during phase 1

    inputs  = tf.keras.Input(shape=input_shape)
    # MobileNetV2 expects inputs in [-1, 1]
    x       = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x       = base(x, training=False)
    outputs = _classification_head(x, num_classes)

    model = models.Model(inputs, outputs, name="MobileNetV2_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base


def unfreeze_mobilenetv2(model, base_model, fine_lr=1e-5, unfreeze_from=100):
    """
    Unfreeze the top layers of MobileNetV2 for fine-tuning.
    """
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(fine_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ── EfficientNetB0 ────────────────────────────────────────────

def build_efficientnetb0(num_classes, input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)):
    """
    EfficientNetB0 with ImageNet weights, frozen base, custom head.
    Slightly heavier than MobileNetV2 but usually more accurate.
    """
    base = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs  = tf.keras.Input(shape=input_shape)
    # EfficientNet has its own preprocessing built in (accepts 0-255)
    x       = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
    x       = base(x, training=False)
    outputs = _classification_head(x, num_classes, dropout_rate=0.45)

    model = models.Model(inputs, outputs, name="EfficientNetB0_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base


def unfreeze_efficientnetb0(model, base_model, fine_lr=1e-5, unfreeze_from=200):
    """
    Unfreeze the top portion of EfficientNetB0 for fine-tuning.
    """
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(fine_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ── Callback factory ─────────────────────────────────────────

def get_callbacks(model_name, model_dir, es_patience=7, lr_patience=3):
    """
    Returns a list of Keras callbacks for training.
    """
    import os
    checkpoint_path = os.path.join(model_dir, f"{model_name}_best.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=es_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    return callbacks, checkpoint_path
