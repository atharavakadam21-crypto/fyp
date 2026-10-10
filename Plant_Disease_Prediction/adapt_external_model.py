"""
PlantVision AI — External Domain Adaptation Experiment

Purpose
-------
Improve real-world generalization without contaminating the final external
test set. The source external dataset is NEVER modified.

Pipeline
--------
1. Read the external dataset and map its folder names to the model's 38 labels.
2. Create a deterministic stratified 70/15/15 train/validation/test split.
3. Fine-tune the improved CNN on the external TRAIN split only.
4. Use the external VALIDATION split for early stopping/model selection.
5. Evaluate once on the untouched external TEST split.
6. Save the adapted model separately from production.

Important
---------
The external dataset contains 30 classes while the classifier has 38 outputs.
The eight absent model classes remain in the output layer, but receive no
external-domain training examples. This avoids changing the model's contract.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "SOURCE_MODEL",
        str(PROJECT_ROOT / "trained_plant_disease_model_improved.keras"),
    )
)

EXTERNAL_PATH = Path(
    os.getenv(
        "EXTERNAL_DATASET",
        r"D:\FYP\External_Dataset\Test_OOD",
    )
)

OUTPUT_DIR = PROJECT_ROOT / "external_domain_adaptation"
ADAPTED_MODEL_PATH = OUTPUT_DIR / "trained_plant_disease_model_external_adapted.keras"
SPLIT_DIR = OUTPUT_DIR / "split"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"
METRICS_PATH = OUTPUT_DIR / "external_test_metrics.json"
CONFUSION_PATH = OUTPUT_DIR / "external_test_confusion_matrix.csv"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15
TEST_FRACTION = 0.15

EPOCHS = 8
LEARNING_RATE = 2e-6
PATIENCE = 2
UNFREEZE_LAST_N = 6


# ---------------------------------------------------------------------------
# Model class contract
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

EXTERNAL_TO_MODEL = {
    "Cherry___healthy": "Cherry_(including_sour)___healthy",
    "Corn___Cercospora_leaf_spot": (
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
    ),
    "Corn___Common_rust": "Corn_(maize)___Common_rust_",
    "Corn___Northern_Leaf_Blight": "Corn_(maize)___Northern_Leaf_Blight",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def collect_external_images() -> pd.DataFrame:
    if not EXTERNAL_PATH.is_dir():
        raise FileNotFoundError(
            f"External dataset not found: {EXTERNAL_PATH}"
        )

    rows = []
    skipped = []

    for folder in sorted(EXTERNAL_PATH.iterdir()):
        if not folder.is_dir():
            continue

        model_class = EXTERNAL_TO_MODEL.get(folder.name, folder.name)

        if model_class not in CLASS_TO_INDEX:
            skipped.append(folder.name)
            continue

        label_index = CLASS_TO_INDEX[model_class]

        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rows.append(
                {
                    "path": str(image_path.resolve()),
                    "external_class": folder.name,
                    "model_class": model_class,
                    "label": label_index,
                }
            )

    if skipped:
        print("\nSkipped external classes with no model mapping:")
        for name in skipped:
            print(f"  - {name}")

    if not rows:
        raise RuntimeError("No usable external images were found.")

    frame = pd.DataFrame(rows)

    print(f"\nExternal images discovered: {len(frame):,}")
    print(f"Mapped classes: {frame['model_class'].nunique()}")

    return frame


def stratified_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = frame["label"]

    train_df, temp_df = train_test_split(
        frame,
        test_size=(1.0 - TRAIN_FRACTION),
        random_state=SEED,
        stratify=labels,
    )

    temp_test_fraction = TEST_FRACTION / (VALID_FRACTION + TEST_FRACTION)

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=temp_test_fraction,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLIT_DIR / "external_train.csv", index=False)
    valid_df.to_csv(SPLIT_DIR / "external_validation.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "external_test.csv", index=False)

    print("\nSplit sizes:")
    print(f"  Train:      {len(train_df):,}")
    print(f"  Validation: {len(valid_df):,}")
    print(f"  Test:       {len(test_df):,}")


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------


def load_and_resize(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    label = tf.cast(label, tf.int32)

    return image, label


def make_dataset(frame: pd.DataFrame, training: bool) -> tf.data.Dataset:
    paths = frame["path"].to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_and_resize,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomTranslation(0.08, 0.08),
                tf.keras.layers.RandomZoom(0.10),
            ],
            name="external_domain_augmentation",
        )

        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_class_weights(train_df: pd.DataFrame) -> dict[int, float]:
    counts = Counter(train_df["label"].tolist())
    total = len(train_df)
    class_count = len(counts)

    weights = {
        label: total / (class_count * count)
        for label, count in sorted(counts.items())
    }

    # Keras requires class_weight keys for every output class (0..37).
    # External classes absent from this dataset receive a neutral weight.
    return {
        label: weights.get(label, 1.0)
        for label in range(len(CLASS_NAMES))
    }


def configure_model(model: tf.keras.Model) -> tf.keras.Model:
    # Freeze the feature extractor first, then expose only the last few layers
    # to the small external-domain adaptation learning rate.
    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[-UNFREEZE_LAST_N:]:
        layer.trainable = True

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} outputs, got {model.output_shape[-1]}"
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def evaluate_test(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    test_ds: tf.data.Dataset,
) -> dict:
    probabilities = model.predict(test_ds, verbose=1)
    predicted = np.argmax(probabilities, axis=1)
    actual = test_df["label"].to_numpy()

    accuracy = accuracy_score(actual, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        average="macro",
        zero_division=0,
    )

    labels_present = sorted(set(actual.tolist()) | set(predicted.tolist()))
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels_present,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[CLASS_NAMES[i] for i in labels_present],
        columns=[CLASS_NAMES[i] for i in labels_present],
    )
    matrix_df.to_csv(CONFUSION_PATH)

    return {
        "test_images": int(len(test_df)),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "labels_evaluated": len(labels_present),
    }


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Source model not found: {MODEL_PATH}"
        )

    print("=" * 72)
    print("PLANTVISION AI — EXTERNAL DOMAIN ADAPTATION")
    print("=" * 72)
    print(f"Source model: {MODEL_PATH}")
    print(f"External data: {EXTERNAL_PATH}")
    print(f"Output model: {ADAPTED_MODEL_PATH}")

    frame = collect_external_images()
    train_df, valid_df, test_df = stratified_split(frame)

    save_splits(train_df, valid_df, test_df)

    train_ds = make_dataset(train_df, training=True)
    valid_ds = make_dataset(valid_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print(f"\nLoaded model with {model.count_params():,} parameters.")

    model = configure_model(model)

    trainable = sum(
        int(np.prod(layer.shape))
        for layer in model.trainable_weights
    )
    print(f"Trainable parameters after freezing: {trainable:,}")

    class_weights = make_class_weights(train_df)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ADAPTED_MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("\nStarting external-domain fine-tuning...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    with HISTORY_PATH.open("w", encoding="utf-8") as handle:
        # Convert NumPy/TensorFlow scalar values to native Python floats.
        serializable_history = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        json.dump(serializable_history, handle, indent=2)

    if ADAPTED_MODEL_PATH.exists():
        model = tf.keras.models.load_model(
            ADAPTED_MODEL_PATH,
            compile=False,
        )

    metrics = evaluate_test(model, test_df, test_ds)

    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("\n" + "=" * 72)
    print("UNTOUCHED EXTERNAL TEST RESULT")
    print("=" * 72)
    print(f"Accuracy:        {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro precision: {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro recall:    {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1:        {metrics['macro_f1'] * 100:.2f}%")
    print(f"Test images:     {metrics['test_images']:,}")
    print(f"\nAdapted model: {ADAPTED_MODEL_PATH}")
    print(f"Metrics:        {METRICS_PATH}")
    print(f"Confusion:      {CONFUSION_PATH}")
    print("\nProduction model was not modified.")


if __name__ == "__main__":
    main()
