"""
Compare the production, improved, and externally adapted models on the
same untouched external test split produced by adapt_external_model.py.

Run after adapt_external_model.py. This script never trains or modifies
any model.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parent
ADAPTATION_DIR = PROJECT_ROOT / "external_domain_adaptation"
TEST_MANIFEST = ADAPTATION_DIR / "split" / "external_test.csv"

MODEL_PATHS = {
    "Production": PROJECT_ROOT / "trained_plant_disease_model.keras",
    "Improved": PROJECT_ROOT / "trained_plant_disease_model_improved.keras",
    "External-adapted": (
        ADAPTATION_DIR / "trained_plant_disease_model_external_adapted.keras"
    ),
}

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.int32)


def make_dataset(frame):
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            frame["path"].to_numpy(),
            frame["label"].to_numpy(dtype=np.int32),
        )
    )
    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def evaluate(name, path, dataset, labels):
    if not path.exists():
        print(f"{name}: model not found — skipping")
        return None

    model = tf.keras.models.load_model(path, compile=False)
    probabilities = model.predict(dataset, verbose=0)
    predicted = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(labels, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted,
        average="macro",
        zero_division=0,
    )

    return {
        "model": name,
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "test_images": int(len(labels)),
    }


def main():
    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(
            "External test manifest does not exist. Run "
            "adapt_external_model.py first."
        )

    frame = pd.read_csv(TEST_MANIFEST)
    labels = frame["label"].to_numpy(dtype=np.int32)
    dataset = make_dataset(frame)

    results = []

    print("=" * 72)
    print("PLANTVISION AI — EXTERNAL TEST MODEL COMPARISON")
    print("=" * 72)
    print(f"Untouched test images: {len(frame):,}")

    for name, path in MODEL_PATHS.items():
        result = evaluate(name, path, dataset, labels)
        if result:
            results.append(result)

    if not results:
        raise RuntimeError("No models were available for comparison.")

    results_df = pd.DataFrame(results)
    output_path = ADAPTATION_DIR / "model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nResults:")
    print(results_df.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
