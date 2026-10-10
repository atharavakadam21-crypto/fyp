"""Analyze candidate-model errors on a labeled manifest.

Writes per-image predictions (including top-3 classes) and the most common
actual->predicted confusion pairs. This is diagnostic only; do not use a
validation set as an unbiased final benchmark.

Example:
  python analyze_mixed_domain_errors.py --model trained_plant_disease_model_mixed_domain.keras --manifest data/mixed_domain_v1/field_validation_manifest.csv --class-names data/mixed_domain_v1/class_names.txt --output-dir error_analysis_field_mixed
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = (224, 224)


def load_manifest(path: Path, class_to_id: dict[str, int]):
    paths, labels = [], []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image_path = Path(row["path"])
            label = row["label"]
            if label not in class_to_id:
                raise ValueError(f"Unknown label {label!r} in {path}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            paths.append(str(image_path.resolve()))
            labels.append(class_to_id[label])
    if not paths:
        raise ValueError(f"No images in manifest: {path}")
    return paths, np.asarray(labels, dtype=np.int64)


def make_dataset(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode(path, label):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
        return tf.cast(image, tf.float32), label

    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size
    ).prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--class-names", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("error_analysis_mixed_domain"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-confusions", type=int, default=20)
    args = parser.parse_args()

    for path in (args.model, args.manifest, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(path)

    class_names = args.class_names.read_text(encoding="utf-8").splitlines()
    if len(class_names) != 38 or len(set(class_names)) != 38:
        raise ValueError(f"Expected 38 unique class names; got {len(class_names)}")
    class_to_id = {name: i for i, name in enumerate(class_names)}
    paths, y_true = load_manifest(args.manifest, class_to_id)

    model = keras.models.load_model(args.model)
    if model.output_shape[-1] != len(class_names):
        raise ValueError("Model output count does not match class_names.txt")

    probabilities = model.predict(make_dataset(paths, y_true, args.batch_size), verbose=1)
    y_pred = probabilities.argmax(axis=1)
    top_k = min(3, len(class_names))
    top_ids = np.argsort(probabilities, axis=1)[:, -top_k:][:, ::-1]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "image_predictions.csv"
    fields = [
        "path", "true_label", "predicted_label", "correct", "confidence",
        "top2_label", "top2_probability", "top3_label", "top3_probability",
    ]
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, path in enumerate(paths):
            ids = top_ids[i]
            row = {
                "path": path,
                "true_label": class_names[y_true[i]],
                "predicted_label": class_names[y_pred[i]],
                "correct": bool(y_true[i] == y_pred[i]),
                "confidence": float(probabilities[i, y_pred[i]]),
                "top2_label": class_names[ids[1]] if top_k > 1 else "",
                "top2_probability": float(probabilities[i, ids[1]]) if top_k > 1 else "",
                "top3_label": class_names[ids[2]] if top_k > 2 else "",
                "top3_probability": float(probabilities[i, ids[2]]) if top_k > 2 else "",
            }
            writer.writerow(row)

    confusions = Counter(
        (class_names[actual], class_names[predicted])
        for actual, predicted in zip(y_true, y_pred)
        if actual != predicted
    )
    confusion_path = output_dir / "top_confusions.csv"
    with confusion_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual_label", "predicted_label", "count"])
        writer.writerows([[actual, predicted, count] for (actual, predicted), count in
                          confusions.most_common(args.top_confusions)])

    mistakes = int(np.sum(y_true != y_pred))
    print("\n========== ERROR ANALYSIS ==========")
    print(f"Images: {len(y_true)}")
    print(f"Correct: {len(y_true) - mistakes}")
    print(f"Incorrect: {mistakes}")
    print(f"Accuracy: {float(np.mean(y_true == y_pred)):.4f}")
    print(f"Per-image predictions: {prediction_path}")
    print(f"Most common confusion pairs: {confusion_path}")
    print("\nTop confusion pairs:")
    for (actual, predicted), count in confusions.most_common(args.top_confusions):
        print(f"{count:>3}  actual={actual}  predicted={predicted}")


if __name__ == "__main__":
    main()
