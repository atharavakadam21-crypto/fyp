"""Evaluate a candidate model against a labeled CSV manifest.

The manifest must have columns: path,label. Labels must match class_names.txt.
This script reports top-1 accuracy, macro precision/recall/F1, per-class metrics,
a confusion matrix, and a basic expected calibration error (ECE).

Example:
  python evaluate_mixed_domain_model.py --model trained_plant_disease_model_mixed_domain.keras --manifest external_domain_adaptation/split/external_test.csv --class-names data/mixed_domain_v1/class_names.txt
"""
from __future__ import annotations

import argparse
import csv
import json
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
                raise ValueError(f"Manifest label {label!r} is not in class_names.txt")
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            paths.append(str(image_path.resolve()))
            labels.append(class_to_id[label])
    if not paths:
        raise ValueError(f"No images found in manifest: {path}")
    return paths, np.asarray(labels, dtype=np.int64)


def build_dataset(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode(path, label):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
        image = tf.cast(image, tf.float32)  # EfficientNetV2 include_preprocessing=True
        return image, label

    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size
    ).prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--class-names", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_mixed_domain"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ece-bins", type=int, default=15)
    args = parser.parse_args()

    for path in (args.model, args.manifest, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(path)

    class_names = args.class_names.read_text(encoding="utf-8").splitlines()
    if len(class_names) != 38:
        raise ValueError(f"Expected 38 class names; found {len(class_names)}")
    class_to_id = {name: i for i, name in enumerate(class_names)}
    paths, y_true = load_manifest(args.manifest, class_to_id)

    model = keras.models.load_model(args.model)
    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Model outputs {model.output_shape[-1]} classes, but class list has "
            f"{len(class_names)}. Do not evaluate with a mismatched class list."
        )

    dataset = build_dataset(paths, y_true, args.batch_size)
    probabilities = model.predict(dataset, verbose=1)
    y_pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    n_classes = len(class_names)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        confusion[actual, predicted] += 1

    per_class = []
    precisions, recalls, f1s = [], [], []
    for i, name in enumerate(class_names):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int(confusion[i, :].sum())
        if support:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        per_class.append({
            "class": name,
            "support": support,
            "correct": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    accuracy = float(np.mean(y_true == y_pred))
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # Expected calibration error: confidence bins compared with observed accuracy.
    ece = 0.0
    bin_rows = []
    edges = np.linspace(0.0, 1.0, args.ece_bins + 1)
    for i in range(args.ece_bins):
        lower, upper = edges[i], edges[i + 1]
        if i == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        count = int(mask.sum())
        if count:
            mean_conf = float(confidence[mask].mean())
            mean_acc = float((y_pred[mask] == y_true[mask]).mean())
            ece += count / len(y_true) * abs(mean_conf - mean_acc)
        else:
            mean_conf = 0.0
            mean_acc = 0.0
        bin_rows.append({
            "bin_lower": lower,
            "bin_upper": upper,
            "count": count,
            "mean_confidence": mean_conf,
            "accuracy": mean_acc,
        })

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class[0].keys()))
        writer.writeheader()
        writer.writerows(per_class)

    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *class_names])
        for name, row in zip(class_names, confusion.tolist()):
            writer.writerow([name, *row])

    with (output_dir / "calibration_bins.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bin_rows)

    summary = {
        "model": str(args.model.resolve()),
        "manifest": str(args.manifest.resolve()),
        "images_evaluated": len(y_true),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "ece": float(ece),
        "ece_bins": args.ece_bins,
        "classes_present": int(sum(item["support"] > 0 for item in per_class)),
        "note": "Interpret results only if manifest labels are verified and this set was not used for training/model selection.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== EVALUATION ==========")
    print(f"Images:          {len(y_true):,}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"ECE:             {ece:.4f}")
    print(f"Results saved:   {output_dir}")


if __name__ == "__main__":
    main()
