"""Deterministic, single-entry-point PlantVision evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from preprocess import preprocess_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def read_class_names(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = [data[str(i)] for i in range(len(data))]
    if len(names) != 38 or len(set(names)) != 38:
        raise ValueError(f"Expected 38 unique class names; got {len(names)}")
    return names


def read_mapping(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("Reading label_mapping.yaml requires PyYAML (pip install pyyaml)") from error
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Mapping file must contain a label-to-class mapping: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Keras .keras checkpoint")
    parser.add_argument("--dataset", required=True, help="Folder with one true-label subfolder per class")
    parser.add_argument("--mapping", required=True, help="label_mapping.yaml or JSON mapping")
    parser.add_argument("--output", required=True, help="Directory for predictions.csv and metrics.json")
    parser.add_argument("--class-names", default="class_names.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    model_path, dataset_path = Path(args.model), Path(args.dataset)
    mapping_path, output_dir = Path(args.mapping), Path(args.output)
    class_names = read_class_names(Path(args.class_names))
    mapping = read_mapping(mapping_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    if int(model.output_shape[-1]) != len(class_names):
        raise ValueError(f"Model has {model.output_shape[-1]} outputs but mapping has {len(class_names)} names")

    files = sorted(
        p for p in dataset_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    rows = []
    skipped = []
    images, metadata = [], []
    for path in files:
        relative = path.relative_to(dataset_path)
        external_label = relative.parts[0] if len(relative.parts) > 1 else ""
        mapped_label = mapping.get(external_label, external_label)
        if mapped_label is not None and mapped_label not in class_names:
            raise ValueError(f"Mapping target {mapped_label!r} for {external_label!r} is absent from class_names.json")
        try:
            with Image.open(path) as image:
                tensor = preprocess_image(image)
            images.append(tensor)
            metadata.append((path, external_label, mapped_label))
        except Exception as error:
            skipped.append({"filepath": str(path), "reason": str(error)})

    if not images:
        raise RuntimeError("No readable images found.")

    predicted_indices, confidences, top3_values = [], [], []
    for start in range(0, len(images), args.batch_size):
        batch = np.stack(images[start:start + args.batch_size])
        probabilities = model(batch, training=False).numpy()
        order = np.argsort(-probabilities, axis=1, kind="stable")
        for prob, indices in zip(probabilities, order):
            predicted_indices.append(int(indices[0]))
            confidences.append(float(prob[indices[0]]))
            top3_values.append([
                {"index": int(i), "name": class_names[int(i)], "confidence": float(prob[i])}
                for i in indices[:3]
            ])

    total = len(files)
    evaluated = len(metadata)
    mapped_count = 0
    overall_correct = 0
    mapped_correct = 0
    class_counts = defaultdict(lambda: {"support": 0, "correct": 0, "precision_tp": 0})
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    csv_rows = []
    for (path, external_label, mapped_label), pred, conf, top3 in zip(
        metadata, predicted_indices, confidences, top3_values
    ):
        is_mapped = mapped_label is not None and mapped_label in class_names
        true_idx = class_names.index(mapped_label) if is_mapped else None
        is_correct = is_mapped and pred == true_idx
        overall_correct += int(bool(is_correct))
        if is_mapped:
            mapped_count += 1
            mapped_correct += int(bool(is_correct))
            class_counts[mapped_label]["support"] += 1
            class_counts[mapped_label]["correct"] += int(bool(is_correct))
            matrix[true_idx, pred] += 1
        csv_rows.append({
            "filepath": str(path),
            "true_label": external_label,
            "mapped_label": "" if mapped_label is None else mapped_label,
            "predicted_idx": pred,
            "predicted_name": class_names[pred],
            "confidence": f"{conf:.10f}",
            "top3": json.dumps(top3, ensure_ascii=False, separators=(",", ":")),
        })

    # Precision/recall/F1 computed on mapped samples; unmapped rows count as
    # incorrect in the all-readable-sample accuracy denominator.
    per_class = {}
    for i, name in enumerate(class_names):
        tp = int(matrix[i, i])
        support = int(matrix[i].sum())
        predicted_as = int(matrix[:, i].sum())
        per_class[name] = {
            "support": support,
            "correct": tp,
            "accuracy": (tp / support) if support else None,
            "precision": (tp / predicted_as) if predicted_as else 0.0,
            "recall": (tp / support) if support else 0.0,
            "f1": (2 * tp / (predicted_as + support)) if predicted_as + support else 0.0,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "filepath", "true_label", "mapped_label", "predicted_idx",
            "predicted_name", "confidence", "top3"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    metrics = {
        "model": str(model_path),
        "dataset": str(dataset_path),
        "mapping_file": str(mapping_path),
        "seed": SEED,
        "n_total": total,
        "n_skipped": len(skipped),
        "n_evaluated": evaluated,
        "accuracy": (overall_correct / evaluated) if evaluated else None,
        "accuracy_definition": "All readable images; unmapped true labels count as incorrect.",
        "mapped_class_count": mapped_count,
        "mapped_class_accuracy": (mapped_correct / mapped_count) if mapped_count else None,
        "mapped_class_accuracy_definition": "Only readable images with an explicit target in class_names.json.",
        "per_class_metrics": per_class,
        "confusion_matrix_labels": class_names,
        "confusion_matrix": matrix.tolist(),
        "skipped": skipped,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Total image files: {total}")
    print(f"Skipped unreadable: {len(skipped)}")
    print(f"Evaluated: {evaluated}")
    print(f"Overall accuracy (unmapped counted wrong): {metrics['accuracy']:.4%}")
    print(f"Mapped-class accuracy: {metrics['mapped_class_accuracy']:.4%}" if metrics["mapped_class_accuracy"] is not None else "Mapped-class accuracy: N/A")
    print(f"Results: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
