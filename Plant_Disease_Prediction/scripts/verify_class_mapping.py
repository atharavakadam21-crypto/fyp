"""Verify the checkpoint's class-index semantics on the original validation set."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

SEED = 42
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


def load_names(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = [data[str(i)] for i in range(len(data))]
    if len(names) != 38 or len(set(names)) != 38:
        raise ValueError(f"Expected 38 unique indexed class names, got {len(names)}")
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trained_plant_disease_model_improved.keras")
    parser.add_argument("--dataset", default=os.path.join(
        "New Plant Diseases Dataset(Augmented)",
        "New Plant Diseases Dataset(Augmented)", "valid"))
    parser.add_argument("--mapping", default="class_names.json")
    parser.add_argument("--output", default="reports/class_mapping_check.md")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    model_path, dataset_path = Path(args.model), Path(args.dataset)
    mapping_path, output_path = Path(args.mapping), Path(args.output)
    names = load_names(mapping_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {dataset_path}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path, labels="inferred", label_mode="int",
        color_mode="rgb", batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE, shuffle=False, seed=SEED,
        interpolation="bilinear",
    )
    inferred = list(dataset.class_names)
    if inferred != names:
        raise ValueError(
            "Validation folder order differs from class_names.json. "
            f"First mismatch: {next(((i, a, b) for i, (a, b) in enumerate(zip(inferred, names)) if a != b), 'class count')}"
        )
    model = tf.keras.models.load_model(model_path, compile=False)
    if int(model.output_shape[-1]) != len(names):
        raise ValueError(f"Model outputs {model.output_shape[-1]} classes; mapping has {len(names)}")

    per_class_total = np.zeros(len(names), dtype=np.int64)
    per_class_correct = np.zeros(len(names), dtype=np.int64)
    total = correct = 0
    for images, labels in dataset:
        probabilities = model(images, training=False).numpy()
        predicted = np.argmax(probabilities, axis=1)
        actual = labels.numpy().astype(int)
        correct_mask = predicted == actual
        total += len(actual)
        correct += int(correct_mask.sum())
        for index in actual:
            per_class_total[index] += 1
        for index in actual[correct_mask]:
            per_class_correct[index] += 1

    overall = correct / total if total else 0.0
    rows = []
    suspicious = []
    for i, name in enumerate(names):
        count = int(per_class_total[i])
        class_acc = float(per_class_correct[i] / count) if count else 0.0
        flag = count == 0 or class_acc < 0.50 or class_acc < overall - 0.30
        if flag:
            suspicious.append(name)
        rows.append(f"| {i} | {name} | {int(per_class_correct[i])}/{count} | {class_acc:.2%} | {'CHECK' if flag else 'OK'} |")

    lines = [
        "# Class mapping verification", "",
        f"- Model: `{model_path}`",
        f"- Validation dataset: `{dataset_path}`",
        f"- Mapping: `{mapping_path}`",
        "- Ordering rule: TensorFlow `image_dataset_from_directory` inferred alphabetical folder order.",
        f"- Evaluated: {total:,}; correct: {correct:,}; overall accuracy: {overall:.4%}",
        "- Suspicious-class rule: no examples, accuracy below 50%, or at least 30 percentage points below overall accuracy.",
        "", "| Index | Class | Correct / Total | Accuracy | Status |",
        "|---:|---|---:|---:|---|", *rows, "",
        "## Result", "",
        ("**REVIEW REQUIRED:** " + ", ".join(suspicious) if suspicious else "**No classes triggered the heuristic.**"),
        "",
        "This is a diagnostic heuristic, not proof of a shifted index. Confirm class order against the original training artifacts before claiming the mapping is verified.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"Full report: {output_path}")
    return 1 if suspicious else 0


if __name__ == "__main__":
    raise SystemExit(main())
