"""Fair, reproducible comparison of PlantVision AI Keras checkpoints.

This script evaluates existing checkpoints on the same directory-based validation
split. It does not train, modify, or overwrite any model. The generated report
explicitly labels this as validation data, not an independent held-out test set.

Run from the Plant_Disease_Prediction directory:
    python evaluate_models.py
    python evaluate_models.py --data-dir "path/to/valid" --models production=trained_plant_disease_model.keras candidate=trained_plant_disease_model_improved.keras
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DEFAULT_DATA_DIR = Path(
    "New Plant Diseases Dataset(Augmented)"
) / "New Plant Diseases Dataset(Augmented)" / "valid"
DEFAULT_MODELS = {
    "production": Path("trained_plant_disease_model.keras"),
    "candidate": Path("trained_plant_disease_model_improved.keras"),
}
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_DATA_DIR.parent / "train",
        help="Training directory used to verify that class folders match validation.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_comparison"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=[f"{name}={path}" for name, path in DEFAULT_MODELS.items()],
        help="One or more name=path entries for Keras model files.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def parse_models(entries: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected name=path, received: {entry}")
        name, path = entry.split("=", 1)
        name = name.strip()
        if not name or name in models:
            raise ValueError(f"Model names must be non-empty and unique: {name!r}")
        models[name] = Path(path.strip())
    if not models:
        raise ValueError("At least one model is required.")
    return models


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    model_paths = parse_models(args.models)

    if not args.data_dir.is_dir():
        raise FileNotFoundError(
            f"Validation directory not found: {args.data_dir}\n"
            "Pass --data-dir with the path to your existing validation directory."
        )
    for name, path in model_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint for '{name}' not found: {path}\n"
                "This script intentionally does not download or replace model files."
            )

    if args.train_dir.is_dir():
        train_class_names = sorted(
            path.name for path in args.train_dir.iterdir() if path.is_dir()
        )
        validation_class_names = sorted(
            path.name for path in args.data_dir.iterdir() if path.is_dir()
        )
        if train_class_names != validation_class_names:
            raise ValueError(
                "Training and validation class folders do not match exactly. "
                "Resolve the mapping before comparing model results."
            )
    else:
        raise FileNotFoundError(
            f"Training directory not found: {args.train_dir}. "
            "Pass --train-dir to verify class folder alignment."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # No rescaling is applied: this mirrors the current Streamlit app's RGB,
    # 128x128, float32 pixel input. Change only after verifying model provenance.
    generator = ImageDataGenerator().flow_from_directory(
        str(args.data_dir),
        target_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        class_mode="sparse",
        shuffle=False,
    )
    class_names = list(generator.class_indices.keys())
    y_true = np.asarray(generator.classes, dtype=np.int64)

    if not class_names:
        raise ValueError(f"No class folders found in {args.data_dir}")

    summary: dict[str, Any] = {
        "evaluation_split": "validation",
        "is_independent_held_out_test": False,
        "warning": (
            "These are validation metrics. If this split influenced training, "
            "fine-tuning, threshold selection, or checkpoint selection, do not "
            "report it as an untouched independent test set."
        ),
        "data_directory": str(args.data_dir),
        "sample_count": int(generator.samples),
        "image_size": list(IMAGE_SIZE),
        "pixel_scaling": "raw RGB pixel values, float32; no division by 255",
        "class_names": class_names,
        "models": {},
    }

    per_model_rows: list[list[Any]] = []
    for model_name, model_path in model_paths.items():
        print(f"\nLoading {model_name}: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)

        input_shape = model.input_shape
        output_shape = model.output_shape
        if isinstance(input_shape, list) or isinstance(output_shape, list):
            raise ValueError(f"{model_name}: multi-input/output models are unsupported.")
        if len(input_shape) != 4 or tuple(input_shape[1:3]) != IMAGE_SIZE:
            raise ValueError(
                f"{model_name}: expected model input spatial shape {IMAGE_SIZE}, "
                f"found {input_shape}."
            )
        if int(output_shape[-1]) != len(class_names):
            raise ValueError(
                f"{model_name}: model outputs {output_shape[-1]} classes but "
                f"the validation folder has {len(class_names)} classes."
            )

        generator.reset()
        start = time.perf_counter()
        raw_predictions = model.predict(generator, verbose=1)
        elapsed = time.perf_counter() - start
        if raw_predictions.ndim != 2 or raw_predictions.shape[0] != len(y_true):
            raise ValueError(f"{model_name}: unexpected prediction shape {raw_predictions.shape}.")
        if not np.isfinite(raw_predictions).all():
            raise ValueError(f"{model_name}: predictions contain NaN or infinity.")

        y_pred = np.argmax(raw_predictions, axis=1)
        top_k = min(3, raw_predictions.shape[1])
        top_k_indices = np.argsort(raw_predictions, axis=1)[:, -top_k:]
        top_k_accuracy = float(np.mean([int(actual) in row for actual, row in zip(y_true, top_k_indices)]))
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

        metrics = {
            "checkpoint": str(model_path),
            "input_shape": str(input_shape),
            "output_shape": str(output_shape),
            "parameter_count": int(model.count_params()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "top_3_accuracy": top_k_accuracy,
            "sample_count": int(len(y_true)),
            "inference_seconds_total": float(elapsed),
            "inference_seconds_per_image": float(elapsed / max(1, len(y_true))),
            "classification_report": report,
        }
        summary["models"][model_name] = metrics
        per_model_rows.append([
            model_name,
            str(model_path),
            len(y_true),
            metrics["accuracy"],
            metrics["macro_precision"],
            metrics["macro_recall"],
            metrics["macro_f1"],
            metrics["top_3_accuracy"],
            metrics["parameter_count"],
            metrics["inference_seconds_per_image"],
        ])

        write_csv(
            args.output_dir / f"{model_name}_per_class_metrics.csv",
            ["class", "precision", "recall", "f1-score", "support"],
            [
                [
                    class_name,
                    report[class_name]["precision"],
                    report[class_name]["recall"],
                    report[class_name]["f1-score"],
                    report[class_name]["support"],
                ]
                for class_name in class_names
            ],
        )
        write_csv(
            args.output_dir / f"{model_name}_confusion_matrix.csv",
            ["actual\\predicted", *class_names],
            [[class_names[i], *matrix[i].tolist()] for i in range(len(class_names))],
        )
        print(
            f"{model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro-F1={metrics['macro_f1']:.4f}, "
            f"top-3={metrics['top_3_accuracy']:.4f}"
        )

        del model

    write_csv(
        args.output_dir / "model_comparison.csv",
        [
            "model",
            "checkpoint",
            "samples",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "top_3_accuracy",
            "parameters",
            "inference_seconds_per_image",
        ],
        per_model_rows,
    )
    (args.output_dir / "model_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nReports written to: {args.output_dir.resolve()}")
    print("Reminder: validation results are not independent test results.")


if __name__ == "__main__":
    main()
