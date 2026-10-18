"""Fit temperature scaling on a deterministic split of the held-out validation set.

No external-test images are read. The model is used in inference mode, and the
checkpoint is not modified. Outputs are written under reports/calibration/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from preprocess import preprocess_image

SEED = 42
IMAGE_SIZE = (128, 128)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            score += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return score


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="trained_plant_disease_model_improved.keras")
    parser.add_argument("--validation", default=os.path.join(
        "New Plant Diseases Dataset(Augmented)",
        "New Plant Diseases Dataset(Augmented)", "valid"))
    parser.add_argument("--mapping", default="class_names.json")
    parser.add_argument("--output", default="reports/calibration")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    model_path, validation_path = Path(args.model), Path(args.validation)
    output = Path(args.output)
    names = [json.loads(Path(args.mapping).read_text(encoding="utf-8"))[str(i)] for i in range(38)]
    if not model_path.is_file() or not validation_path.is_dir():
        raise FileNotFoundError("Model or validation directory not found; see --model and --validation.")

    files, labels = [], []
    for label_index, class_name in enumerate(names):
        class_dir = validation_path / class_name
        if not class_dir.is_dir():
            raise ValueError(f"Validation class folder missing: {class_dir}")
        class_files = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        for p in class_files:
            files.append(p)
            labels.append(label_index)
    if not files:
        raise RuntimeError("No validation images found.")

    # Stratified, deterministic half split: calibration fit and held-out reporting.
    fit_indices, eval_indices = [], []
    by_class = {i: [] for i in range(len(names))}
    for i, label in enumerate(labels):
        by_class[label].append(i)
    for indices in by_class.values():
        midpoint = len(indices) // 2
        fit_indices.extend(indices[:midpoint])
        eval_indices.extend(indices[midpoint:])
    if not fit_indices or not eval_indices:
        raise RuntimeError("Validation set is too small to split for calibration.")

    model = tf.keras.models.load_model(model_path, compile=False)
    if int(model.output_shape[-1]) != len(names):
        raise ValueError("Checkpoint output dimension does not match class_names.json.")

    def predict(indices):
        probabilities = []
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            batch = []
            for idx in batch_indices:
                with Image.open(files[idx]) as image:
                    batch.append(preprocess_image(image, IMAGE_SIZE))
            outputs = model(np.stack(batch), training=False).numpy()
            probabilities.extend(outputs)
        probs = np.asarray(probabilities, dtype=np.float64)
        # The network's final layer is softmax. Convert probabilities to stable
        # log-probabilities, then apply temperature to these logits.
        logits = np.log(np.clip(probs, 1e-12, 1.0))
        return logits, np.asarray([labels[i] for i in indices], dtype=np.int64)

    fit_logits, fit_y = predict(fit_indices)
    eval_logits, eval_y = predict(eval_indices)

    # Optimize log-temperature so T is constrained positive.
    log_temperature = tf.Variable(0.0, dtype=tf.float64)
    logits_tensor = tf.constant(fit_logits, dtype=tf.float64)
    labels_tensor = tf.constant(fit_y, dtype=tf.int32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)
    for _ in range(500):
        with tf.GradientTape() as tape:
            temperature = tf.exp(log_temperature)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    labels_tensor, logits_tensor / temperature, from_logits=True
                )
            )
        gradient = tape.gradient(loss, [log_temperature])
        optimizer.apply_gradients(zip(gradient, [log_temperature]))
        log_temperature.assign(tf.clip_by_value(log_temperature, np.log(0.05), np.log(20.0)))
    temperature = float(tf.exp(log_temperature).numpy())

    before = softmax(eval_logits)
    after = softmax(eval_logits / temperature)
    true = eval_y
    pred_before = before.argmax(axis=1)
    pred_after = after.argmax(axis=1)
    correct_before = pred_before == true
    correct_after = pred_after == true
    conf_before = before.max(axis=1)
    conf_after = after.max(axis=1)
    ece_before = ece(conf_before, correct_before)
    ece_after = ece(conf_after, correct_after)

    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration_metrics.json").write_text(json.dumps({
        "model": str(model_path),
        "validation_dataset": str(validation_path),
        "seed": SEED,
        "fit_samples": len(fit_y),
        "held_out_evaluation_samples": len(eval_y),
        "temperature": temperature,
        "accuracy_before": float(correct_before.mean()),
        "accuracy_after": float(correct_after.mean()),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "note": "Temperature fitted on half of the original validation set and reported on the other deterministic half. This is validation-only calibration, not external-test evidence.",
    }, indent=2) + "\n", encoding="utf-8")

    # Reliability diagram.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    for conf, corr, label in [(conf_before, correct_before, "Before"), (conf_after, correct_after, "After")]:
        edges = np.linspace(0, 1, 16)
        xs, ys = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (conf >= low) & (conf < high if high < 1 else conf <= high)
            if mask.any():
                xs.append(float(conf[mask].mean()))
                ys.append(float(corr[mask].mean()))
        ax.plot(xs, ys, marker="o", label=label)
    ax.set(xlabel="Mean confidence", ylabel="Accuracy", title="Validation reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "reliability_diagram.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(conf_before[correct_before], bins=20, alpha=0.6, label="Correct")
    ax.hist(conf_before[~correct_before], bins=20, alpha=0.6, label="Wrong")
    ax.set(xlabel="Maximum softmax confidence (before calibration)", ylabel="Image count", title="Validation confidence: correct vs wrong")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "confidence_correct_vs_wrong.png", dpi=160)
    plt.close(fig)

    # Accuracy-versus-coverage for confidence and predictive entropy rejection.
    max_entropy = float(np.log(len(names)))
    rows = []
    for method in ("max_softmax", "entropy"):
        for threshold in np.linspace(0.0, 1.0, 21):
            if method == "entropy":
                entropy = -(after * np.log(np.clip(after, 1e-12, 1.0))).sum(axis=1)
                accepted = entropy <= threshold * max_entropy
                threshold_value = threshold * max_entropy
            else:
                accepted = conf_after >= threshold
                threshold_value = threshold
            coverage = float(accepted.mean())
            accuracy = float(correct_after[accepted].mean()) if accepted.any() else None
            rows.append({"method": method, "threshold": threshold_value, "coverage": coverage, "accuracy_on_accepted": accuracy})
    with (output / "accuracy_vs_coverage.csv").open("w", encoding="utf-8") as handle:
        handle.write("method,threshold,coverage,accuracy_on_accepted\n")
        for row in rows:
            acc = "" if row["accuracy_on_accepted"] is None else f"{row['accuracy_on_accepted']:.8f}"
            handle.write(f"{row['method']},{row['threshold']:.8f},{row['coverage']:.8f},{acc}\n")

    lines = [
        "# Validation calibration report", "",
        f"- Temperature: {temperature:.6f}",
        f"- Held-out evaluation samples: {len(eval_y)}",
        f"- Accuracy before/after: {correct_before.mean():.4%} / {correct_after.mean():.4%}",
        f"- ECE before/after: {ece_before:.6f} / {ece_after:.6f}",
        "- Artifacts: `reliability_diagram.png`, `confidence_correct_vs_wrong.png`, `accuracy_vs_coverage.csv`, `calibration_metrics.json`.",
        "- Rejection: the CSV reports accepted-sample accuracy and coverage for max-softmax confidence and normalized predictive entropy thresholds.",
        "- Interpretation: high confidence on incorrect predictions can reflect miscalibration or distribution shift; it is not proof of overfitting.",
        "- The external test set is not used to fit temperature or calculate these calibration metrics.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
