"""Run repeatable Grad-CAM checks over one or more images.

Examples (from this directory):
    python validate_gradcam.py
    python validate_gradcam.py test_images/test/TomatoEarlyBlight1.JPG
    python validate_gradcam.py test_images/test/TomatoEarlyBlight1.JPG test_images/test/TomatoHealthy1.JPG

Saves original/heatmap/overlay images and one cumulative CSV report.
Deletion tests are diagnostics, not proof that highlighted pixels are disease lesions.
"""
import ast
import csv
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "trained_plant_disease_model.keras"
MAIN_PATH = ROOT / "main.py"
OUTPUT_DIR = ROOT / "gradcam_validation"
DEFAULT_IMAGE = ROOT / "test_images" / "test" / "TomatoEarlyBlight1.JPG"
MASK_FRACTIONS = (0.10, 0.20, 0.30)
RANDOM_TRIALS = 30
SEED = 20261027


def get_class_names():
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CLASS_NAMES"
            for target in node.targets
        ):
            names = ast.literal_eval(node.value)
            if isinstance(names, list) and len(names) == 38:
                return names
    raise RuntimeError("Could not read the 38 CLASS_NAMES from main.py.")


def preprocess(image):
    return np.expand_dims(
        np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float32),
        axis=0,
    )


def _score_tensor(model, predictions, class_index):
    """Use logits where the model exposes a final softmax Dense layer."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense) and layer.activation == tf.keras.activations.softmax:
            try:
                # The Dense input is the pre-softmax representation. Applying its
                # kernel directly avoids gradients shrinking due to softmax saturation.
                dense_input = tf.keras.Model(model.inputs, layer.input)
                return tf.linalg.matmul(
                    dense_input.output, layer.kernel
                ) + (layer.bias if layer.use_bias else 0.0)
            except Exception:
                break
    # Generic fallback for models whose output head cannot be inspected.
    return tf.math.log(tf.clip_by_value(predictions, 1e-7, 1.0))


def gradcam(model, image, class_index):
    candidates = []
    for name in ("conv2d_7", "conv2d_8", "conv2d_9"):
        try:
            layer = model.get_layer(name)
            if isinstance(layer, tf.keras.layers.Conv2D):
                candidates.append(layer)
        except ValueError:
            continue
    if not candidates:
        candidates = [
            layer for layer in model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
    if not candidates:
        raise RuntimeError("No convolutional layer is available for Grad-CAM.")

    # If the classifier ends in Dense(softmax), differentiate the pre-softmax
    # logit instead of its probability. This avoids saturated probabilities
    # making a useful explanation look blank.
    softmax_head = next((
        layer for layer in reversed(model.layers)
        if isinstance(layer, tf.keras.layers.Dense)
        and layer.activation == tf.keras.activations.softmax
    ), None)
    last_error = None
    for target in candidates:
        try:
            outputs = [target.output, model.output]
            if softmax_head is not None:
                outputs = [target.output, softmax_head.input]
            grad_model = tf.keras.Model(model.inputs, outputs)
            tensor = tf.convert_to_tensor(preprocess(image))
            with tf.GradientTape() as tape:
                activations, head_or_predictions = grad_model(tensor, training=False)
                if softmax_head is not None:
                    logits = tf.linalg.matmul(head_or_predictions, softmax_head.kernel)
                    if softmax_head.use_bias:
                        logits = tf.nn.bias_add(logits, softmax_head.bias)
                    score = logits[:, class_index]
                else:
                    score = tf.math.log(tf.clip_by_value(
                        head_or_predictions[:, class_index], 1e-7, 1.0
                    ))
            gradients = tape.gradient(score, activations)
            if gradients is None:
                raise RuntimeError(f"Gradients are disconnected at {target.name}.")
            weights = tf.reduce_mean(gradients, axis=(1, 2))
            heat = tf.reduce_sum(
                activations[0] * weights[0][tf.newaxis, tf.newaxis, :], axis=-1
            )
            heat = tf.nn.relu(heat)
            maximum = tf.reduce_max(heat)
            heat = tf.math.divide_no_nan(heat, maximum).numpy()
            if not np.isfinite(heat).all() or float(heat.max()) <= 0:
                raise RuntimeError(f"Grad-CAM map is empty at {target.name}.")
            resized = Image.fromarray(
                np.uint8(np.clip(heat, 0, 1) * 255)
            ).resize(image.size, Image.Resampling.BILINEAR)
            return np.asarray(resized, dtype=np.float32) / 255.0, target.name, tuple(
                int(d) if d is not None else -1 for d in activations.shape[1:]
            )
        except Exception as error:
            last_error = error
    raise RuntimeError("Could not generate a valid Grad-CAM map.") from last_error


def probabilities_for(model, image):
    values = np.asarray(model(preprocess(image), training=False)[0], dtype=np.float64)
    # Support either softmax probabilities or raw logits.
    if np.all(values >= 0) and np.isclose(values.sum(), 1.0, atol=1e-3):
        return np.clip(values, 1e-7, 1.0)
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def save_heatmap_and_overlay(image, heat, stem, output_dir):
    import matplotlib

    # Use the modern colormap registry; avoid the deprecated cm.get_cmap API.
    cmap = matplotlib.colormaps["jet"]
    heat_rgb = np.uint8(np.clip(cmap(heat)[..., :3], 0, 1) * 255)
    colored = Image.fromarray(heat_rgb).convert("RGB")
    overlay = Image.blend(image.convert("RGB"), colored, alpha=0.35)
    image.convert("RGB").save(output_dir / f"{stem}_original.png")
    Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255)).save(
        output_dir / f"{stem}_heatmap.png"
    )
    overlay.save(output_dir / f"{stem}_overlay.png")


def masked_image(rgb, selected, mean_color):
    flat = rgb.reshape(-1, 3).copy()
    flat[selected] = mean_color
    return Image.fromarray(np.uint8(np.clip(flat.reshape(rgb.shape), 0, 255)))


def validate_image(model, classes, image_path, rng):
    image = Image.open(image_path).convert("RGB")
    probabilities = probabilities_for(model, image)
    top_indices = np.argsort(probabilities)[::-1][:3]
    class_index = int(top_indices[0])
    heat, layer_name, feature_shape = gradcam(model, image, class_index)

    stem = image_path.stem
    save_heatmap_and_overlay(image, heat, stem, OUTPUT_DIR)
    rgb = np.asarray(image, dtype=np.float32)
    flat_heat = heat.reshape(-1)
    flat_rgb = rgb.reshape(-1, 3)
    mean_color = flat_rgb.mean(axis=0)
    baseline = float(probabilities[class_index])
    rows = []

    for fraction in MASK_FRACTIONS:
        count = max(1, int(round(fraction * flat_heat.size)))
        salient = np.argpartition(flat_heat, -count)[-count:]
        salient_score = float(probabilities_for(
            model, masked_image(rgb, salient, mean_color)
        )[class_index])
        random_scores = []
        for _ in range(RANDOM_TRIALS):
            selected = rng.choice(flat_heat.size, size=count, replace=False)
            random_scores.append(float(probabilities_for(
                model, masked_image(rgb, selected, mean_color)
            )[class_index]))
        random_mean = float(np.mean(random_scores))
        random_std = float(np.std(random_scores, ddof=1))
        salient_drop = baseline - salient_score
        random_drop_mean = baseline - random_mean
        rows.append({
            "image": str(image_path),
            "predicted_class": classes[class_index],
            "confidence": baseline,
            "target_layer": layer_name,
            "feature_map_shape": str(feature_shape),
            "heatmap_min": float(heat.min()),
            "heatmap_max": float(heat.max()),
            "heatmap_std": float(heat.std()),
            "mask_fraction": fraction,
            "random_trials": RANDOM_TRIALS,
            "confidence_after_salient_mask": salient_score,
            "confidence_after_random_mask_mean": random_mean,
            "confidence_after_random_mask_std": random_std,
            "salient_confidence_drop": salient_drop,
            "random_confidence_drop_mean": random_drop_mean,
            "salient_drop_exceeds_random_mean": salient_drop > random_drop_mean,
        })

    print(f"\n{image_path.name}: {classes[class_index]} ({baseline:.4f})")
    print(f"  Grad-CAM layer: {layer_name}, feature map: {feature_shape}")
    print(f"  Heatmap: min={heat.min():.4f}, max={heat.max():.4f}, std={heat.std():.4f}")
    for row in rows:
        print(
            f"  Mask {row['mask_fraction']:.0%}: salient score "
            f"{row['confidence_after_salient_mask']:.4f}; random mean "
            f"{row['confidence_after_random_mask_mean']:.4f} "
            f"± {row['confidence_after_random_mask_std']:.4f}; "
            f"salient drop larger={row['salient_drop_exceeds_random_mean']}"
        )
    print("  Top predictions:")
    for index in top_indices:
        print(f"    {classes[int(index)]}: {probabilities[int(index)]:.4f}")
    return rows


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not MAIN_PATH.is_file():
        raise FileNotFoundError(f"main.py not found: {MAIN_PATH}")
    if sys.argv[1:]:
        paths = [Path(arg) for arg in sys.argv[1:]]
        paths = [p if p.is_absolute() else ROOT / p for p in paths]
    else:
        # Test images may be copied either as test_images/<file> or
        # test_images/test/<file>; find the default sample in either layout.
        matches = list(ROOT.glob("test_images/**/TomatoEarlyBlight1.JPG"))
        paths = [matches[0] if matches else DEFAULT_IMAGE]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing image(s):\n  " + "\n  ".join(missing)
            + "\nPass existing image paths or copy test_images/test into this folder."
        )

    classes = get_class_names()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    all_rows = []
    for path in paths:
        all_rows.extend(validate_image(model, classes, path, rng))

    report_path = OUTPUT_DIR / "gradcam_validation_report.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved visualizations and combined report to: {OUTPUT_DIR}")
    print(f"Report rows: {len(all_rows)}; images tested: {len(paths)}")
    print("A good sanity signal is a larger score drop for salient masks than for "
          "random masks across several mask sizes. This is not proof of lesion-level faithfulness.")


if __name__ == "__main__":
    main()
