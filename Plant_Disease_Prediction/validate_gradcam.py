"""Validate PlantVision Grad-CAM with visual outputs and a deletion sanity check.

Run from this directory:
    python validate_gradcam.py
    python validate_gradcam.py test_images/test/TomatoEarlyBlight1.JPG

This is a diagnostic, not a scientific proof of explanation faithfulness.
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
IMAGE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "test_images" / "test" / "TomatoEarlyBlight1.JPG"
)
IMAGE_PATH = IMAGE_PATH if IMAGE_PATH.is_absolute() else ROOT / IMAGE_PATH


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


def gradcam(model, image, class_index):
    target = model.get_layer("conv2d_7")
    grad_model = tf.keras.Model(model.inputs, [target.output, model.output])
    tensor = tf.convert_to_tensor(preprocess(image))
    with tf.GradientTape() as tape:
        activations, predictions = grad_model(tensor, training=False)
        score = predictions[:, class_index]
    gradients = tape.gradient(score, activations)
    if gradients is None:
        raise RuntimeError("Gradient is None; selected feature layer is disconnected.")
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heat = tf.reduce_sum(
        activations[0] * weights[0][tf.newaxis, tf.newaxis, :], axis=-1
    )
    heat = tf.nn.relu(heat)
    maximum = tf.reduce_max(heat)
    heat = tf.math.divide_no_nan(heat, maximum).numpy()
    if not np.isfinite(heat).all() or float(heat.max()) <= 0:
        raise RuntimeError("Grad-CAM map is empty or non-finite.")
    heat_img = Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255)).resize(
        image.size, Image.Resampling.BILINEAR
    )
    return np.asarray(heat_img, dtype=np.float32) / 255.0


def score_for(model, image, class_index):
    return float(model(preprocess(image), training=False)[0, class_index].numpy())


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not MAIN_PATH.is_file():
        raise FileNotFoundError(f"main.py not found: {MAIN_PATH}")
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}\n"
            "Pass an existing image path as the first command-line argument."
        )

    classes = get_class_names()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    image = Image.open(IMAGE_PATH).convert("RGB")
    probabilities = model.predict(preprocess(image), verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    class_index = int(top_indices[0])
    heat = gradcam(model, image, class_index)

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = IMAGE_PATH.stem
    original = image.convert("RGB")
    heat_img = Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255)).resize(image.size)
    colored = Image.fromarray(
        np.uint8(tf.keras.utils.array_to_img(heat_img).convert("RGB"))
    )
    # Use matplotlib's jet map to match the app's overlay style.
    import matplotlib.cm as cm
    heat_values = np.asarray(heat_img, dtype=np.float32) / 255.0
    colored = Image.fromarray(
        np.uint8(np.clip(cm.get_cmap("jet")(heat_values)[..., :3], 0, 1) * 255)
    ).convert("RGB")
    overlay = Image.blend(original, colored, alpha=0.35)
    original.save(OUTPUT_DIR / f"{stem}_original.png")
    Image.fromarray(np.uint8(heat_values * 255)).save(
        OUTPUT_DIR / f"{stem}_heatmap.png"
    )
    overlay.save(OUTPUT_DIR / f"{stem}_overlay.png")

    baseline = score_for(model, image, class_index)
    # Replace the top 20% heatmap pixels with the image's per-channel mean.
    # Compare with 10 random masks of equal area as a rough sanity check.
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    resized_heat = np.asarray(
        Image.fromarray(np.uint8(heat * 255)).resize(image.size),
        dtype=np.float32,
    ) / 255.0
    count = max(1, int(0.20 * resized_heat.size))
    flat_order = np.argsort(resized_heat.ravel())
    salient = flat_order[-count:]
    mean_color = rgb.reshape(-1, 3).mean(axis=0)
    masked = rgb.reshape(-1, 3).copy()
    masked[salient] = mean_color
    salient_image = Image.fromarray(np.uint8(np.clip(masked.reshape(rgb.shape), 0, 255)))
    salient_score = score_for(model, salient_image, class_index)

    rng = np.random.default_rng(20261027)
    random_scores = []
    for _ in range(10):
        selected = rng.choice(resized_heat.size, size=count, replace=False)
        random_pixels = rgb.reshape(-1, 3).copy()
        random_pixels[selected] = mean_color
        random_image = Image.fromarray(
            np.uint8(np.clip(random_pixels.reshape(rgb.shape), 0, 255))
        )
        random_scores.append(score_for(model, random_image, class_index))

    random_mean = float(np.mean(random_scores))
    row = {
        "image": str(IMAGE_PATH),
        "predicted_class": classes[class_index],
        "confidence": baseline,
        "target_layer": "conv2d_7",
        "feature_map_shape": str(tuple(model.get_layer("conv2d_7").output.shape[1:])),
        "heatmap_min": float(heat.min()),
        "heatmap_max": float(heat.max()),
        "heatmap_std": float(heat.std()),
        "confidence_after_salient_mask": salient_score,
        "confidence_after_random_mask_mean": random_mean,
        "salient_confidence_drop": baseline - salient_score,
        "random_confidence_drop_mean": baseline - random_mean,
        "salient_drop_exceeds_random_mean": (baseline - salient_score) > (baseline - random_mean),
    }
    report_path = OUTPUT_DIR / f"{stem}_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print("\nTOP PREDICTIONS")
    for idx in top_indices:
        print(f"  {classes[int(idx)]}: {float(probabilities[int(idx)]):.4f}")
    print("\nGRAD-CAM CHECK")
    print(f"  Layer: {row['target_layer']} / feature map {row['feature_map_shape']}")
    print(f"  Heatmap range: {row['heatmap_min']:.4f} to {row['heatmap_max']:.4f}")
    print(f"  Heatmap standard deviation: {row['heatmap_std']:.4f}")
    print(f"  Original class score: {baseline:.4f}")
    print(f"  Score after salient 20% mask: {salient_score:.4f}")
    print(f"  Score after random 20% mask (mean of 10): {random_mean:.4f}")
    print("  Salient mask causes a larger drop than random mean:",
          row["salient_drop_exceeds_random_mean"])
    print(f"\nSaved images and report to: {OUTPUT_DIR}")
    print("Interpretation: one deletion test is only a sanity check, not proof that "
          "the heatmap identifies disease lesions.")


if __name__ == "__main__":
    main()
