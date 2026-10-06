import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv(
        "PLANTVISION_MODEL_PATH",
        str(PROJECT_ROOT / "trained_plant_disease_model_improved.keras"),
    )
)
EXTERNAL_PATH = Path(
    os.getenv(
        "EXTERNAL_DATASET",
        r"D:\FYP\External_Dataset\Test_OOD",
    )
)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

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


def load_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        return np.asarray(image, dtype=np.float32)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not EXTERNAL_PATH.is_dir():
        raise FileNotFoundError(f"External dataset not found: {EXTERNAL_PATH}")

    print("\nLoading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} outputs, got {model.output_shape[-1]}"
        )

    images = []
    rows = []

    for folder in sorted(EXTERNAL_PATH.iterdir()):
        if not folder.is_dir():
            continue

        model_class = EXTERNAL_TO_MODEL.get(folder.name, folder.name)
        if model_class not in CLASS_TO_INDEX:
            print(f"Skipping unmapped class: {folder.name}")
            continue

        true_index = CLASS_TO_INDEX[model_class]

        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                images.append(load_image(image_path))
                rows.append(
                    {
                        "file": str(image_path),
                        "external_class": folder.name,
                        "mapped_true_class": model_class,
                        "true_index": true_index,
                    }
                )
            except Exception as error:
                print(f"Skipping {image_path}: {error}")

    if not images:
        raise RuntimeError("No images were loaded.")

    x = np.stack(images)
    print(f"Images loaded: {len(x):,}")
    print(f"Input shape: {x.shape}")

    print("\nRunning predictions...")
    probabilities = model.predict(x, batch_size=BATCH_SIZE, verbose=1)
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    true = np.array([row["true_index"] for row in rows])
    correct = predicted == true

    for row, pred, conf, is_correct in zip(rows, predicted, confidence, correct):
        row["predicted_class"] = CLASS_NAMES[int(pred)]
        row["confidence"] = float(conf)
        row["correct"] = bool(is_correct)

    diagnostics_dir = PROJECT_ROOT / "results" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter(predicted.tolist())

    print("\n" + "=" * 90)
    print("PREDICTION DISTRIBUTION")
    print("=" * 90)

    for index, count in counts.most_common():
        print(
            f"{index:02d} | {CLASS_NAMES[index]:60s} | "
            f"{count:4d} images ({count / len(predicted) * 100:6.2f}%)"
        )

    print("\n" + "=" * 90)
    print("CONFIDENCE VS CORRECTNESS")
    print("=" * 90)

    accuracy = float(np.mean(correct))
    print(f"Valid samples       : {len(correct):,}")
    print(f"Correct predictions : {int(correct.sum()):,}")
    print(f"Wrong predictions   : {int((~correct).sum()):,}")
    print(f"Accuracy            : {accuracy * 100:.4f}%")

    print(
        f"\nMean confidence when CORRECT: "
        f"{confidence[correct].mean() * 100:.2f}%"
        if correct.any()
        else "\nMean confidence when CORRECT: N/A"
    )
    print(
        f"Mean confidence when WRONG: "
        f"{confidence[~correct].mean() * 100:.2f}%"
        if (~correct).any()
        else "Mean confidence when WRONG: N/A"
    )

    report = pd.DataFrame(rows)
    report.to_csv(
        diagnostics_dir / "external_diagnostic_predictions.csv",
        index=False,
    )

    matrix = confusion_matrix(
        true,
        predicted,
        labels=list(range(len(CLASS_NAMES))),
    )
    pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        diagnostics_dir / "external_confusion_matrix_diagnostic.csv"
    )

    summary = {
        "model": str(MODEL_PATH),
        "images": int(len(correct)),
        "accuracy": accuracy,
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
    }

    pd.Series(summary).to_json(
        diagnostics_dir / "external_prediction_distribution.json",
        indent=2,
    )

    print("\nDiagnostics saved to:")
    print(diagnostics_dir)


if __name__ == "__main__":
    main()
