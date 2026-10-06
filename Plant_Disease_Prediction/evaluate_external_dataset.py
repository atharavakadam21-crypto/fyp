import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

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

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} outputs, got {model.output_shape[-1]}"
        )

    images = []
    y_true = []
    rows = []
    skipped = Counter()

    for folder in sorted(EXTERNAL_PATH.iterdir()):
        if not folder.is_dir():
            continue

        model_class = EXTERNAL_TO_MODEL.get(folder.name, folder.name)

        if model_class not in CLASS_TO_INDEX:
            skipped[folder.name] += 1
            continue

        true_index = CLASS_TO_INDEX[model_class]

        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                images.append(load_image(image_path))
                y_true.append(true_index)
                rows.append(
                    {
                        "file": str(image_path),
                        "external_class": folder.name,
                        "mapped_true_class": model_class,
                    }
                )
            except Exception as error:
                skipped[f"{folder.name}/{image_path.name}: {error}"] += 1

    if not images:
        raise RuntimeError("No images were evaluated.")

    x = np.stack(images)
    probabilities = model.predict(
        x,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    y_pred = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("\nEXTERNAL DATASET EVALUATION")
    print("=" * 60)
    print(f"Model:            {MODEL_PATH}")
    print(f"Images evaluated: {len(y_true):,}")
    print(f"Accuracy:         {accuracy:.4%}")
    print(f"Macro precision:  {precision:.4%}")
    print(f"Macro recall:     {recall:.4%}")
    print(f"Macro F1:         {f1:.4%}")
    print(f"Mean confidence:  {confidence.mean():.4%}")

    labels = sorted(set(y_true) | set(y_pred))
    target_names = [CLASS_NAMES[index] for index in labels]

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )
    )

    report_dir = PROJECT_ROOT / "external_evaluation_results"
    report_dir.mkdir(parents=True, exist_ok=True)

    for row, pred, conf in zip(rows, y_pred, confidence):
        row["predicted_class"] = CLASS_NAMES[int(pred)]
        row["confidence"] = float(conf)
        row["correct"] = bool(pred == row.get("true_index", -1))

    pd.DataFrame(rows).to_csv(
        report_dir / "predictions.csv",
        index=False,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    pd.DataFrame(
        matrix,
        index=target_names,
        columns=target_names,
    ).to_csv(report_dir / "confusion_matrix.csv")

    print(f"\nDetailed predictions: {report_dir / 'predictions.csv'}")
    print(f"Confusion matrix:     {report_dir / 'confusion_matrix.csv'}")

    if skipped:
        print("\nSKIPPED ITEMS")
        for item, count in skipped.items():
            print(f"{count}x {item}")


if __name__ == "__main__":
    main()
