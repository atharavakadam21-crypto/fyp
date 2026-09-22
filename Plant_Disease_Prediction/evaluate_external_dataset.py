import os
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

MODEL_PATH = r"trained_plant_disease_model.keras"
EXTERNAL_PATH = r"D:\FYP\External_Dataset\Test_OOD"
IMAGE_SIZE = (128, 128)

# External folder name -> exact model class name.
CLASS_MAPPING = {
    "Cherry___healthy": "Cherry_(including_sour)___healthy",
    "Corn___Cercospora_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust": "Corn_(maize)___Common_rust_",
    "Corn___Northern_Leaf_Blight": "Corn_(maize)___Northern_Leaf_Blight",
}


def get_model_classes():
    train_path = (
        r"D:\FYP\machineLearning\Plant_Disease_Prediction\"
        r"New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
    )
    return sorted(
        folder
        for folder in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, folder))
    )


def load_image(image_path):
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run this script from the folder containing the model "
            "or change MODEL_PATH to the correct absolute path."
        )

    model_classes = get_model_classes()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    y_true = []
    y_pred = []
    rows = []
    skipped = Counter()

    for external_class in sorted(os.listdir(EXTERNAL_PATH)):
        external_class_path = os.path.join(EXTERNAL_PATH, external_class)
        if not os.path.isdir(external_class_path):
            continue

        model_class = CLASS_MAPPING.get(external_class, external_class)
        if model_class not in model_classes:
            skipped[external_class] += 1
            continue

        true_index = model_classes.index(model_class)

        for filename in sorted(os.listdir(external_class_path)):
            image_path = os.path.join(external_class_path, filename)
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue

            try:
                image = load_image(image_path)
                prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
                predicted_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                y_true.append(true_index)
                y_pred.append(predicted_index)
                rows.append(
                    {
                        "file": image_path,
                        "external_class": external_class,
                        "mapped_true_class": model_class,
                        "predicted_class": model_classes[predicted_index],
                        "confidence": confidence,
                        "correct": predicted_index == true_index,
                    }
                )
            except Exception as error:
                skipped[f"{external_class}/{filename}: {error}"] += 1

    if not y_true:
        raise RuntimeError("No images were evaluated. Check EXTERNAL_PATH and MODEL_PATH.")

    labels = sorted(set(y_true) | set(y_pred))
    target_names = [model_classes[index] for index in labels]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print("\nEXTERNAL DATASET EVALUATION")
    print("=" * 60)
    print(f"Images evaluated: {len(y_true)}")
    print(f"Accuracy:         {accuracy:.4%}")
    print(f"Macro precision:  {precision:.4%}")
    print(f"Macro recall:     {recall:.4%}")
    print(f"Macro F1:         {f1:.4%}")

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

    report_dir = "external_evaluation_results"
    os.makedirs(report_dir, exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(report_dir, "predictions.csv"), index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=target_names, columns=target_names)
    matrix_df.to_csv(os.path.join(report_dir, "confusion_matrix.csv"))

    print(f"Detailed predictions saved to: {report_dir}/predictions.csv")
    print(f"Confusion matrix saved to:     {report_dir}/confusion_matrix.csv")

    if skipped:
        print("\nSKIPPED ITEMS")
        for item, count in skipped.items():
            print(f"{count}x {item}")


if __name__ == "__main__":
    main()
