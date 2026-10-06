import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATHS = {
    "Production": "trained_plant_disease_model.keras",
    "Improved": "trained_plant_disease_model_improved.keras",
}

DATA_ROOT = os.path.join(
    "New Plant Diseases Dataset(Augmented)",
    "New Plant Diseases Dataset(Augmented)",
)
VALID_DIR = os.path.join(DATA_ROOT, "valid")
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

if not os.path.isdir(VALID_DIR):
    raise FileNotFoundError(f"Validation directory not found: {VALID_DIR}")

valid_generator = ImageDataGenerator().flow_from_directory(
    VALID_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

print("\n========================================")
print("INTERNAL MODEL COMPARISON")
print("========================================")
print(f"Validation images: {valid_generator.samples:,}")
print(f"Classes: {valid_generator.num_classes}")

for model_name, model_path in MODEL_PATHS.items():
    if not os.path.exists(model_path):
        print(f"\nSkipping {model_name}: file not found")
        continue

    print(f"\nLoading {model_name} model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    valid_generator.reset()
    loss, accuracy = model.evaluate(valid_generator, verbose=1)

    print(f"\n{model_name} Results")
    print("-------------------------")
    print(f"Loss:     {loss:.6f}")
    print(f"Accuracy: {accuracy * 100:.4f}%")

print("\n========================================")
print("COMPARISON COMPLETE")
print("========================================")
