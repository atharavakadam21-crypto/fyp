import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

MODEL_PATH = "trained_plant_disease_model.keras"
CANDIDATE_PATH = "trained_plant_disease_model_improved.keras"

DATA_ROOT = os.path.join(
    "New Plant Diseases Dataset(Augmented)",
    "New Plant Diseases Dataset(Augmented)"
)

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VALID_DIR = os.path.join(DATA_ROOT, "valid")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Current model not found: {MODEL_PATH}")

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

if not os.path.isdir(VALID_DIR):
    raise FileNotFoundError(f"Validation directory not found: {VALID_DIR}")

print("\nLoading current production model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Current model loaded.")
print(f"Parameters: {model.count_params():,}")

# The current application feeds raw float32 pixel values (0-255).
# Do not add rescale=1./255 unless the model is retrained with it.
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.12,
    shear_range=0.08,
    horizontal_flip=True,
    fill_mode="nearest",
)

valid_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

print(f"\nTraining images: {train_generator.samples:,}")
print(f"Validation images: {valid_generator.samples:,}")
print(f"Classes: {train_generator.num_classes}")

if train_generator.num_classes != 38:
    raise ValueError(
        f"Expected 38 classes, found {train_generator.num_classes}"
    )

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    ModelCheckpoint(
        CANDIDATE_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=1,
    ),
]

print("\n============================================================")
print("STARTING CONTROLLED FINE-TUNING")
print("Current production model will NOT be overwritten.")
print(f"Candidate output: {CANDIDATE_PATH}")
print("============================================================\n")

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
)

history_path = "training_hist_improved_experiment.json"
with open(history_path, "w", encoding="utf-8") as file:
    json.dump(history.history, file, indent=2)

print("\n============================================================")
print("EXPERIMENT COMPLETE")
print("============================================================")
print(f"Candidate model: {CANDIDATE_PATH}")
print(f"Experiment history: {history_path}")

best_val = max(history.history.get("val_accuracy", [0]))
best_train = max(history.history.get("accuracy", [0]))

print(f"Best training accuracy:   {best_train * 100:.2f}%")
print(f"Best validation accuracy: {best_val * 100:.2f}%")
