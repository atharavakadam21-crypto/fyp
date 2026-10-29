"""
Convert PlantVision's existing .keras model to a legacy HDF5 .h5 model.

Run this with the TensorFlow/Keras environment in which the current .keras
model loads successfully. This does not retrain or simulate predictions.
It validates that the exported model reloads and produces equivalent outputs.
"""
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "trained_plant_disease_model.keras"
DESTINATION = ROOT / "trained_plant_disease_model.h5"

if not SOURCE.is_file():
    raise FileNotFoundError(f"Source model not found: {SOURCE}")

print("TensorFlow:", tf.__version__)
print("Keras:", getattr(tf.keras, "__version__", "unknown"))
print("Loading source model:", SOURCE)
model = tf.keras.models.load_model(SOURCE, compile=False)

if tuple(model.input_shape[1:]) != (128, 128, 3):
    raise ValueError(f"Unexpected input shape: {model.input_shape}")
if model.output_shape[-1] != 38:
    raise ValueError(f"Expected 38 outputs, got: {model.output_shape}")

# Deterministic smoke test before conversion.
sample = np.zeros((1, 128, 128, 3), dtype=np.float32)
before = np.asarray(model(sample, training=False))

print("Saving HDF5 model:", DESTINATION)
model.save(str(DESTINATION), include_optimizer=False, save_format="h5")

print("Reloading exported model for validation...")
reloaded = tf.keras.models.load_model(DESTINATION, compile=False)
after = np.asarray(reloaded(sample, training=False))

if tuple(reloaded.input_shape[1:]) != (128, 128, 3):
    raise RuntimeError(f"Exported input shape changed: {reloaded.input_shape}")
if reloaded.output_shape[-1] != 38:
    raise RuntimeError(f"Exported output shape changed: {reloaded.output_shape}")
if before.shape != after.shape or not np.allclose(before, after, rtol=1e-5, atol=1e-6):
    raise RuntimeError("Converted model output does not match source model output.")

print("SUCCESS: HDF5 model reloads and matches the source model output.")
print(f"Output file: {DESTINATION}")
print(f"File size: {DESTINATION.stat().st_size:,} bytes")
