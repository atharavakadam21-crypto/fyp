"""Inspect a local Keras checkpoint without changing or saving model weights."""
import argparse
import json
from pathlib import Path
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trained_plant_disease_model.keras")
    parser.add_argument("--output", default="reports/model_checkpoint_inspection.json")
    args = parser.parse_args()
    path = Path(args.model)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = tf.keras.models.load_model(path, compile=False)
    layers = []
    for layer in model.layers:
        weights = []
        for weight in layer.weights:
            weights.append({"name": weight.name, "shape": [int(x) for x in weight.shape]})
        layers.append({
            "name": layer.name,
            "class": layer.__class__.__name__,
            "input_shape": str(getattr(layer, "input_shape", None)),
            "output_shape": str(getattr(layer, "output_shape", None)),
            "config": layer.get_config(),
            "weights": weights,
        })
    report = {
        "checkpoint": str(path),
        "model_class": model.__class__.__name__,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "parameter_count": int(model.count_params()),
        "layers": layers,
        "note": "Inspection only; this script does not modify or save the checkpoint.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Model: {model.__class__.__name__}")
    print(f"Input: {model.input_shape}; output: {model.output_shape}")
    print(f"Parameters: {model.count_params():,}")
    print(f"Layer/weight details: {output}")


if __name__ == "__main__":
    main()
