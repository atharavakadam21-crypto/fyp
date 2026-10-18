# Model provenance and training verification

## Verified from repository source
- **Framework:** TensorFlow / Keras.
- **Architecture in the original training notebook:** Sequential custom CNN, not MobileNetV2 or ResNet50.
- **Input:** RGB, 128×128×3.
- **Input interpolation:** TensorFlow `image_dataset_from_directory(... interpolation="bilinear")`.
- **Pixel normalization:** The visible original input pipeline contains no rescaling/mean-std normalization; pixels are fed in the 0–255 range. No ImageNet mean/std normalization is documented.
- **Output:** Dense 38-unit softmax layer.
- **Notebook architecture:** convolution blocks with 32, 64, 128, 256, and 512 filters (two Conv2D layers per block), max pooling, Dropout(0.25), Flatten, Dense(1500, ReLU), Dropout(0.4), Dense(38, softmax).
- **Notebook model summary:** 7,842,762 parameters, 29.92 MB, according to the saved notebook output.
- **Original compile settings visible in notebook:** legacy Adam, learning rate 0.0001; categorical cross-entropy; accuracy metric. Saved training output shows 10 epochs.
- **Fine-tuning script:** loads `trained_plant_disease_model.keras`, uses Adam at 1e-5, categorical cross-entropy, batch size 32, 5 maximum epochs, image augmentation, ModelCheckpoint on best validation accuracy, EarlyStopping patience 2 with restore-best-weights, and ReduceLROnPlateau factor 0.5/patience 1/min LR 1e-7. It writes to the separate candidate filename.

## Not verified from checkpoint bytes
- The actual production and candidate `.keras` files are not available to inspect in the remote GitHub editing environment. Therefore their actual serialized layer keys/configs/weight shapes, checksum, output shape, and equality with the notebook architecture are **UNVERIFIED here**.
- The notebook's saved summary is evidence about the notebook's model, not by itself proof that the current checkpoint is identical.
- Exact original training split/seed, optimizer state in the checkpoint, and full history provenance are not fully established by the checked-in source.

## Read-only checkpoint inspection
Run this on the machine holding the checkpoint:
```bash
python scripts/inspect_model.py --model trained_plant_disease_model.keras --output reports/model_checkpoint_inspection.json
python scripts/inspect_model.py --model trained_plant_disease_model_improved.keras --output reports/candidate_checkpoint_inspection.json
```
This prints input/output shapes, parameter count, layer configurations, and all layer weight names/shapes. It does not save or modify model weights.
