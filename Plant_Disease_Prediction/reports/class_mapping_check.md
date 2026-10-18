# Class mapping verification

**Status: NOT RUN in the remote editing environment.**

The canonical `class_names.json` was generated from the existing 38-name list in the application. The original notebook uses TensorFlow `image_dataset_from_directory` with `class_names=None`, which infers alphabetical folder order. The verification script independently infers the validation folder order and fails loudly if it differs from the JSON mapping.

The private/local training dataset and model checkpoint are not available to execute in this GitHub editing session, so no validation accuracy or per-class scores are claimed here. Run from `Plant_Disease_Prediction/`:

```bash
python scripts/verify_class_mapping.py --model trained_plant_disease_model_improved.keras --dataset "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid" --mapping class_names.json --output reports/class_mapping_check.md
```

The script writes overall and per-class accuracy and flags classes with no samples, below 50% accuracy, or at least 30 percentage points below overall accuracy. These are review heuristics, not proof of an index shift.
