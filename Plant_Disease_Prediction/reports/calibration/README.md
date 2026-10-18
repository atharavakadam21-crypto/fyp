# Calibration run status

**Pending local execution.** The script `calibration.py` deterministically splits each original validation class in half, fits a positive temperature on one half, and reports metrics on the other half. It does not use the external test set and does not modify the checkpoint.

Run:
```bash
python calibration.py --model trained_plant_disease_model_improved.keras --validation "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid" --mapping class_names.json --output reports/calibration
```

The run generates ECE before/after, accuracy before/after, a reliability diagram, correct-vs-wrong confidence histogram, and accuracy-vs-coverage CSV for max-softmax and predictive-entropy rejection. No metric is claimed until the run completes locally.
