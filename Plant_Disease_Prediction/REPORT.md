# PlantVision AI — verified evaluation pipeline

## Problem
PlantVision had inconsistent image preprocessing, duplicated class names, a 9.14% vs 9.22% external-score discrepancy, incomplete external-label equivalence, overconfident errors, and checkpoint provenance that was not fully confirmed.

## What changed
- Added a shared TensorFlow preprocessing function: RGB, 128×128, bilinear resize, float32 raw pixels (0–255); wired it into app, evaluation, and fine-tuning.
- Added canonical 38-index `class_names.json` and deterministic class-order/per-class-accuracy verification.
- Added `evaluate.py` as the single reproducible evaluation entry point with per-image CSV, JSON metrics, confusion matrix, skipped-file accounting, and mapped-only vs all-readable accuracy.
- Marked legacy evaluation scripts deprecated; added discrepancy investigation notes rather than guessing a cause.
- Added external label map with four aliases already present in legacy code, plus validation that every non-null target exists.
- Added validation-only temperature scaling, ECE, reliability/confidence plots, and max-softmax/entropy accuracy-coverage analysis.
- Added checkpoint inspection and a conservative read-only promotion gate. Production model was not promoted or overwritten.

## Evidence and current numbers
- Original notebook documents 128×128 RGB, bilinear resize, no explicit pixel rescaling, a custom Sequential CNN, 38-way softmax, and 7,842,762 parameters.
- Previously recorded candidate validation accuracy: **96.86%**; previously recorded external accuracy: approximately **9.22%**. These figures were not rerun in this remote editing session and should be presented as prior measurements, not newly verified results.
- The historical **9.14% vs 9.22% cause remains OPEN** because the external dataset and checkpoint files were not available for execution.
- No calibration ECE or promotion-gate result is claimed until scripts run locally.

## Seven-problem status

| Problem | Status | Limitation |
|---|---|---|
| Preprocessing consistency | Partially | Shared function and tests committed; tests still need local execution. |
| Class-index mapping | Partially | JSON derives from the app's 38 labels and notebook's alphabetical-folder convention; actual checkpoint/data accuracy check is pending. |
| Reproducible evaluation / discrepancy | Partially | Unified evaluator and legacy warnings added; old scripts were not run side-by-side, so cause and byte-for-byte repeatability remain unverified. |
| External label equivalence | Partially | Four aliases are evidenced in old code; exact 30-folder inventory and disease-definition review remain pending. |
| Calibration / unknown rejection | Partially | Validation-only implementation committed; numerical metrics and plots await local run. |
| Architecture / training provenance | Partially | Notebook architecture and hyperparameters are documented; actual serialized checkpoint keys/shapes need local inspection. |
| Production protection gate | Partially | Read-only gate and README process added; gate has not run. No existing CI workflow was found by the repository search. |

## Remaining risks and next steps
1. Reconcile `label_mapping.yaml` with all 30 actual external folder names; manually review the combined corn leaf-spot label.
2. Run tests and class mapping verification with the real validation dataset and both checkpoints.
3. Run both legacy scripts with separate prediction exports to isolate the two-score discrepancy.
4. Generate calibration artifacts and run the promotion gate. Treat a FAIL as a stop signal; do not promote automatically.

## How to reproduce
Run commands from `Plant_Disease_Prediction/` after pulling this branch and confirming local dataset paths:

```bash
python -m pytest tests/test_preprocess.py -q
python scripts/verify_class_mapping.py --model trained_plant_disease_model.keras
python scripts/verify_class_mapping.py --model trained_plant_disease_model_improved.keras
python evaluate.py --model trained_plant_disease_model.keras --dataset "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid" --mapping label_mapping.yaml --output reports/validation_production
python calibration.py --model trained_plant_disease_model_improved.keras
python scripts/promotion_gate.py
```

**Safety:** `trained_plant_disease_model.keras` remains the production checkpoint. No promotion or checkpoint write is performed by these tools.
