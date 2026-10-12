# Mixed-domain Plant Disease Model

This experiment adds a separate 38-class EfficientNetV2B0 candidate trained with
both the existing clean training set and mapped PlantDoc training images. It does
not replace or modify the production model file.

## Files

- prepare_mixed_domain_data.py: reads the local original training directory and PlantDoc ZIP, excludes ambiguous PlantDoc labels, filters exact and perceptual duplicates, and writes manifests.
- train_mixed_domain_model.py: trains a separate ImageNet-pretrained EfficientNetV2B0 candidate with source-balanced sampling, realistic augmentation, label smoothing, and field-only validation.
- evaluate_mixed_domain_model.py: evaluates a candidate on a labeled CSV manifest and writes per-class metrics, confusion matrix, calibration bins, and summary JSON.

## Before running

1. Pull this branch and open the project folder Plant_Disease_Prediction.
2. Activate the existing Python 3.10 environment with TensorFlow 2.15.
3. Ensure the original dataset is at:

   New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train

4. Ensure the downloaded archive exists at %USERPROFILE%\Downloads\PlantDoc-Dataset-master.zip, or pass its path explicitly.
5. Keep the final external test set outside all training and validation folders.

The preparation script needs Pillow and TensorFlow is needed for training/evaluation.
EfficientNetV2 ImageNet weights may download on the first training run.

## 1. Prepare data

From the project root in PowerShell:

    python prepare_mixed_domain_data.py

If the ZIP is in another location:

    python prepare_mixed_domain_data.py --plantdoc-zip "D:\Datasets\PlantDoc-Dataset-master.zip"

Output is written to data\mixed_domain_v1\. Inspect these files before training:

- class_counts.csv: original, PlantDoc train, and field-validation counts per class.
- label_mapping_audit.csv: excluded labels, corrupt images, and duplicates.
- field_validation_manifest.csv: mapped PlantDoc validation images only.
- train_manifest.csv: original train images plus mapped PlantDoc train images.

The script deliberately excludes generic labels (for example, Cherry leaf or
Tomato leaf) because they do not establish a healthy label. It only reads
PlantDoc paths under train/; it does not use PlantDoc test images.

If you want a quicker preparation pass without perceptual deduplication:

    python prepare_mixed_domain_data.py --skip-perceptual-dedupe

## 2. Train the candidate

    python train_mixed_domain_model.py

Optional shorter run for a smoke test:

    python train_mixed_domain_model.py --head-epochs 1 --epochs 1 --batch-size 16

Default candidate:
trained_plant_disease_model_mixed_domain.keras

Other artifacts include the best checkpoint, history CSV, and metadata JSON.
A small field dataset can overfit; inspect validation loss and compare against the
frozen external test only after model selection is complete.

## 3. Evaluate

The evaluation manifest must contain path,label columns, with canonical labels
matching class_names.txt. For a prepared, genuinely held-out manifest:

    python evaluate_mixed_domain_model.py --model trained_plant_disease_model_mixed_domain.keras --manifest PATH_TO_HELD_OUT_MANIFEST.csv --class-names data/mixed_domain_v1/class_names.txt

Do not use the field-validation set as the final test set. Do not report an
unverified label mapping as a definitive accuracy benchmark.

## Safety notes

- No script replaces the production model.
- Source datasets and the PlantDoc ZIP are read-only.
- Generated data and model artifacts should remain local and should not be committed.
- The existing app expects a 128x128 model; this candidate uses 224x224 input.
  It must not be connected to the current app without explicitly updating the
  inference preprocessing and testing that integration.
- This is an experiment, not a claim that field accuracy will reach a particular target.
