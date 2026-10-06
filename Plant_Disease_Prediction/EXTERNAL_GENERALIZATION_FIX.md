# PlantVision AI — External Generalization Fix

The current reproducible baseline is:

- Internal validation: 96.86% for the improved model.
- Independent external diagnostic: about 9.22% for the improved model.
- Production external diagnostic: about 7.20%.

The goal of this experiment is **not** to make the external metric look better by training on the final test images. Instead, it creates a deterministic 70/15/15 split of the external dataset:

- 70% external adaptation training
- 15% external validation
- 15% untouched final external test

The source external dataset is not modified.

## 1. Verify class mapping

From `Plant_Disease_Prediction`:

```powershell
python check_class_mapping.py
```

Every external folder must map to one of the model's 38 output classes.

## 2. Run controlled domain adaptation

```powershell
python adapt_external_model.py
```

The script:

- starts from `trained_plant_disease_model_improved.keras`
- preserves the 38-class output
- freezes most of the CNN
- fine-tunes only the last layers at a very small learning rate
- uses augmentation only on the adaptation training split
- uses class weights to reduce class-imbalance effects
- selects the best checkpoint using external validation accuracy
- evaluates once on the untouched external test split
- never overwrites the production model

Output:

```text
external_domain_adaptation/
  split/
    external_train.csv
    external_validation.csv
    external_test.csv
  trained_plant_disease_model_external_adapted.keras
  training_history.json
  external_test_metrics.json
  external_test_confusion_matrix.csv
```

## 3. Compare models fairly

```powershell
python compare_external_models.py
```

This evaluates Production, Improved, and External-adapted models on the **same untouched external test manifest**.

## 4. Run the application with the adapted model

The application now supports model selection through environment variables.

```powershell
$env:PLANTVISION_MODEL_PATH = "external_domain_adaptation/trained_plant_disease_model_external_adapted.keras"
$env:PLANTVISION_MODEL_VERSION = "external-adapted-experiment"
streamlit run main.py
```

To return to production:

```powershell
Remove-Item Env:PLANTVISION_MODEL_PATH -ErrorAction SilentlyContinue
Remove-Item Env:PLANTVISION_MODEL_VERSION -ErrorAction SilentlyContinue
streamlit run main.py
```

## Important evaluation rule

Do **not** use the 15% final external test split for training, augmentation, threshold tuning, or repeated model selection. If the adapted model performs poorly there, investigate the dataset/domain mismatch rather than moving test images into training.

## Model files

Large `.keras` model files are intentionally not added by these source-code commits. They are generated locally by the training pipeline and should only be published to GitHub if repository storage/LFS policy is configured for them.