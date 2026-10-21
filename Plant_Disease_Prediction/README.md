# PlantVision AI

PlantVision AI is a TensorFlow/Keras plant-leaf image classifier with a Streamlit interface, Grad-CAM visualization, top-three predictions, prediction history, and human feedback features.

## Current implementation

- 38 plant/disease classes.
- 128 x 128 RGB model input.
- Existing production checkpoint: \`trained_plant_disease_model.keras\`.
- Separate candidate checkpoint: \`trained_plant_disease_model_improved.keras\`.
- Streamlit interface: \`main.py\`.
- Reproducible model comparison: \`evaluate_models.py\`.

**Safety rule:** Do not overwrite the production checkpoint during experiments. Keep candidate and external-adaptation checkpoints separate until they pass documented evaluation.

## Data and model files

The dataset and Keras checkpoints are local assets and are intentionally not committed by this repository configuration. Put the existing dataset and model files in the project directory as expected by the scripts, or supply the validation directory explicitly to the evaluation script.

Default dataset layout:

\`\`\`text
Plant_Disease_Prediction/
├── New Plant Diseases Dataset(Augmented)/
│   └── New Plant Diseases Dataset(Augmented)/
│       ├── train/
│       └── valid/
├── trained_plant_disease_model.keras
├── trained_plant_disease_model_improved.keras
├── main.py
└── evaluate_models.py
\`\`\`

The class-folder names are used to create the evaluation labels. Before interpreting metrics, verify that the original model output indices match the folder class order used during training.

## Environment

Use Python 3.10.x. The existing project environment was developed with TensorFlow 2.15.0.

Create and activate a virtual environment if needed:

\`\`\`powershell
py -3.10 -m venv .venv-model
.\\.venv-model\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

If \`.venv-model\` already exists and works, keep using it instead of recreating it.

## Fair model evaluation

Run from this project directory:

\`\`\`powershell
python evaluate_models.py
\`\`\`

If your validation data lives elsewhere:

\`\`\`powershell
python evaluate_models.py --data-dir "D:\\path\\to\\valid"
\`\`\`

The script evaluates every supplied checkpoint on the same ordered validation images and exports:

- \`reports/model_comparison/model_comparison.csv\`
- \`reports/model_comparison/model_comparison.json\`
- Per-class precision/recall/F1 CSVs
- Confusion-matrix CSVs

The current validation folder is **not automatically an independent test set**. If it was used to train, fine-tune, select checkpoints, or tune thresholds, label its metrics as validation metrics. Use a separate untouched test split for a final generalization claim.

The evaluation intentionally uses RGB float32 pixels without dividing by 255 to match the current application preprocessing. Do not change this unless model provenance confirms that normalization is required.

## Run the web application

From this project directory, with the production model present:

\`\`\`powershell
streamlit run main.py
\`\`\`

To test a different checkpoint without replacing production:

\`\`\`powershell
$env:PLANTVISION_MODEL_PATH = "trained_plant_disease_model_improved.keras"
$env:PLANTVISION_MODEL_VERSION = "candidate"
streamlit run main.py
\`\`\`

Remove the overrides to return to the default model:

\`\`\`powershell
Remove-Item Env:PLANTVISION_MODEL_PATH -ErrorAction SilentlyContinue
Remove-Item Env:PLANTVISION_MODEL_VERSION -ErrorAction SilentlyContinue
\`\`\`

## Known limitations to address

- A high softmax score is not proof that an image belongs to one of the supported classes.
- Grad-CAM is a visualization aid, not a clinically or agronomically validated disease localization map.
- Generic disease advice must not be presented as verified treatment guidance without a reliable source.
- External datasets may use different labels, crops, image conditions, or class definitions. Evaluate only labels that can be mapped reliably.
- Model metrics have not been run by this repository update because the local model files and dataset are not included here.

## Research reporting

Report accuracy, macro precision, macro recall, macro-F1, per-class metrics, confusion matrix, dataset split, preprocessing, model checkpoint identity, and limitations. Do not call validation performance independent test accuracy.
