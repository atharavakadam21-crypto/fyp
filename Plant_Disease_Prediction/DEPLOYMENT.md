# Deploying PlantVision AI

## Recommended first deployment: Streamlit Community Cloud

1. Confirm the trained model file is available as `Plant_Disease_Prediction/trained_plant_disease_model.keras`, or configure `PLANTVISION_MODEL_PATH` to its mounted/downloaded location.
2. Ensure the model file is available to the deployed app. It is currently not tracked in this GitHub folder. Do not commit it if it exceeds GitHub's file-size limit; use Git LFS or model hosting instead.
3. Open https://share.streamlit.io/ and connect the GitHub account that can access this repository.
4. Create an app using repository `atharavakadam21-crypto/fyp`, branch `main`, and main file path `Plant_Disease_Prediction/main.py`.
5. In Advanced settings, select Python 3.10 because TensorFlow 2.10 does not support newer Python runtimes such as Python 3.11.
6. If AI guidance is needed, add the following under the app's Secrets settings (use your own key; never commit it):
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
   The classifier can run without Gemini, but the Gemini guidance feature needs the key.
7. Deploy and inspect the app logs. Confirm a prediction works before sharing the URL.

## Notes

- `requirements.txt` is kept beside `main.py` so Streamlit Cloud can find the app's dependencies.
- The app resolves its default model, background image, and writable data directory relative to this folder, regardless of the deployment working directory.
- Prediction history and real-world verification CSVs are local to the running instance. On hosted free services, local files may be ephemeral and should not be treated as permanent storage.
- If the model is hosted separately, configure `PLANTVISION_MODEL_PATH` to the local file path where the model has been downloaded before app startup. The current app does not automatically download the model from a URL.
