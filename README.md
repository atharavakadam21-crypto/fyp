# Machine Learning Projects

Welcome to the Machine Learning Projects repository! This collection showcases various machine learning projects I've worked on, encompassing a range of domains and techniques.

## Table of Contents

- [Projects](#projects)
- [Installation](#installation)

## Projects

Below is a list of the projects included in this repository:

1. **Adult Income Prediction**: Predicts whether an individual's income exceeds $50K/yr based on census data.
2. **Audi Price Prediction**: Analyzes and predicts the prices of Audi cars using a dataset of car features.
3. **Face, Eye, and Smile Detection**: Implements real-time detection of faces, eyes, and smiles using OpenCV.
4. **Fruit Recognition**: Classifies different types of fruits using image processing and machine learning techniques.
5. **Fruits and Vegetable Recognition**: Extends the fruit recognition project to include vegetables.
6. **Heart Failure Prediction**: Predicts the likelihood of heart failure based on clinical features.
7. **Human Eye Disease Prediction System**: Diagnoses common eye diseases using image data.
8. **IVT Data Visualization (NetCDF dataset)**: Visualizes Integrated Vapor Transport data from NetCDF files.
9. **Music Genre Classification**: Classifies music tracks into genres based on audio features.
10. **Object Detection with YOLOv4**: Implements object detection in images and videos using the YOLOv4 algorithm.
11. **Plant Disease Prediction**: Identifies diseases in plants from leaf images.
12. **Python for Finance**: Analyzes financial data and implements trading strategies using Python.
13. **Sentiment Analysis**: Analyzes the sentiment of text data, such as movie reviews or tweets.
14. **Stock Predictor**: Predicts stock prices using machine learning models.
15. **Weather Dataset Analysis**: Analyzes and visualizes weather data to identify patterns and trends.

Each project is contained within its own directory and includes relevant code, datasets, and documentation.

## Installation

To run any of these projects locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/animesh1012/machineLearning.git
   cd machineLearning


## Plant Disease Prediction — Model promotion process

The PlantVision model is **not promoted automatically**. Keep `trained_plant_disease_model.keras` as the protected production checkpoint and save experiments only to `trained_plant_disease_model_improved.keras` (or another candidate path).

From `Plant_Disease_Prediction/`, run the preprocessing regression test, verify the class mapping, evaluate production and candidate with the same pipeline, and run the promotion gate:

```bash
python -m pytest tests/test_preprocess.py -q
python scripts/verify_class_mapping.py --model trained_plant_disease_model.keras
python scripts/verify_class_mapping.py --model trained_plant_disease_model_improved.keras
python scripts/promotion_gate.py
```

The gate returns PASS only if preprocessing tests and both class-mapping checks pass, candidate validation accuracy does not regress, candidate external mapped-class accuracy is not lower, and candidate held-out validation ECE is no worse than production. Missing datasets, invalid mappings, or missing metrics are failures. The gate is read-only and never replaces the production model. External label mapping must be reviewed against the actual external folder names before relying on mapped-class scores.
