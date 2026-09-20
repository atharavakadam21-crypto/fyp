import os
import csv
import json
import base64
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.cm as cm

from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PlantVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "trained_plant_disease_model.keras"
BACKGROUND_PATH = "home_page.jpeg"

DATA_DIR = "data"
HISTORY_FILE = os.path.join(
    DATA_DIR,
    "prediction_history.csv"
)

LEARNING_DIR = os.path.join(
    DATA_DIR,
    "learning_queue"
)

REAL_WORLD_FILE = os.path.join(
    DATA_DIR,
    "real_world_tests.csv"
)

MODEL_VERSION = "v1.0"

DEFAULT_THRESHOLD = 0.60


# =========================================================
# CREATE DIRECTORIES
# =========================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

os.makedirs(
    LEARNING_DIR,
    exist_ok=True
)


# =========================================================
# SAFE HTML RENDERER
#
# IMPORTANT:
# We deliberately flatten the HTML before passing it
# to Streamlit. This prevents indented HTML from being
# interpreted as a Markdown code block.
# =========================================================

def render_html(content):

    cleaned = " ".join(
        line.strip()
        for line in content.splitlines()
        if line.strip()
    )

    st.markdown(
        cleaned,
        unsafe_allow_html=True
    )


# =========================================================
# PREMIUM CSS
# =========================================================

def inject_css():

    background_image = ""

    if os.path.exists(BACKGROUND_PATH):

        try:

            with open(
                BACKGROUND_PATH,
                "rb"
            ) as file:

                encoded = base64.b64encode(
                    file.read()
                ).decode()

            background_image = (
                "url('data:image/jpeg;base64,"
                + encoded
                + "')"
            )

        except Exception:

            background_image = ""

    if background_image:

        app_background = f"""
        background-image:
            linear-gradient(
                rgba(2, 16, 11, 0.92),
                rgba(2, 25, 17, 0.96)
            ),
            {background_image};
        """

    else:

        app_background = """
        background:
            linear-gradient(
                135deg,
                #02130b,
                #06291b
            );
        """

    css = f"""
    <style>

    /* =====================================================
       GLOBAL
    ===================================================== */

    .stApp {{
        {app_background}

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        min-height: 100vh;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;

        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(50,255,145,0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 85%,
                rgba(40,255,130,0.07),
                transparent 30%
            );

        pointer-events: none;
        z-index: 0;
    }}

    .main .block-container {{
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    .stMarkdown,
    .stButton,
    .stFileUploader,
    .stSelectbox,
    .stTextInput,
    .stSlider,
    .stDataFrame,
    .stMetric,
    .stExpander,
    .stAlert {{
        position: relative;
        z-index: 1;
    }}

    /* =====================================================
       BRANDING
    ===================================================== */

    .brand {{
        font-size: 28px;
        font-weight: 900;
        letter-spacing: -1px;
        color: #72eaa5;
        margin-bottom: 0;
    }}

    .brand-sub {{
        color: #8bb39c;
        font-size: 12px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 25px;
    }}

    .hero {{
        padding: 34px;
        border: 1px solid rgba(114,234,165,0.16);
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            rgba(15, 58, 38, 0.72),
            rgba(3, 25, 17, 0.55)
        );
        box-shadow: 0 20px 60px rgba(0,0,0,0.22);
        margin-bottom: 25px;
    }}

    .hero-eyebrow {{
        color: #72eaa5;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 3px;
        margin-bottom: 13px;
    }}

    .hero-title {{
        color: #f1fff6;
        font-size: clamp(34px, 5vw, 65px);
        font-weight: 900;
        line-height: 1;
        letter-spacing: -2px;
    }}

    .hero-title span {{
        color: #72eaa5;
    }}

    .hero-description {{
        color: #a9c7b5;
        font-size: 15px;
        line-height: 1.8;
        max-width: 750px;
        margin-top: 20px;
    }}

    .pill {{
        display: inline-block;
        padding: 7px 12px;
        margin: 4px 5px 0 0;
        border-radius: 999px;
        background: rgba(114,234,165,0.10);
        border: 1px solid rgba(114,234,165,0.22);
        color: #a9f5c6;
        font-size: 11px;
        font-weight: 800;
    }}

    .metric-card {{
        background: rgba(9, 40, 27, 0.72);
        border: 1px solid rgba(114,234,165,0.15);
        border-radius: 18px;
        padding: 20px;
        min-height: 135px;
    }}

    .metric-icon {{
        font-size: 24px;
        margin-bottom: 10px;
    }}

    .metric-number {{
        color: #eafff1;
        font-size: 25px;
        font-weight: 900;
    }}

    .metric-label {{
        color: #789b88;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-top: 5px;
    }}

    .glass-card,
    .result-card,
    .upload-card {{
        background: rgba(8, 36, 24, 0.72);
        border: 1px solid rgba(114,234,165,0.14);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
    }}

    .result-card {{
        background: linear-gradient(
            135deg,
            rgba(18, 76, 47, 0.75),
            rgba(5, 31, 20, 0.82)
        );
    }}

    .upload-card {{
        text-align: center;
        border-style: dashed;
        padding: 35px;
    }}

    .upload-icon {{
        font-size: 38px;
        margin-bottom: 10px;
    }}

    .upload-title {{
        color: #dfffea;
        font-size: 21px;
        font-weight: 800;
    }}

    .upload-subtitle {{
        color: #789b88;
        font-size: 12px;
        line-height: 1.8;
        margin-top: 8px;
    }}

    .section-title {{
        color: #e5fff0;
        font-size: 22px;
        font-weight: 900;
        margin: 28px 0 15px;
    }}

    .info-row {{
        display: flex;
        justify-content: space-between;
        gap: 20px;
        padding: 13px 0;
        border-bottom: 1px solid rgba(114,234,165,0.09);
    }}

    .info-row:last-child {{
        border-bottom: none;
    }}

    .info-label {{
        color: #72eaa5;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 1px;
    }}

    .info-value {{
        color: #b5cebf;
        font-size: 13px;
        text-align: right;
    }}

    .status-badge {{
        display: inline-block;
        border: 1px solid rgba(114,234,165,0.3);
        background: rgba(114,234,165,0.10);
        color: #72eaa5;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 1px;
    }}

    .result-label {{
        color: #7fae91;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 2px;
    }}

    .result-plant {{
        color: #b7e8c8;
        font-size: 24px;
        font-weight: 800;
    }}

    .result-disease {{
        color: #effff4;
        font-size: 25px;
        font-weight: 900;
    }}

    .result-confidence {{
        color: #72eaa5;
        font-size: 38px;
        font-weight: 900;
    }}

    .footer,
    .sidebar-footer {{
        color: #527962;
        font-size: 11px;
        text-align: center;
        line-height: 1.8;
        padding: 25px 0;
    }}

    .sidebar-model {{
        padding: 15px;
        border: 1px solid rgba(114,234,165,0.13);
        border-radius: 14px;
        background: rgba(8,36,24,0.55);
        margin-top: 18px;
    }}

    .sidebar-model-title {{
        color: #6e987e;
        font-size: 10px;
        letter-spacing: 2px;
        font-weight: 900;
    }}

    .sidebar-model-value {{
        color: #b9f4cc;
        font-size: 13px;
        font-weight: 800;
        margin-top: 5px;
    }}

    </style>
    """

    st.markdown(
        css,
        unsafe_allow_html=True
    )


inject_css()


# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# =========================================================
# MODEL
# =========================================================

@st.cache_resource

def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


model = None

try:

    model = load_model()

except Exception as error:

    st.error(
        "Model could not be loaded. Ensure trained_plant_disease_model.keras "
        f"exists beside main.py. Details: {error}"
    )


# =========================================================
# HELPERS
# =========================================================

def split_label(label):

    parts = label.split("___", 1)

    if len(parts) == 2:
        return parts[0], parts[1]

    return label, "Unknown"


def pretty_name(label):

    plant, disease = split_label(label)

    plant = plant.replace("_", " ")
    disease = disease.replace("_", " ")

    return f"{plant} — {disease}"


def validate_image(image):

    if image is None:
        return False, "No image was provided."

    if image.width < 32 or image.height < 32:
        return False, "Please upload an image at least 32 × 32 pixels."

    if image.width > 6000 or image.height > 6000:
        return False, "Please upload an image smaller than 6000 × 6000 pixels."

    return True, "Image validated successfully."


def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((128, 128))

    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    return array


def predict_image(image):

    if model is None:
        raise RuntimeError("The trained model is not available.")

    input_array = preprocess_image(image)
    probabilities = model.predict(
        input_array,
        verbose=0
    )[0]

    indices = np.argsort(probabilities)[::-1][:3]

    return probabilities, indices


def save_history(label, confidence, status, feedback="", image_path=""):

    os.makedirs(DATA_DIR, exist_ok=True)

    file_exists = os.path.exists(HISTORY_FILE)

    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "prediction",
                "confidence_percent",
                "status",
                "feedback",
                "image_path"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            label,
            round(float(confidence), 2),
            status,
            feedback,
            image_path
        ])


def save_real_world_test(
    image_name,
    actual_class,
    predicted_class,
    confidence,
    correct
):

    file_exists = os.path.exists(REAL_WORLD_FILE)

    with open(
        REAL_WORLD_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "image_name",
                "actual_class",
                "predicted_class",
                "confidence",
                "correct"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            image_name,
            actual_class,
            predicted_class,
            round(float(confidence), 4),
            correct
        ])


def disease_information(label):

    plant, disease = split_label(label)

    if disease.lower() == "healthy":
        return {
            "description": f"The model classified this {plant} leaf as healthy.",
            "symptoms": "No disease pattern was identified by the classifier.",
            "management": "Continue appropriate crop care and monitor the plant regularly."
        }

    return {
        "description": f"The model classified this {plant} leaf as {disease}.",
        "symptoms": "Symptoms vary by crop and disease.",
        "management": "Use this result as an aid and seek expert confirmation when required."
    }


# =========================================================
# DASHBOARD
# =========================================================

def dashboard():

    render_html(
        """
        <div class="hero">
        <div class="hero-eyebrow">AI POWERED PLANT DISEASE DETECTION</div>
        <div class="hero-title">Plant<span>Vision AI</span></div>
        <div class="hero-description">
        Detect · Understand · Protect
        <br><br>
        A computer-vision system designed to analyze plant
        leaf images and assist with early disease identification.
        </div>
        <div style="margin-top:24px;">
        <span class="pill">CNN</span>
        <span class="pill">38 Classes</span>
        <span class="pill">Grad-CAM</span>
        <span class="pill">Real-World Testing</span>
        <span class="pill">Human Feedback</span>
        </div>
        </div>
        """
    )

    columns = st.columns(4)

    metrics = [
        ("🌿", "38", "SUPPORTED CLASSES"),
        ("🧠", "7.84M", "MODEL PARAMETERS"),
        ("🖼️", "128×128", "INPUT RESOLUTION"),
        ("⚡", "CNN", "ARCHITECTURE")
    ]

    for column, data in zip(columns, metrics):

        icon, number, label = data

        with column:
            render_html(
                f"""
                <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-number">{number}</div>
                <div class="metric-label">{label}</div>
                </div>
                """
            )

    render_html(
        """
        <div class="section-title">
        How PlantVision AI Works
        </div>

        <div class="glass-card">

        <div class="info-row">
        <div class="info-label">01 — Capture</div>
        <div class="info-value">
        Upload a real-world plant leaf photograph.
        </div>
        </div>

        <div class="info-row">
        <div class="info-label">02 — Preprocess</div>
        <div class="info-value">
        Convert the image to RGB and resize it to 128×128.
        </div>
        </div>

        <div class="info-row">
        <div class="info-label">03 — Analyze</div>
        <div class="info-value">
        The CNN evaluates visual patterns across 38 trained classes.
        </div>
        </div>

        <div class="info-row">
        <div class="info-label">04 — Explain</div>
        <div class="info-value">
        Grad-CAM visualizes regions contributing to the prediction.
        </div>
        </div>

        <div class="info-row">
        <div class="info-label">05 — Learn</div>
        <div class="info-value">
        Human-confirmed samples can be collected for controlled future retraining.
        </div>
        </div>

        </div>
        """
    )

    render_html(
        """
        <div class="section-title">
        🌾 Supported Crops
        </div>
        """
    )

    crops = sorted(
        set(
            split_label(item)[0]
            for item in CLASS_NAMES
        )
    )

    render_html(
        f"""
        <div class="glass-card">
        <div style="color:#c3dbce;font-size:14px;line-height:2;">
        {" &nbsp; • &nbsp; ".join(crops)}
        </div>
        </div>

        <div class="footer">
        PlantVision AI · Final Year Project
        <br>
        Detect · Understand · Protect
        </div>
        """
    )


# =========================================================
# GRAD-CAM
# =========================================================

def generate_gradcam(image, class_index):

    if model is None:
        raise RuntimeError("The trained model is not available.")

    convolution_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]

    if not convolution_layers:
        raise RuntimeError("No convolutional layer was found for Grad-CAM.")

    last_conv_layer = convolution_layers[-1]

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    input_array = preprocess_image(image)
    input_tensor = tf.convert_to_tensor(input_array)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(input_tensor)
        class_channel = predictions[:, class_index]

    gradients = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)

    if float(maximum) > 0:
        heatmap /= maximum

    heatmap = heatmap.numpy()

    heatmap = np.uint8(255 * heatmap)
    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(np.arange(256))[:, :3]
    colored_heatmap = colored_heatmap[heatmap]
    colored_heatmap = np.uint8(colored_heatmap * 255)

    colored_heatmap = Image.fromarray(colored_heatmap).resize(
        image.size
    )

    original = image.convert("RGB")
    overlay = Image.blend(
        original,
        colored_heatmap.convert("RGB"),
        alpha=0.45
    )

    return overlay


# =========================================================
# DISEASE RECOGNITION
# =========================================================

def disease_recognition():

    render_html(
        """
        <div class="hero">
        <div class="hero-eyebrow">
        COMPUTER VISION ANALYSIS
        </div>

        <div class="hero-title">
        Disease <span>Recognition</span>
        </div>

        <div class="hero-description">
        Upload a real-world plant leaf photograph
        and let PlantVision AI analyze it.
        </div>
        </div>
        """
    )

    render_html(
        """
        <div class="upload-card">
        <div>
        <div class="upload-icon">🌿</div>
        <div class="upload-title">
        Drag & Drop Your Leaf Image
        </div>
        <div class="upload-subtitle">
        Drop your image below or click Browse Files
        <br>
        JPG · JPEG · PNG · WEBP
        </div>
        </div>
        </div>
        """
    )

    uploaded = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    threshold = st.slider(
        "Confidence threshold",
        min_value=0.30,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.05
    )

    if uploaded is None:

        render_html(
            """
            <div class="glass-card">
            <div style="text-align:center;color:#789b88;padding:10px;">
            <span style="color:#72eaa5;font-weight:800;">TIP</span>
            <br>
            Use a clear image of a single leaf with
            good lighting for better results.
            </div>
            </div>
            """
        )

        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be read.")
        return

    valid, message = validate_image(image)

    if not valid:
        st.error(message)
        return

    st.success(message)

    image_column, info_column = st.columns([1.3, 1])

    with image_column:
        st.image(
            image,
            caption="Uploaded Leaf",
            use_container_width=True
        )

    with info_column:

        render_html(
            f"""
            <div class="glass-card">
            <div class="info-row">
            <div class="info-label">FILE</div>
            <div class="info-value">{uploaded.name}</div>
            </div>
            <div class="info-row">
            <div class="info-label">DIMENSIONS</div>
            <div class="info-value">{image.width} × {image.height}</div>
            </div>
            <div class="info-row">
            <div class="info-label">MODEL INPUT</div>
            <div class="info-value">128 × 128 RGB</div>
            </div>
            <div class="info-row">
            <div class="info-label">MODEL</div>
            <div class="info-value">CNN · {MODEL_VERSION}</div>
            </div>
            </div>
            """
        )

    st.divider()

    if st.button(
        "🔎 Analyze Leaf",
        use_container_width=True
    ):

        with st.spinner("PlantVision AI is analyzing the leaf..."):
            predictions, indices = predict_image(image)

        top_index = int(indices[0])
        confidence = float(predictions[top_index])
        label = CLASS_NAMES[top_index]

        status = (
            "KNOWN CLASS"
            if confidence >= threshold
            else "UNKNOWN / LOW CONFIDENCE"
        )

        st.session_state["prediction_result"] = {
            "label": label,
            "confidence": confidence,
            "predictions": predictions,
            "indices": indices,
            "image": image,
            "status": status
        }

        save_history(
            pretty_name(label),
            confidence * 100,
            status
        )

    if "prediction_result" not in st.session_state:
        return

    result = st.session_state["prediction_result"]

    label = result["label"]
    confidence = result["confidence"]
    predictions = result["predictions"]
    indices = result["indices"]
    status = result["status"]

    plant, disease = split_label(label)

    render_html(
        """
        <div class="section-title">AI Diagnosis</div>
        """
    )

    badge = "KNOWN CLASS" if status == "KNOWN CLASS" else "LOW CONFIDENCE"

    render_html(
        f"""
        <div class="result-card">
        <span class="status-badge">{badge}</span>
        <div class="result-label" style="margin-top:20px;">PLANT</div>
        <div class="result-plant">🌿 {plant}</div>
        <div class="result-label" style="margin-top:20px;">DETECTED CONDITION</div>
        <div class="result-disease">{disease}</div>
        <div class="result-label" style="margin-top:21px;">MODEL CONFIDENCE</div>
        <div class="result-confidence">{confidence * 100:.2f}%</div>
        </div>
        """
    )

    if status != "KNOWN CLASS":
        st.error(
            "⚠️ Low confidence: this image may not closely match one of the model's trained classes."
        )

    render_html(
        """
        <div class="section-title">📊 Prediction Analysis</div>
        """
    )

    prediction_columns = st.columns(3)

    for column, rank, index in zip(
        prediction_columns,
        [1, 2, 3],
        indices
    ):

        index = int(index)
        probability = float(predictions[index])

        with column:
            st.metric(
                f"Top {rank}",
                pretty_name(CLASS_NAMES[index]),
                f"{probability * 100:.2f}%"
            )

    render_html(
        """
        <div class="section-title">🔥 Grad-CAM Explainability</div>
        """
    )

    try:
        gradcam_image = generate_gradcam(
            result["image"],
            int(indices[0])
        )

        st.image(
            gradcam_image,
            caption="Grad-CAM — Regions influencing the prediction",
            use_container_width=True
        )

    except Exception as error:
        st.warning(f"Grad-CAM could not be generated: {error}")

    render_html(
        """
        <div class="section-title">📚 Disease Information</div>
        """
    )

    information = disease_information(label)

    with st.expander("🔍 Description", expanded=True):
        st.write(information["description"])

    with st.expander("🩺 Typical Symptoms"):
        st.write(information["symptoms"])

    with st.expander("🌱 Management Guidance"):
        st.write(information["management"])

    render_html(
        """
        <div class="section-title">👤 Human Feedback</div>
        <div class="glass-card">
        <div style="color:#b0ccbd;font-size:13px;">
        Your feedback can contribute to a verified real-world sample collection pipeline.
        </div>
        </div>
        """
    )

    feedback_left, feedback_right = st.columns(2)

    with feedback_left:

        if st.button("✅ Prediction Correct", use_container_width=True):

            save_history(
                pretty_name(label),
                confidence * 100,
                "CONFIRMED",
                "Correct"
            )

            st.success("Prediction confirmed.")

    with feedback_right:

        if st.button("❌ Prediction Incorrect", use_container_width=True):
            st.session_state["feedback_mode"] = True

    if st.session_state.get("feedback_mode", False):

        st.markdown("### Correct the Prediction")

        correct_label = st.selectbox(
            "Correct class",
            CLASS_NAMES
        )

        if st.button("💾 Save Corrected Sample", use_container_width=True):

            folder = os.path.join(
                LEARNING_DIR,
                correct_label
            )

            os.makedirs(folder, exist_ok=True)

            filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
            image_path = os.path.join(folder, filename)

            result["image"].convert("RGB").save(
                image_path,
                quality=95
            )

            save_history(
                pretty_name(label),
                confidence * 100,
                "HUMAN_CORRECTED",
                "Correct label: " + pretty_name(correct_label),
                image_path
            )

            st.success("✅ Corrected sample saved to the learning queue.")


# =========================================================
# REAL-WORLD TESTING
# =========================================================

def real_world_testing():

    render_html(
        """
        <div class="hero">
        <div class="hero-eyebrow">FIELD VALIDATION</div>
        <div class="hero-title">Real-World <span>Testing</span></div>
        <div class="hero-description">
        Test PlantVision AI on photographs outside the validation dataset
        and record human-verified results.
        </div>
        </div>
        """
    )

    render_html(
        """
        <div class="glass-card">
        <div class="info-row">
        <div class="info-label">IMPORTANT</div>
        <div class="info-value">
        Upload your own photographs. Select the actual class after viewing
        the prediction. The resulting score is a separate real-world test
        metric and is not part of the 93.31% validation accuracy.
        </div>
        </div>
        </div>
        """
    )

    uploaded = st.file_uploader(
        "Upload a real-world leaf photograph",
        type=["jpg", "jpeg", "png", "webp"],
        key="real_world_uploader"
    )

    if uploaded is None:
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("The uploaded image could not be read.")
        return

    valid, message = validate_image(image)

    if not valid:
        st.error(message)
        return

    image_col, result_col = st.columns([1.2, 1])

    with image_col:
        st.image(
            image,
            caption="Real-World Test Image",
            use_container_width=True
        )

    with result_col:

        if st.button(
            "🔎 Analyze Real-World Image",
            use_container_width=True,
            key="real_world_analyze"
        ):

            predictions, indices = predict_image(image)

            st.session_state["real_world_result"] = {
                "predictions": predictions,
                "indices": indices,
                "image_name": uploaded.name,
                "image": image
            }

    if "real_world_result" not in st.session_state:
        return

    result = st.session_state["real_world_result"]
    predictions = result["predictions"]
    indices = result["indices"]

    top_index = int(indices[0])
    confidence = float(predictions[top_index])
    predicted_class = CLASS_NAMES[top_index]

    render_html(
        f"""
        <div class="result-card">
        <div class="result-label">AI PREDICTION</div>
        <div class="result-disease">{pretty_name(predicted_class)}</div>
        <div class="result-confidence">{confidence * 100:.2f}%</div>
        </div>
        """
    )

    top_cols = st.columns(3)

    for column, rank, index in zip(top_cols, [1, 2, 3], indices):

        index = int(index)
        probability = float(predictions[index])

        with column:
            st.metric(
                f"Top {rank}",
                pretty_name(CLASS_NAMES[index]),
                f"{probability * 100:.2f}%"
            )

    actual_class = st.selectbox(
        "Human-verified actual class",
        CLASS_NAMES,
        key="real_world_actual_class"
    )

    if st.button(
        "✅ Record Verified Result",
        use_container_width=True,
        key="real_world_record"
    ):

        correct = actual_class == predicted_class

        save_real_world_test(
            result["image_name"],
            actual_class,
            predicted_class,
            confidence,
            correct
        )

        if correct:
            st.success("Correct prediction recorded in the real-world test set.")
        else:
            st.warning("Incorrect prediction recorded. This sample is useful for error analysis.")

    try:
        gradcam_image = generate_gradcam(image, top_index)

        st.image(
            gradcam_image,
            caption="Grad-CAM — Real-World Test",
            use_container_width=True
        )

    except Exception as error:
        st.warning(f"Grad-CAM could not be generated: {error}")


# =========================================================
# ANALYTICS
# =========================================================

def analytics():

    render_html(
        """
        <div class="hero">
        <div class="hero-eyebrow">MODEL OBSERVABILITY</div>
        <div class="hero-title">Prediction <span>Analytics</span></div>
        <div class="hero-description">
        Review prediction activity and human-verified real-world results.
        </div>
        </div>
        """
    )

    history = pd.DataFrame()

    if os.path.exists(HISTORY_FILE):
        try:
            history = pd.read_csv(HISTORY_FILE)
        except Exception as error:
            st.warning(f"Prediction history could not be read: {error}")

    real_world = pd.DataFrame()

    if os.path.exists(REAL_WORLD_FILE):
        try:
            real_world = pd.read_csv(REAL_WORLD_FILE)
        except Exception as error:
            st.warning(f"Real-world test history could not be read: {error}")

    metric_columns = st.columns(3)

    with metric_columns[0]:
        st.metric("Predictions", len(history))

    with metric_columns[1]:
        st.metric("Real-World Tests", len(real_world))

    with metric_columns[2]:
        if len(real_world) > 0 and "correct" in real_world.columns:
            accuracy = pd.to_numeric(
                real_world["correct"],
                errors="coerce"
            ).mean() * 100
            st.metric("Verified Accuracy", f"{accuracy:.2f}%")
        else:
            st.metric("Verified Accuracy", "N/A")

    if not history.empty:
        render_html("<div class=\"section-title\">Prediction History</div>")
        st.dataframe(history, use_container_width=True)

    if not real_world.empty:
        render_html("<div class=\"section-title\">Real-World Results</div>")
        st.dataframe(real_world, use_container_width=True)


# =========================================================
# PREDICTION HISTORY
# =========================================================

def prediction_history():

    render_html(
        """
        <div class="hero">
        <div class="hero-eyebrow">ACTIVITY LOG</div>
        <div class="hero-title">Prediction <span>History</span></div>
        <div class="hero-description">
        Review previous predictions saved by PlantVision AI.
        </div>
        </div>
        """
    )

    if not os.path.exists(HISTORY_FILE):
        st.info("No prediction history has been recorded yet.")
        return

    try:
        history = pd.read_csv(HISTORY_FILE)
    except Exception as error:
        st.error(f"Prediction history could not be read: {error}")
        return

    st.dataframe(history, use_container_width=True)

    st.download_button(
        "⬇️ Download Prediction History",
        history.to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv"
    )


# =========================================================
# LEARNING CENTER
# =========================================================

def learning_center():

    render_html(
        """
        <div class="hero">
        <div class="hero-eyebrow">HUMAN-IN-THE-LOOP WORKFLOW</div>
        <div class="hero-title">Learning <span>Center</span></div>
        <div class="hero-description">
        Review how confirmed samples can support controlled future model improvement.
        </div>
        </div>
        """
    )

    learning_samples = 0

    if os.path.exists(LEARNING_DIR):
        for _, _, files in os.walk(LEARNING_DIR):
            learning_samples += len(files)

    st.metric("Collected Learning Samples", learning_samples)

    render_html(
        """
        <div class="glass-card">
        <div class="info-row">
        <div class="info-label">01 — COLLECT</div>
        <div class="info-value">Save human-corrected samples by class.</div>
        </div>
        <div class="info-row">
        <div class="info-label">02 — VERIFY</div>
        <div class="info-value">Review labels and remove questionable images.</div>
        </div>
        <div class="info-row">
        <div class="info-label">03 — RETRAIN</div>
        <div class="info-value">Use verified data in a controlled training experiment.</div>
        </div>
        <div class="info-row">
        <div class="info-label">04 — EVALUATE</div>
        <div class="info-value">Compare the new model against held-out and real-world data.</div>
        </div>
        </div>
        """
    )

    st.warning(
        "Collected samples should be reviewed before being used for retraining."
    )


# =========================================================
# DISEASE LIBRARY
# =========================================================

def disease_library():

    render_html(
        """
        <div class="hero">
        <div class="hero-eyebrow">KNOWLEDGE BASE</div>
        <div class="hero-title">Disease <span>Library</span></div>
        <div class="hero-description">
        Explore the disease categories supported by the model.
        </div>
        </div>
        """
    )

    search = st.text_input("🔎 Search plant or disease").lower().strip()

    filtered = [
        label
        for label in CLASS_NAMES
        if not search or search in label.lower()
    ]

    st.caption(f"{len(filtered)} matching classes")

    for label in filtered:

        plant, disease = split_label(label)

        with st.expander(f"🌿 {plant} — {disease}"):

            info = disease_information(label)

            st.write("**Description**")
            st.write(info["description"])

            st.write("**Symptoms**")
            st.write(info["symptoms"])

            st.write("**Management**")
            st.write(info["management"])


# =========================================================
# ABOUT
# =========================================================

def about():

    render_html(
        """
        <div class="hero">
        <div class="hero-eyebrow">FINAL YEAR PROJECT</div>
        <div class="hero-title">About <span>PlantVision AI</span></div>
        <div class="hero-description">
        An intelligent computer-vision prototype for plant disease classification.
        </div>
        </div>
        """
    )

    render_html(
        """
        <div class="glass-card">
        <div class="section-title" style="margin-top:0;">🌱 Project Overview</div>
        <div style="color:#b5cebf;font-size:14px;line-height:1.8;">
        PlantVision AI uses a Convolutional Neural Network
        to analyze plant leaf images and classify them into
        38 supported plant-health categories.
        <br><br>
        The enhanced prototype includes confidence-aware
        classification, Grad-CAM interpretability,
        human feedback collection, prediction history,
        analytics and a verified learning queue.
        </div>
        </div>
        """
    )

    render_html(
        f"""
        <div class="section-title">🧠 Technical Details</div>
        <div class="glass-card">
        <div class="info-row">
        <div class="info-label">MODEL</div>
        <div class="info-value">Sequential Convolutional Neural Network</div>
        </div>
        <div class="info-row">
        <div class="info-label">CLASSES</div>
        <div class="info-value">{len(CLASS_NAMES)}</div>
        </div>
        <div class="info-row">
        <div class="info-label">INPUT</div>
        <div class="info-value">128 × 128 × 3 RGB</div>
        </div>
        <div class="info-row">
        <div class="info-label">EXPLAINABILITY</div>
        <div class="info-value">Grad-CAM</div>
        </div>
        <div class="info-row">
        <div class="info-label">MODEL VERSION</div>
        <div class="info-value">{MODEL_VERSION}</div>
        </div>
        </div>
        """
    )

    st.warning(
        "The classifier directly recognizes only the disease categories represented in its training classes. An unfamiliar disease may be assigned to the closest known category."
    )

    render_html(
        """
        <div class="section-title">🧬 Experimental Extension</div>
        <div class="glass-card">
        <div class="info-row">
        <div class="info-label">GAN EXTENSION</div>
        <div class="info-value">
        GAN-generated samples can be evaluated as an experimental augmentation technique for classes with limited real-world examples.
        </div>
        </div>
        </div>
        """
    )


# =========================================================
# SIDEBAR
# =========================================================

render_html(
    """
    <div class="brand">🌿 PlantVision AI</div>
    <div class="brand-sub">Intelligent Plant Health</div>
    """
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔬 Disease Recognition",
        "🧪 Real-World Testing",
        "📊 Analytics",
        "🕘 Prediction History",
        "🧠 Learning Center",
        "📚 Disease Library",
        "ℹ️ About"
    ]
)

render_html(
    f"""
    <div class="sidebar-model">
    <div class="sidebar-model-title">ACTIVE MODEL</div>
    <div class="sidebar-model-value">CNN · {MODEL_VERSION}</div>
    <div style="color:#708e7c;font-size:10px;margin-top:6px;">38 disease classes</div>
    </div>
    <div class="sidebar-footer">
    PlantVision AI
    <br>
    Final Year Project
    <br><br>
    Detect · Understand · Protect
    </div>
    """
)


# =========================================================
# PAGE ROUTING
# =========================================================

if page == "🏠 Dashboard":
    dashboard()
elif page == "🔬 Disease Recognition":
    disease_recognition()
elif page == "🧪 Real-World Testing":
    real_world_testing()
elif page == "📊 Analytics":
    analytics()
elif page == "🕘 Prediction History":
    prediction_history()
elif page == "🧠 Learning Center":
    learning_center()
elif page == "📚 Disease Library":
    disease_library()
elif page == "ℹ️ About":
    about()
