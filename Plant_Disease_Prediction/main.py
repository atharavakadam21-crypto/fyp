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
import matplotlib
import matplotlib.cm as cm

from PIL import Image

from rag_knowledge import retrieve_knowledge


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

MODEL_PATH = os.getenv("PLANTVISION_MODEL_PATH", "trained_plant_disease_model.keras")
MODEL_VERSION = os.getenv("PLANTVISION_MODEL_VERSION", "production")
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
    """Apply the full PlantVision botanical-luxury interface theme."""
    background_image = ""
    if os.path.exists(BACKGROUND_PATH):
        try:
            with open(BACKGROUND_PATH, "rb") as file:
                encoded = base64.b64encode(file.read()).decode()
            background_image = "url('data:image/jpeg;base64," + encoded + "')"
        except Exception:
            background_image = ""

    app_background = f"""
        background-image:
            radial-gradient(circle at 12% 8%, rgba(86,227,159,.13), transparent 30%),
            radial-gradient(circle at 88% 78%, rgba(46,213,115,.10), transparent 32%),
            linear-gradient(135deg, rgba(3,17,12,.96), rgba(6,35,23,.94))
            {", " + background_image if background_image else ""};
    """

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {{
        --pv-bg:#04150e; --pv-panel:rgba(10,39,26,.76); --pv-panel2:rgba(15,53,35,.84);
        --pv-mint:#68f0aa; --pv-green:#2ed573; --pv-text:#effff5; --pv-muted:#a2c2af;
        --pv-border:rgba(104,240,170,.19); --pv-glow:0 18px 48px rgba(0,0,0,.25),0 0 24px rgba(104,240,170,.055);
    }}
    html {{ scroll-behavior:smooth; }}
    .stApp {{
        {app_background}
        background-size:cover; background-position:center; background-attachment:fixed;
        color:var(--pv-text); font-family:'DM Sans',sans-serif;
    }}
    [data-testid="stAppViewContainer"] {{ background:transparent; }}
    .main .block-container {{ max-width:1460px; padding:2rem clamp(1rem,3vw,3rem) 4rem; }}
    h1,h2,h3,h4,[data-testid="stMetricValue"] {{ font-family:'Manrope','DM Sans',sans-serif !important; letter-spacing:-.035em; }}
    p,li,label {{ line-height:1.7; }}
    .stMarkdown,.stButton,.stFileUploader,.stSelectbox,.stTextInput,.stSlider,.stDataFrame,.stMetric,.stExpander,.stAlert {{
        animation:pv-rise .42s ease both;
    }}
    @keyframes pv-rise {{ from {{ opacity:0; transform:translateY(9px); }} to {{ opacity:1; transform:translateY(0); }} }}
    @keyframes pv-scroll-reveal {{
        from {{ opacity: 0; transform: translateY(28px); filter: blur(3px); }}
        to {{ opacity: 1; transform: translateY(0); filter: blur(0); }}
    }}
    /* Scroll-driven reveal in modern Chromium; unsupported browsers keep content visible. */
    @supports (animation-timeline: view()) {{
        .hero, .hero-card, .section-title, .glass-card, .metric-card,
        .metric-grid-card, .result-card, .upload-card, .info-row, .data-row,
        [data-testid="stMetric"], [data-testid="stExpander"], .stDataFrame {{
            animation-name: pv-scroll-reveal;
            animation-duration: 1ms;
            animation-timing-function: linear;
            animation-fill-mode: both;
            animation-timeline: view();
            animation-range: entry 0% entry 38%;
        }}
        .hero-title, .hero-description, .hero-desc, .hero-eyebrow, .hero-tag {{
            animation-name: pv-scroll-reveal;
            animation-duration: 1ms;
            animation-timing-function: linear;
            animation-fill-mode: both;
            animation-timeline: view();
            animation-range: entry 0% entry 55%;
        }}
    }}
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            scroll-behavior: auto !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }}
    }}
    @keyframes pv-glow {{ 0%,100% {{ box-shadow:0 0 14px rgba(104,240,170,.10); }} 50% {{ box-shadow:0 0 27px rgba(104,240,170,.22); }} }}
    ::selection {{ background:rgba(104,240,170,.28); color:white; }}
    ::-webkit-scrollbar {{ width:9px; height:9px; }}
    ::-webkit-scrollbar-track {{ background:#04150e; }}
    ::-webkit-scrollbar-thumb {{ background:#1a5b3c; border:2px solid #04150e; border-radius:99px; }}
    ::-webkit-scrollbar-thumb:hover {{ background:var(--pv-mint); }}

    /* Premium glass panels and visual hierarchy */
    .hero,.hero-card {{
        position:relative; overflow:hidden; padding:clamp(25px,4vw,48px);
        border:1px solid var(--pv-border); border-radius:26px;
        background:linear-gradient(125deg,rgba(17,66,42,.82),rgba(5,27,18,.76));
        box-shadow:var(--pv-glow); backdrop-filter:blur(18px); margin-bottom:28px;
        animation:pv-rise .5s ease both;
    }}
    .hero:after,.hero-card:after {{
        content:""; position:absolute; width:290px; height:290px; right:-100px; top:-130px;
        border-radius:50%; background:radial-gradient(circle,rgba(104,240,170,.18),transparent 68%); pointer-events:none;
    }}
    .hero-title {{ color:#f4fff8; font-family:'Manrope',sans-serif; font-size:clamp(34px,5vw,62px); font-weight:800; line-height:1.08; letter-spacing:-.055em; }}
    .hero-title span {{ color:var(--pv-mint); text-shadow:0 0 28px rgba(104,240,170,.22); }}
    .hero-eyebrow,.hero-tag {{ display:inline-flex; align-items:center; gap:8px; color:var(--pv-mint); font-size:11px; font-weight:800; letter-spacing:2.3px; text-transform:uppercase; margin-bottom:14px; }}
    .hero-description,.hero-desc {{ color:#b4d1bf; max-width:820px; font-size:15px; line-height:1.85; margin-top:16px; }}
    .pill,.pill-badge {{ display:inline-flex; align-items:center; gap:6px; padding:7px 12px; margin:5px 5px 0 0; border:1px solid rgba(104,240,170,.22); border-radius:999px; background:rgba(104,240,170,.08); color:#c9ffdc; font-size:11px; font-weight:700; transition:all .2s ease; }}
    .pill:hover,.pill-badge:hover {{ transform:translateY(-2px); border-color:var(--pv-mint); background:rgba(104,240,170,.15); }}
    .metric-card,.metric-grid-card {{ min-height:128px; padding:21px 23px; border:1px solid var(--pv-border); border-radius:19px; background:linear-gradient(145deg,rgba(13,49,32,.88),rgba(6,28,19,.86)); box-shadow:0 12px 32px rgba(0,0,0,.14); transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease; }}
    .metric-card:hover,.metric-grid-card:hover {{ transform:translateY(-4px); border-color:rgba(104,240,170,.48); box-shadow:var(--pv-glow); }}
    .metric-icon {{ font-size:25px; margin-bottom:10px; }}
    .metric-number,.metric-val {{ color:#f1fff6; font-family:'Manrope',sans-serif; font-size:clamp(25px,2.4vw,34px); font-weight:800; letter-spacing:-.04em; }}
    .metric-label,.metric-header {{ color:#83a994; font-size:10px; font-weight:800; letter-spacing:1.6px; text-transform:uppercase; }}
    .metric-sub {{ color:var(--pv-mint); font-size:12px; margin-top:7px; }}
    .glass-card,.result-card,.upload-card {{ border:1px solid var(--pv-border); border-radius:20px; background:linear-gradient(145deg,rgba(11,43,29,.84),rgba(5,25,17,.83)); padding:clamp(18px,2.4vw,28px); margin-bottom:20px; box-shadow:0 12px 35px rgba(0,0,0,.12); backdrop-filter:blur(14px); transition:border-color .2s ease,transform .2s ease,box-shadow .2s ease; }}
    .glass-card:hover,.result-card:hover {{ border-color:rgba(104,240,170,.32); box-shadow:var(--pv-glow); }}
    .result-card {{ background:linear-gradient(135deg,rgba(18,76,47,.86),rgba(5,29,19,.92)); }}
    .upload-card {{ text-align:center; border-style:dashed; padding:38px 24px; }}
    .upload-icon {{ font-size:42px; margin-bottom:12px; }}
    .upload-title {{ color:#eafff1; font-size:21px; font-weight:800; }}
    .upload-subtitle {{ color:#91b6a0; font-size:13px; line-height:1.8; margin-top:8px; }}
    .section-title {{ color:#f0fff5; font-family:'Manrope',sans-serif; font-size:clamp(19px,2vw,24px); font-weight:800; letter-spacing:-.035em; margin:30px 0 16px; }}
    .section-title:before {{ content:""; display:inline-block; width:4px; height:22px; vertical-align:-4px; margin-right:10px; border-radius:9px; background:var(--pv-mint); box-shadow:0 0 12px rgba(104,240,170,.55); }}
    .info-row,.data-row {{ display:flex; justify-content:space-between; align-items:flex-start; gap:18px; padding:13px 0; border-bottom:1px solid rgba(104,240,170,.09); }}
    .info-row:last-child,.data-row:last-child {{ border-bottom:none; }}
    .info-label,.data-label {{ color:var(--pv-mint); font-size:10px; font-weight:800; letter-spacing:1.3px; }}
    .info-value,.data-value {{ color:#c0d8ca; font-size:13px; text-align:right; overflow-wrap:anywhere; }}
    .status-badge,.diagnosis-status {{ display:inline-flex; border:1px solid rgba(104,240,170,.3); background:rgba(104,240,170,.1); color:var(--pv-mint); border-radius:999px; padding:7px 12px; font-size:10px; font-weight:800; letter-spacing:1.2px; }}
    .result-label {{ color:#8db79c; font-size:10px; font-weight:800; letter-spacing:2px; }}
    .result-plant {{ color:#c8f0d6; font-size:24px; font-weight:800; }}
    .result-disease {{ color:#f2fff6; font-size:clamp(23px,3vw,32px); font-weight:800; }}
    .result-confidence {{ color:var(--pv-mint); font-size:clamp(30px,4vw,44px); font-weight:800; }}
    .footer,.sidebar-footer {{ color:#759984; font-size:11px; text-align:center; line-height:1.9; padding:22px 0; }}

    /* Sidebar: branded, clear, and visibly interactive */
    [data-testid="stSidebar"] {{ background:linear-gradient(180deg,rgba(3,19,12,.98),rgba(5,31,20,.98)) !important; border-right:1px solid var(--pv-border); }}
    [data-testid="stSidebar"] > div:first-child {{ background:transparent; }}
    [data-testid="stSidebar"] .block-container {{ padding:1.7rem 1rem 2rem; }}
    .brand-lockup {{ display:flex; align-items:center; gap:11px; margin-bottom:4px; }}
    .brand-logo {{ display:block; width:47px; height:47px; flex:0 0 47px; filter:drop-shadow(0 0 12px rgba(104,240,170,.16)); }}
    .brand {{ color:var(--pv-mint); font-family:'Manrope',sans-serif; font-size:22px; line-height:1.1; font-weight:800; letter-spacing:-1px; white-space:nowrap; }}
    .brand span {{ color:#f2fff6; }}
    .brand-sub {{ color:#83a994; font-size:9px; font-weight:800; letter-spacing:1.45px; text-transform:uppercase; margin:5px 0 22px; }}
    .brand-sub {{ color:#83a994; font-size:10px; font-weight:800; letter-spacing:2px; text-transform:uppercase; margin:4px 0 22px; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {{ color:#9bbda8; font-size:11px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px; }}
    [data-testid="stSidebar"] [role="radiogroup"] {{ gap:7px; }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{ border:1px solid rgba(104,240,170,.09); border-radius:12px; padding:9px 12px; background:rgba(10,39,26,.55); transition:all .2s ease; }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{ border-color:rgba(104,240,170,.42); background:rgba(18,62,40,.82); transform:translateX(3px); }}
    .sidebar-model {{ padding:16px; border:1px solid var(--pv-border); border-radius:15px; background:rgba(10,39,26,.7); margin-top:20px; }}
    .sidebar-model-title {{ color:#80a891; font-size:10px; letter-spacing:1.8px; font-weight:800; }}
    .sidebar-model-value {{ color:#c8ffdc; font-size:13px; font-weight:800; margin-top:6px; }}

    /* Buttons, forms, uploaders, tables and expanders */
    div.stButton > button {{ min-height:44px; border:1px solid rgba(104,240,170,.3) !important; border-radius:13px !important; color:#f2fff6 !important; background:linear-gradient(135deg,#1a6743,#0d3c28) !important; font-family:'Manrope',sans-serif !important; font-weight:800 !important; box-shadow:0 7px 20px rgba(0,0,0,.18); transition:all .2s ease !important; }}
    div.stButton > button:hover {{ transform:translateY(-2px); border-color:var(--pv-mint) !important; box-shadow:0 10px 26px rgba(46,213,115,.2); }}
    div.stButton > button:active {{ transform:scale(.985); }}
    [data-testid="stFileUploader"] {{ padding:12px; border:1px dashed rgba(104,240,170,.35); border-radius:16px; background:rgba(9,38,25,.55); }}
    [data-testid="stFileUploader"]:hover {{ border-color:var(--pv-mint); background:rgba(14,53,35,.72); }}
    [data-baseweb="select"] > div,[data-baseweb="input"] > div,[data-baseweb="textarea"] {{ border-color:rgba(104,240,170,.22) !important; border-radius:11px !important; background:rgba(5,25,17,.75) !important; }}
    [data-baseweb="popover"] {{ border:1px solid var(--pv-border); border-radius:14px; }}
    [data-testid="stSlider"] [role="slider"] {{ background:var(--pv-mint) !important; border-color:var(--pv-mint) !important; }}
    [data-testid="stExpander"] {{ border:1px solid var(--pv-border); border-radius:14px; background:rgba(8,34,22,.55); overflow:hidden; }}
    [data-testid="stExpander"] summary:hover {{ color:var(--pv-mint); }}
    [data-testid="stDataFrame"],[data-testid="stTable"] {{ border:1px solid var(--pv-border); border-radius:14px; overflow:hidden; }}
    [data-testid="stAlert"] {{ border-radius:13px; }}
    [data-testid="stMetric"] {{ background:rgba(10,39,26,.55); border:1px solid var(--pv-border); border-radius:15px; padding:14px 16px; }}
    [data-testid="stMetricLabel"] {{ color:#9bbda8 !important; }}
    [data-testid="stMetricValue"] {{ color:#effff5 !important; }}
    hr {{ border-color:rgba(104,240,170,.12) !important; }}
    @media (max-width: 760px) {{
        .main .block-container {{ padding:1rem .8rem 2rem; }}
        .hero,.hero-card {{ padding:24px 20px; border-radius:19px; }}
        .hero-title {{ font-size:36px; }}
        .glass-card,.result-card {{ padding:18px; }}
        .info-row,.data-row {{ gap:10px; }}
        .info-value,.data-value {{ max-width:58%; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


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
    """Load the trained CNN once and verify its inference contract."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained CNN model not found at: {os.path.abspath(MODEL_PATH)}"
        )

    loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    input_shape = loaded_model.input_shape
    output_shape = loaded_model.output_shape

    # This project preprocesses RGB leaves to 128x128 and predicts 38 classes.
    if isinstance(input_shape, list) or len(input_shape) != 4:
        raise ValueError(f"Expected a single image input, got {input_shape!r}")
    if tuple(input_shape[1:]) != (128, 128, 3):
        raise ValueError(
            f"Model expects input shape {input_shape}; PlantVision preprocessing "
            "currently supplies (None, 128, 128, 3)."
        )
    if isinstance(output_shape, list) or len(output_shape) != 2:
        raise ValueError(f"Expected one 38-class output tensor, got {output_shape!r}")
    if output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Model outputs {output_shape[-1]} classes, but CLASS_NAMES contains "
            f"{len(CLASS_NAMES)} labels. Refusing to show mismatched predictions."
        )
    return loaded_model


model = None
MODEL_LOAD_ERROR = None

try:
    model = load_model()
except Exception as error:
    MODEL_LOAD_ERROR = str(error)
    st.error(
        "The trained CNN could not be loaded or does not match the expected "
        f"PlantVision input/output contract. No simulated predictions will be used. "
        f"Details: {MODEL_LOAD_ERROR}"
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
    """Return validated probabilities from the real trained CNN only."""
    if model is None:
        reason = MODEL_LOAD_ERROR or "model loading failed"
        raise RuntimeError(
            f"Trained CNN is unavailable ({reason}). Check MODEL_PATH and model files."
        )

    input_array = preprocess_image(image)
    outputs = np.asarray(model.predict(input_array, verbose=0))
    if outputs.ndim != 2 or outputs.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected model prediction shape {outputs.shape}; expected (1, 38)."
        )

    probabilities = outputs[0].astype(np.float64)
    if probabilities.shape[0] != len(CLASS_NAMES):
        raise RuntimeError(
            f"Model returned {probabilities.shape[0]} scores for "
            f"{len(CLASS_NAMES)} configured classes."
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Model returned NaN or infinite prediction scores.")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise RuntimeError(
            "Model outputs are not probabilities in [0, 1]; verify the final "
            "classification layer before displaying confidence."
        )

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




# =========================================================
# AI PLANT HEALTH ASSISTANT (OpenAI / Gemini)
# =========================================================

def get_secret_value(name):
    """Read API keys from environment variables or Streamlit secrets."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def request_ai_guidance(provider, plant, disease, confidence, symptoms, management):
    """Request cautious crop guidance from OpenAI or Gemini using Python's standard library."""
    import json as _json
    import urllib.request
    import urllib.error
    retrieved_docs = retrieve_knowledge(f"{plant} {disease}", top_k=3)
    retrieved_context = "\n\n".join(
        f"[{doc['id']}] {doc['title']} (source: {doc['source_title']} — {doc['source_url']})\n{doc['text']}"
        for doc in retrieved_docs
    ) or "No relevant local knowledge-base passages were retrieved."
    prompt = f"""You are PlantVision AI's plant-health education assistant.
CNN prediction: crop={plant}; predicted condition={disease}; model confidence={confidence:.1%}.
Existing app notes (possibly generic): symptoms={symptoms}; management={management}.
Retrieved local knowledge passages (supporting context, not guaranteed truth):
{retrieved_context}
Use retrieved passages where relevant. Do not invent citations. Cite a supporting passage by ID such as [PV-GENERAL-01]. If passages do not support a specific claim, label it general guidance and recommend local expert confirmation.
Return a practical, grower-friendly Markdown report using these exact headings:
## 1. What the prediction means
Explain the predicted condition in plain language and state that the CNN prediction is not a confirmed diagnosis.
## 2. Symptoms to check
List visible signs that would support or weaken this prediction, plus important lookalike conditions where relevant.
## 3. Immediate actions
Give safe, practical steps the grower can take now to limit spread or stress while confirming the diagnosis.
## 4. Prevention measures
Give at least 5 specific, actionable prevention steps tailored to this crop and condition when supported: sanitation and removal of infected debris, watering/leaf-wetness practices, airflow and spacing, crop rotation or clean planting material where relevant, monitoring, and resistant varieties where available. Do not force irrelevant measures; explain when a measure does not apply.
## 5. Treatment and ongoing care
Start with non-chemical integrated pest/disease management. If a pesticide may be appropriate, do not name a brand or give a dose/mixing instructions; tell the user to consult local agricultural extension guidance and follow the registered product label.
## 6. When to contact an agricultural expert
Give clear escalation signs, especially severe/rapid spread, uncertain diagnosis, or valuable crops at risk.
## 7. Important cautions
Mention uncertainty and avoid promising a cure.
Treat the classifier result as tentative, not a confirmed diagnosis. Make the advice crop- and disease-specific only when reasonably supported; do not invent facts. Separate general precautions from disease-specific recommendations. Keep the language easy for growers, use concise bullet points, and do not prescribe pesticide brands, exact chemical dosages, or mixing instructions."""
    if provider == "OpenAI":
        api_key = get_secret_value("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured. Add it to your environment or .streamlit/secrets.toml.")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        endpoint = "https://api.openai.com/v1/chat/completions"
        payload = {"model": model_name, "temperature": 0.3, "messages": [
            {"role": "system", "content": "Provide cautious, evidence-aware plant health education. Never claim a classifier confirms disease."},
            {"role": "user", "content": prompt}]}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    else:
        api_key = get_secret_value("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured. Add it to your environment or .streamlit/secrets.toml.")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {"systemInstruction": {"parts": [{"text": "Provide cautious plant health education; never claim a classifier confirms disease."}]},
                   "contents": [{"parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2400}}
        headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(endpoint, data=_json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = _json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"{provider} API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach {provider}: {error.reason}") from error
    if provider == "OpenAI":
        choices = data.get("choices", [])
        result = choices[0].get("message", {}).get("content", "") if choices else ""
    else:
        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        result = "\\n".join(part.get("text", "") for part in parts if part.get("text"))
    if not result.strip():
        raise RuntimeError(f"{provider} returned no text. Check model name, API access, or safety filters.")

    result = result.strip()
    # Gemini can occasionally stop after only the first requested headings.
    # Detect that case and ask once for the missing prevention/treatment sections.
    required_sections = [
        "What the prediction means",
        "Symptoms to check",
        "Immediate actions",
        "Prevention measures",
        "Treatment and ongoing care",
        "When to contact an agricultural expert",
        "Important cautions",
    ]
    missing_sections = [
        heading for heading in required_sections
        if heading.casefold() not in result.casefold()
    ]
    if provider == "Gemini" and missing_sections:
        follow_up = (
            "Your previous answer was incomplete. Return a complete standalone "
            "Markdown report, not just a continuation. Include all seven numbered "
            "headings: "
            + "; ".join(f"{i + 1}. {heading}" for i, heading in enumerate(required_sections))
            + ". The Prevention measures section must contain at least five actionable "
            "steps when applicable. Do not omit sections. Keep advice cautious and "
            "do not provide pesticide brands or dosages.\n\nPrevious incomplete answer:\n"
            + result
        )
        follow_payload = {
            "systemInstruction": {"parts": [{"text": "You are a careful plant-health education assistant. Always return all requested sections and actionable prevention advice. Do not claim an AI prediction is a confirmed diagnosis."}]},
            "contents": [{"parts": [{"text": prompt + "\n\n" + follow_up}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2400},
        }
        follow_request = urllib.request.Request(
            endpoint,
            data=_json.dumps(follow_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(follow_request, timeout=45) as response:
                follow_data = _json.loads(response.read().decode("utf-8"))
            follow_candidates = follow_data.get("candidates", [])
            follow_parts = follow_candidates[0].get("content", {}).get("parts", []) if follow_candidates else []
            follow_result = "\\n".join(
                part.get("text", "") for part in follow_parts if part.get("text")
            ).strip()
            if follow_result:
                result = follow_result
        except (urllib.error.HTTPError, urllib.error.URLError):
            # Preserve the original response if the retry fails.
            pass

    return result


def ai_plant_health_assistant(label, confidence):
    """Render optional LLM-generated summary and prevention guidance."""
    plant, disease = split_label(label)
    information = disease_information(label)
    render_html("""
        <div class="section-title">🌿 AI Plant Health Assistant</div>
        <div class="glass-card">
          <div style="color:#c5ffda;font-size:15px;font-weight:800;margin-bottom:7px;">Understand the result. Plan the next step.</div>
          <div style="color:#9bb9a7;font-size:13px;line-height:1.7;">Generate a plain-English summary and prevention checklist using OpenAI or Gemini. Only the predicted crop, condition, confidence, and generic app notes are sent to the selected provider; your leaf image is not uploaded to the LLM.</div>
        </div>
    """)
    provider = st.selectbox("AI provider", ["Gemini", "OpenAI"], key="plant_health_ai_provider")
    key_name = "OPENAI_API_KEY" if provider == "OpenAI" else "GEMINI_API_KEY"
    st.caption(f"{provider} API key: {'configured' if get_secret_value(key_name) else 'not configured'}")
    if st.button("✨ Generate summary & prevention plan", use_container_width=True, key="generate_plant_health_guidance"):
        with st.spinner(f"Generating plant-health guidance with {provider}..."):
            try:
                report = request_ai_guidance(provider, plant, disease, confidence, information["symptoms"], information["management"])
                st.session_state["plant_health_guidance"] = {
                    "provider": provider,
                    "label": label,
                    "report": report,
                    "sources": retrieve_knowledge(f"{plant} {disease}", top_k=3),
                }
            except Exception as error:
                st.session_state.pop("plant_health_guidance", None)
                st.error(str(error))
    saved = st.session_state.get("plant_health_guidance")
    if saved and saved.get("label") == label and saved.get("provider") == provider:
        st.markdown(saved["report"])
        with st.expander("📚 Retrieved knowledge used by the assistant"):
            sources = saved.get("sources", [])
            if sources:
                for source in sources:
                    st.markdown("**[" + source["id"] + "] " + source["title"] + "**")
                    st.markdown("Source: [" + source["source_title"] + "](" + source["source_url"] + ")")
                    st.caption(source["source_publisher"] + " · Retrieval score: " + str(source["score"]))
                    st.write(source["text"])
            else:
                st.info("No local knowledge passages were retrieved for this report.")
        st.caption("Educational guidance only. The starter knowledge base is not a substitute for verified regional agricultural advice.")


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
        ("🧠", f"{model.count_params() / 1_000_000:.2f}M" if model is not None else "Unavailable", "MODEL PARAMETERS"),
        ("🖼️", "128×128", "INPUT RESOLUTION"),
        ("⚡", "CNN" if model is not None else "Offline", "ARCHITECTURE")
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
# GRAD-CAM EXPLAINABILITY
# =========================================================

def generate_gradcam_details(image, class_index):
    """Return original image, raw heatmap, overlay, and layer metadata."""
    if model is None:
        raise RuntimeError("The trained model is not available.")

    class_index = int(class_index)
    if class_index < 0 or class_index >= len(CLASS_NAMES):
        raise ValueError(f"Invalid class index: {class_index}")

    # Prefer a spatially detailed convolution. Fall back only to a layer
    # that can actually be connected to the model output.
    preferred_names = ("conv2d_7", "conv2d_8", "conv2d_9")
    candidates = []
    for name in preferred_names:
        try:
            layer = model.get_layer(name)
            if isinstance(layer, tf.keras.layers.Conv2D):
                candidates.append(layer)
        except ValueError:
            pass
    if not candidates:
        candidates = [
            layer for layer in model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]

    if not candidates:
        raise RuntimeError("No top-level Conv2D layer is available for Grad-CAM.")

    # Differentiate the pre-softmax class logit when the model exposes a
    # final Dense(softmax) layer. Probabilities near 1.0 can saturate and
    # produce weak/blank Grad-CAM maps even when the convolution is relevant.
    softmax_head = next((
        layer for layer in reversed(model.layers)
        if isinstance(layer, tf.keras.layers.Dense)
        and layer.activation == tf.keras.activations.softmax
    ), None)

    grad_model = None
    target_layer = None
    last_error = None
    for candidate in candidates:
        try:
            feature_outputs = [candidate.output, softmax_head.input if softmax_head is not None else model.output]
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=feature_outputs,
            )
            # Force a forward pass to catch disconnected or incompatible layers.
            grad_model(tf.convert_to_tensor(preprocess_image(image)), training=False)
            target_layer = candidate
            break
        except Exception as error:
            last_error = error
    if grad_model is None or target_layer is None:
        raise RuntimeError(
            "Could not connect a convolutional feature map to the model output."
        ) from last_error

    input_tensor = tf.convert_to_tensor(preprocess_image(image))
    with tf.GradientTape() as tape:
        activations, head_or_predictions = grad_model(input_tensor, training=False)
        if softmax_head is not None:
            class_logits = tf.linalg.matmul(head_or_predictions, softmax_head.kernel)
            if softmax_head.use_bias:
                class_logits = tf.nn.bias_add(class_logits, softmax_head.bias)
            class_score = class_logits[:, class_index]
        else:
            # Generic fallback for models without a visible Dense-softmax head.
            class_score = tf.math.log(tf.clip_by_value(
                head_or_predictions[:, class_index], 1e-7, 1.0
            ))
    gradients = tape.gradient(class_score, activations)
    if gradients is None:
        raise RuntimeError("Gradients are disconnected for the selected class.")

    # Standard Grad-CAM: spatially average the first-order gradients.
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    raw = tf.reduce_sum(
        activations[0] * weights[0][tf.newaxis, tf.newaxis, :],
        axis=-1,
    )
    raw = tf.nn.relu(raw)
    maximum = tf.reduce_max(raw)
    normalized = tf.math.divide_no_nan(raw, maximum).numpy()

    if not np.isfinite(normalized).all() or float(np.max(normalized)) <= 0:
        raise RuntimeError(
            "Grad-CAM returned an empty/non-finite map for this class."
        )

    # Keep the normalized grayscale heatmap separate from the colour overlay.
    heatmap_image = Image.fromarray(
        np.uint8(np.clip(normalized, 0.0, 1.0) * 255)
    ).resize(image.size, Image.Resampling.BILINEAR)
    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    colored = np.uint8(
        np.clip(matplotlib.colormaps["jet"](heatmap_array)[..., :3], 0.0, 1.0) * 255
    )
    original_image = image.convert("RGB")
    colored_heatmap = Image.fromarray(colored).convert("RGB")
    overlay = Image.blend(original_image, colored_heatmap, alpha=0.35)

    return {
        "original": original_image,
        "heatmap": heatmap_image.convert("RGB"),
        "overlay": overlay,
        "layer_name": target_layer.name,
        "feature_shape": tuple(int(d) if d is not None else -1 for d in activations.shape[1:]),
        "heatmap_min": float(np.min(normalized)),
        "heatmap_max": float(np.max(normalized)),
        "heatmap_std": float(np.std(normalized)),
    }


def generate_gradcam(image, class_index):
    """Backward-compatible Grad-CAM overlay helper."""
    return generate_gradcam_details(image, class_index)["overlay"]

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
        <div class="glass-card">
          <div style="color:#b9f4cc;font-size:15px;font-weight:800;margin-bottom:7px;">
            See what influenced the CNN prediction
          </div>
          <div style="color:#9bb9a7;font-size:13px;line-height:1.7;">
            The heatmap highlights image regions that contributed to the selected class score.
            It is an explanation aid, not proof that a region contains a disease.
          </div>
        </div>
        """
    )

    try:
        explanation = generate_gradcam_details(
            result["image"],
            int(indices[0])
        )

        metadata_cols = st.columns(3)
        with metadata_cols[0]:
            st.metric("Target layer", explanation["layer_name"])
        with metadata_cols[1]:
            st.metric("Feature map", " × ".join(str(v) for v in explanation["feature_shape"][:2]))
        with metadata_cols[2]:
            st.metric("Heatmap variation", f"{explanation['heatmap_std']:.3f}")

        opacity = st.slider(
            "Heatmap overlay intensity",
            min_value=0.10,
            max_value=0.80,
            value=0.40,
            step=0.05,
            help="Adjust how strongly the activation heatmap is blended with the original leaf."
        )

        # Rebuild the coloured heatmap from the grayscale activation map so the
        # slider changes the actual overlay, without rerunning CNN inference.
        heatmap_gray = np.asarray(
            explanation["heatmap"].convert("L"),
            dtype=np.float32
        ) / 255.0
        heatmap_rgb = np.uint8(
            np.clip(matplotlib.colormaps["jet"](heatmap_gray)[..., :3], 0.0, 1.0) * 255
        )
        coloured_heatmap = Image.fromarray(heatmap_rgb).convert("RGB")
        adjustable_overlay = Image.blend(
            explanation["original"].convert("RGB"),
            coloured_heatmap,
            alpha=float(opacity)
        )

        explanation_cols = st.columns(3)
        explanation_panels = [
            ("Original leaf", explanation["original"]),
            ("Activation heatmap", explanation["heatmap"]),
            ("Adjustable Grad-CAM overlay", adjustable_overlay),
        ]
        for column, (caption, image_panel) in zip(explanation_cols, explanation_panels):
            with column:
                st.image(image_panel, caption=caption, use_container_width=True)

        st.caption(
            "Highlighted regions show model sensitivity for the predicted class. "
            "They should not be interpreted as a medical-grade or lesion-segmentation mask."
        )

        with st.expander("Advanced Grad-CAM diagnostics"):
            diagnostic_left, diagnostic_right = st.columns(2)
            with diagnostic_left:
                st.write("**Target convolution layer**", explanation["layer_name"])
                st.write("**Feature map shape**", explanation["feature_shape"])
                st.write("**Selected class index**", int(indices[0]))
            with diagnostic_right:
                st.write("**Heatmap minimum**", round(explanation["heatmap_min"], 5))
                st.write("**Heatmap maximum**", round(explanation["heatmap_max"], 5))
                st.write("**Heatmap standard deviation**", round(explanation["heatmap_std"], 5))
            st.info(
                "A non-empty heatmap confirms the visualization was generated, "
                "but does not alone prove that highlighted regions are faithful to the disease."
            )

    except Exception as error:
        st.warning(
            "Grad-CAM could not be generated for this prediction. "
            f"The CNN prediction remains available. Details: {error}"
        )

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

    ai_plant_health_assistant(label, confidence)

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
        explanation = generate_gradcam_details(image, top_index)
        st.caption(
            f"Grad-CAM uses {explanation['layer_name']} "
            f"(feature map {explanation['feature_shape']}). "
            "Highlighted regions indicate model sensitivity, not a confirmed lesion."
        )
        explanation_cols = st.columns(3)
        explanation_panels = [
            ("Original image", "original"),
            ("Raw activation heatmap", "heatmap"),
            ("Grad-CAM overlay", "overlay"),
        ]
        for column, (caption, key) in zip(explanation_cols, explanation_panels):
            with column:
                st.image(explanation[key], caption=caption, use_container_width=True)
        with st.expander("Grad-CAM diagnostics"):
            st.write({
                "target_layer": explanation["layer_name"],
                "feature_map_shape": explanation["feature_shape"],
                "heatmap_min": round(explanation["heatmap_min"], 5),
                "heatmap_max": round(explanation["heatmap_max"], 5),
                "heatmap_standard_deviation": round(explanation["heatmap_std"], 5),
            })

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
    <div class="brand-lockup">
      <img class="brand-logo" src="https://raw.githubusercontent.com/atharavakadam21-crypto/fyp/main/Plant_Disease_Prediction/assets/plantvision-logo.svg" alt="PlantVision AI leaf logo">
      <div>
        <div class="brand">PlantVision <span>AI</span></div>
        <div class="brand-sub">Intelligent Plant Health</div>
      </div>
    </div>
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
