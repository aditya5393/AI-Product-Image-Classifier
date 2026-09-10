
import os
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
MODEL_PATH = REPO_ROOT / "model" / "model.keras"

CLASS_NAMES = ["Apparel", "Electronics", "Home"]
IMG_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# PREDICTION
# ============================================================

def predict_product(image):
    if image is None:
        return None

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    original_image = image.copy()

    image_resized = image.resize(IMG_SIZE)
    image_array = np.asarray(image_resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    top_results = [
        {
            "class": CLASS_NAMES[int(i)],
            "confidence": float(predictions[i])
        }
        for i in top_indices
    ]

    predicted_index = int(top_indices[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    # Confidence state
    if confidence >= 0.90:
        state = "HIGH CONFIDENCE"
        state_symbol = "●"
    elif confidence >= 0.70:
        state = "MODERATE CONFIDENCE"
        state_symbol = "◐"
    else:
        state = "LOW CONFIDENCE"
        state_symbol = "○"

    # Build Top-3 HTML
    top3_html = ""

    for rank, item in enumerate(top_results, start=1):
        percentage = item["confidence"] * 100

        top3_html += f"""
        <div class="top-item">
            <div class="top-header">
                <span>#{rank} {item["class"]}</span>
                <span>{percentage:.2f}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{percentage:.2f}%"></div>
            </div>
        </div>
        """

    result_html = f"""
    <div class="result-card">
        <div class="result-label">PREDICTED PRODUCT CATEGORY</div>

        <div class="prediction">
            {predicted_class}
        </div>

        <div class="confidence">
            <span>{state_symbol} {state}</span>
            <strong>{confidence * 100:.2f}%</strong>
        </div>

        <div class="meter-bg">
            <div class="meter-fill"
                 style="width:{confidence * 100:.2f}%">
            </div>
        </div>

        <div class="section-title">
            TOP-3 PREDICTIONS
        </div>

        {top3_html}
    </div>
    """

    return original_image, result_html


# ============================================================
# PREMIUM UI
# ============================================================

CSS = """
body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background:
        radial-gradient(circle at 20% 20%, rgba(70, 80, 180, 0.25), transparent 35%),
        radial-gradient(circle at 80% 80%, rgba(150, 50, 180, 0.20), transparent 35%),
        #05050b;
    color: white;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.hero {
    text-align: center;
    padding: 35px 20px 20px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
    letter-spacing: 1px;
}

.hero p {
    color: #b8b8c7;
    font-size: 17px;
}

.panel {
    background: rgba(20, 20, 32, 0.78);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 25px;
    backdrop-filter: blur(15px);
}

.result-card {
    padding: 10px;
}

.result-label {
    font-size: 12px;
    color: #9b9bad;
    letter-spacing: 2px;
}

.prediction {
    font-size: 42px;
    font-weight: 800;
    margin: 12px 0 18px;
}

.confidence {
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
    color: #c8c8d5;
}

.confidence strong {
    color: white;
}

.meter-bg,
.bar-bg {
    width: 100%;
    height: 10px;
    background: rgba(255,255,255,0.10);
    border-radius: 20px;
    overflow: hidden;
}

.meter-fill,
.bar-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #6d5dfc, #b65cff);
}

.section-title {
    margin-top: 30px;
    margin-bottom: 18px;
    font-size: 12px;
    letter-spacing: 2px;
    color: #9b9bad;
}

.top-item {
    margin-bottom: 18px;
}

.top-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 7px;
}

.footer {
    text-align: center;
    color: #777789;
    padding: 30px;
    font-size: 13px;
}
"""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="AI Product Image Classifier"
) as demo:

    gr.HTML("""
    <div class="hero">
        <h1>AI Product Image Classifier</h1>
        <p>
            Explainable • Confidence-Aware • Deep Learning
        </p>
    </div>
    """)

    with gr.Row():

        with gr.Column(scale=1, elem_classes="panel"):

            gr.Markdown("### Upload Product Image")

            input_image = gr.Image(
                type="pil",
                label="Product Image"
            )

            predict_button = gr.Button(
                "⚡ ANALYZE PRODUCT",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1, elem_classes="panel"):

            output_image = gr.Image(
                label="Analyzed Image",
                type="pil"
            )

            result = gr.HTML(
                """
                <div class="result-card">
                    <div class="result-label">
                        READY FOR ANALYSIS
                    </div>
                    <div class="prediction">
                        Upload an image
                    </div>
                    <p style="color:#999;">
                        The AI model will classify the product
                        into Apparel, Electronics, or Home.
                    </p>
                </div>
                """
            )

    predict_button.click(
        fn=predict_product,
        inputs=input_image,
        outputs=[output_image, result]
    )

    gr.HTML("""
    <div class="footer">
        AI Product Image Classifier<br>
        Developed by Aditya ♥️
    </div>
    """)


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=gr.themes.Base()
    )
