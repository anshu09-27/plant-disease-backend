"""
app.py — Flask backend for AI Plant Disease Diagnosis

Endpoints:
  GET  /health           -> simple liveness check
  GET  /classes          -> list of supported disease classes
  POST /predict           -> multipart/form-data image upload, returns
                              predicted class, confidence, and treatment info

Run locally:
  pip install -r requirements.txt
  python app.py

Deploy on Render.com:
  - Build command: pip install -r requirements.txt
  - Start command: gunicorn app:app
"""

import io
import os
import logging


import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

from disease_info import CLASS_NAMES, get_disease_info, get_ai_enriched_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow requests from the Flutter app (mobile + web)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model/plant_disease_model.tflite")
IMG_SIZE = int(os.environ.get("IMG_SIZE", 224))  # MobileNetV2 default input size
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Model loading (TFLite)
# ---------------------------------------------------------------------------
interpreter = None
input_details = None
output_details = None


def load_model():
    """Load the TFLite model once at startup. Uses tflite-runtime if available
    (lighter, preferred for Render's free tier); falls back to full TensorFlow."""
    global interpreter, input_details, output_details

    if not os.path.exists(MODEL_PATH):
        logger.warning(
            "Model file not found at %s — /predict will return 503 until it's present.",
            MODEL_PATH,
        )
        return

    try:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter_cls = tflite.Interpreter
        except ImportError:
            import tensorflow as tf
            interpreter_cls = tf.lite.Interpreter

        interpreter = interpreter_cls(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        interpreter = None


load_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


CHANNEL_ORDER = os.environ.get("CHANNEL_ORDER", "RGB").upper()  # "RGB" or "BGR"


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Resize and normalize the image to a 0-1 float array shaped for model input.
    Quantization (if the model needs it) is applied separately in run_inference,
    since it depends on the model's input scale/zero_point.

    CHANNEL_ORDER env var lets us test whether the model expects BGR (common if
    it was trained using OpenCV-loaded images) instead of RGB (PIL's default)."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32)
    arr = (arr / 127.5) - 1.0  # MobileNetV2 standard preprocessing: -1 to 1
    if CHANNEL_ORDER == "BGR":
        arr = arr[:, :, ::-1]  # reverse channel order RGB -> BGR
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr


def run_inference(input_array: np.ndarray):
    """Run the TFLite interpreter and return real-valued (dequantized) output.

    Handles both float32 models and int8-quantized models transparently:
    - If the model's input is int8, the 0-1 float image is quantized using the
      model's input scale/zero_point before feeding it in.
    - If the model's output is int8, the raw int8 output is dequantized back
      to real probability values using the model's output scale/zero_point.
    """
    in_detail = input_details[0]
    out_detail = output_details[0]

    # --- Prepare input ---
    if in_detail["dtype"] == np.int8:
        scale, zero_point = in_detail["quantization"]
        quantized = input_array / scale + zero_point
        quantized = np.clip(np.round(quantized), -128, 127).astype(np.int8)
        interpreter.set_tensor(in_detail["index"], quantized)
    else:
        interpreter.set_tensor(in_detail["index"], input_array.astype(in_detail["dtype"]))

    interpreter.invoke()
    raw_output = interpreter.get_tensor(out_detail["index"])[0]  # drop batch dim

    # --- Convert output back to real probabilities ---
    if out_detail["dtype"] == np.int8:
        scale, zero_point = out_detail["quantization"]
        real_output = scale * (raw_output.astype(np.float32) - zero_point)
    else:
        real_output = raw_output.astype(np.float32)

    return real_output


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": interpreter is not None,
            "num_classes": len(CLASS_NAMES),
        }
    )


@app.route("/classes", methods=["GET"])
def classes():
    return jsonify({"classes": CLASS_NAMES, "count": len(CLASS_NAMES)})


@app.route("/predict", methods=["POST"])
def predict():
    if interpreter is None:
        return jsonify(
            {"error": "Model not loaded on server. Check MODEL_PATH deployment config."}
        ), 503

    if "image" not in request.files:
        return jsonify({"error": "No 'image' file part in request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}
        ), 400

    try:
        file_bytes = file.read()
        input_array = preprocess_image(file_bytes)
        predictions = run_inference(input_array)

        top_index = int(np.argmax(predictions))
        confidence = float(predictions[top_index])
        class_name = CLASS_NAMES[top_index]
        info = get_disease_info(class_name)

# Try to enrich the treatment text with AI-generated detail;
# falls back to the static text automatically if it fails
        enriched_treatment = get_ai_enriched_info(info["display_name"], info["treatment"])

        response = {
            "class_name": class_name,
            "display_name": info["display_name"],
            "healthy": info["healthy"],
            "confidence": round(confidence, 4),
            "treatment": enriched_treatment,
        }
        return jsonify(response)

    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return jsonify({"error": "Failed to process image.", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)