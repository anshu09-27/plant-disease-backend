"""
diagnose.py — standalone test script (does NOT need the Flask server running)

Loads the TFLite model directly, runs one image through it in both RGB and
BGR channel order, and prints the top 5 predicted classes with their real
(dequantized) confidence for each. This helps us see whether the model is
"close" in one ordering vs totally lost in the other, using more signal
than just the single top prediction.

Usage:
    python diagnose.py "C:\\path\\to\\image.jpg"
"""

import sys
import numpy as np
from PIL import Image
import tensorflow as tf

from disease_info import CLASS_NAMES

MODEL_PATH = "model/plant_disease_model.tflite"
IMG_SIZE = 224


def load_and_prep(path, channel_order):
    image = Image.open(path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if channel_order == "BGR":
        arr = arr[:, :, ::-1]
    return np.expand_dims(arr, axis=0)


def run(path, channel_order, interpreter, input_details, output_details):
    arr = load_and_prep(path, channel_order)

    in_detail = input_details[0]
    out_detail = output_details[0]

    if in_detail["dtype"] == np.int8:
        scale, zero_point = in_detail["quantization"]
        q = np.clip(np.round(arr / scale + zero_point), -128, 127).astype(np.int8)
        interpreter.set_tensor(in_detail["index"], q)
    else:
        interpreter.set_tensor(in_detail["index"], arr.astype(in_detail["dtype"]))

    interpreter.invoke()
    raw = interpreter.get_tensor(out_detail["index"])[0]

    if out_detail["dtype"] == np.int8:
        scale, zero_point = out_detail["quantization"]
        real = scale * (raw.astype(np.float32) - zero_point)
    else:
        real = raw.astype(np.float32)

    top5_idx = np.argsort(real)[::-1][:5]
    print(f"\n--- {channel_order} ---")
    for i in top5_idx:
        print(f"  {CLASS_NAMES[i]:55s}  {real[i]:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python diagnose.py "C:\\path\\to\\image.jpg"')
        sys.exit(1)

    image_path = sys.argv[1]

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Testing image: {image_path}")
    run(image_path, "RGB", interpreter, input_details, output_details)
    run(image_path, "BGR", interpreter, input_details, output_details)