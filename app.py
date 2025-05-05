"""
Package app implements a Flask web application that classifies uploaded images
as hot dog or not hot dog using a Hugging Face vision pipeline.
"""

import pillow_heif

pillow_heif.register_heif_opener()

import io
import base64
import re
from PIL import Image
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

image_classification_pipeline = pipeline(
    task="image-classification", model="google/vit-base-patch16-224", device=0
)


def is_image_hot_dog(pil_image: Image.Image) -> bool:
    """
    Determine whether the provided image contains a hot dog based on the
    top-ten predictions from the Hugging Face image-classification pipeline.
    """
    prediction_entries = image_classification_pipeline(pil_image, top_k=10)
    for prediction_entry in prediction_entries:
        normalized_label = re.sub(r"[^a-z0-9]", "", prediction_entry["label"].lower())
        if "hotdog" in normalized_label:
            return True
    return False


@app.route("/", methods=["GET"])
def display_upload_form():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def handle_image_upload_and_prediction():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        return render_template(
            "index.html", error_message="Please choose an image file."
        )
    original_bytes = uploaded_file.read()
    pil_image = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    hot_dog_detected = is_image_hot_dog(pil_image)
    classification_result_text = "🌭 Hot Dog!" if hot_dog_detected else "🚫 Not Hot Dog"
    jpeg_buffer = io.BytesIO()
    pil_image.save(jpeg_buffer, format="JPEG")
    jpeg_bytes = jpeg_buffer.getvalue()
    encoded_image_data = base64.b64encode(jpeg_bytes).decode("utf-8")
    return render_template(
        "result.html",
        classification_result_text=classification_result_text,
        encoded_image_data=encoded_image_data,
    )


if __name__ == "__main__":
    app.run(debug=True)
