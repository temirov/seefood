from flask import Flask, render_template, request
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as mobilenet_preprocess_input,
    decode_predictions as mobilenet_decode_predictions
)
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)

# Load pretrained classifier once at startup
pretrained_image_classifier = MobileNetV2(weights="imagenet")
pretrained_image_classifier.trainable = False

@app.route("/", methods=["GET"])
def display_upload_form():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def handle_image_upload_and_prediction():
    uploaded_file_storage = request.files.get("image")
    if uploaded_file_storage is None or uploaded_file_storage.filename == "":
        return render_template("index.html", error_message="Please choose an image file.")
    
    # Read image bytes and prepare for model
    uploaded_image_bytes = uploaded_file_storage.read()
    pil_image = Image.open(io.BytesIO(uploaded_image_bytes)).convert("RGB")
    resized_image = pil_image.resize((224, 224))
    image_array = np.array(resized_image, dtype=np.float32)
    image_array = mobilenet_preprocess_input(image_array)
    batched_image_array = np.expand_dims(image_array, axis=0)
    
    # Run inference
    prediction_probabilities = pretrained_image_classifier.predict(batched_image_array)
    decoded_prediction_list = mobilenet_decode_predictions(prediction_probabilities, top=5)[0]
    
    # Decide hot dog vs not hot dog
    is_hot_dog_detected = any(prediction_label == "hotdog" for (_, prediction_label, _) in decoded_prediction_list)
    classification_result_text = "🌭 Hot Dog!" if is_hot_dog_detected else "🚫 Not Hot Dog"
    
    # Encode image for inline display
    encoded_image_data = base64.b64encode(uploaded_image_bytes).decode("utf-8")
    
    return render_template(
        "result.html",
        classification_result_text=classification_result_text,
        encoded_image_data=encoded_image_data
    )

if __name__ == "__main__":
    app.run(debug=True)
