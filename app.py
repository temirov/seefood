from flask import Flask, render_template, request
from transformers import pipeline
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load Hugging Face image-classification pipeline once at startup
image_classification_pipeline = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224",
    device=0  # or -1 for CPU-only
)

@app.route("/", methods=["GET"])
def display_upload_form():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def handle_image_upload_and_prediction():
    uploaded_file_storage = request.files.get("image")
    if uploaded_file_storage is None or uploaded_file_storage.filename == "":
        return render_template("index.html", error_message="Please choose an image file.")
    
    # Read image bytes and load into PIL
    uploaded_image_bytes = uploaded_file_storage.read()
    pil_image = Image.open(io.BytesIO(uploaded_image_bytes)).convert("RGB")
    
    # Run Hugging Face inference
    prediction_results = image_classification_pipeline(pil_image, top_k=5)
    
    # Decide hot dog vs not hot dog
    is_hot_dog_detected = any(
        prediction_entry["label"].lower().replace("-", "").replace(" ", "") == "hotdog"
        for prediction_entry in prediction_results
    )
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
