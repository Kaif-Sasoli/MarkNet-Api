from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from models.crnn import CRNNModel, CRNNPredictor
from models.yolo_detect import YOLODetector


# GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default Characters
DEFAULT_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-/%:|, "


# Load CRNN
crnn_model = CRNNModel(num_classes=len(DEFAULT_CHARS))
checkpoint = torch.load("weights/crnn_checkpoint_cropped2.5v_40.pth", map_location=DEVICE)
crnn_model.load_state_dict(checkpoint["model_state"])


crnn_predictor = CRNNPredictor(crnn_model, DEFAULT_CHARS, device=DEVICE)

# Load YOLO
yolo_detector = YOLODetector("weights/yolo_model.pt")

# Initialize Flask
app = Flask(__name__)


# Test Route
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Flask API is working!"})

# Predcit
@app.route("/predict_text", methods=["POST"])
def predict_text():
    """
    Accepts image file, detects text regions, predicts text using CRNN
    Returns JSON: { "detections": [ {"class": cls, "text": pred_text}, ... ] }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")


    # Detect text regions using YOLO
    crops = yolo_detector.detect_and_crop(img)

    # Predict text for each crop
    results = []
    for crop_img, cls in crops:
        text = crnn_predictor.predict(crop_img, cls)
        results.append({
            "class": cls,
            "text": text
        })

    return jsonify({"detections": results})



# ---- START SERVER ----
if __name__ == "__main__":
    app.run(debug=True)
