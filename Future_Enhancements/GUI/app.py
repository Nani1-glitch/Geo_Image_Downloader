import sys
import os

# Adjust the path to include the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import torch
import torchvision.transforms as transforms
from custom_model import load_model
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__, template_folder='templates', static_folder='static')

# File paths
geo_model_path = "/Users/nithinrajulapati/Downloads/PROJECT 1/output/geography_aware_model.pth"
moco_model_path = "/Users/nithinrajulapati/Downloads/PROJECT 1/output/moco_model.pth"
model_path = "/Users/nithinrajulapati/Downloads/PROJECT 1/Future_Enhancements/augmented_rf_model.pkl"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
geo_model = load_model(geo_model_path, model_type='custom').to(device)
moco_model = load_model(moco_model_path, model_type='custom').to(device)
rf_model = joblib.load(model_path)

# Define the transformation function
def convert_to_rgb(img):
    return img.convert("RGB") if img.mode != "RGB" else img

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(convert_to_rgb),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = geo_model(image).cpu().numpy()
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    features = extract_features(image)
    prediction = rf_model.predict(features)

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
