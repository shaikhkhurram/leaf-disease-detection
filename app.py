import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'plant_model.h5'
CLASS_NAMES_FILE = 'class_names.txt'

# Load the model
model = load_model(MODEL_PATH)

# Load class labels
with open(CLASS_NAMES_FILE, 'r') as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def getResult(image_path):
    try:
        print("[INFO] Image preprocessed")

        # Load and preprocess image
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_label = labels[predicted_class]
        confidence = float(np.max(prediction))
        print(f"[INFO] Prediction raw: {prediction}")
        print(f"[INFO] Predicted label: {predicted_label}")

        # Extract plant and condition
        if '___' in predicted_label:
            plant, condition = predicted_label.split('___')
        else:

            # Fallback if format is unexpected
            parts = predicted_label.split('_')
            plant = parts[0]
            condition = predicted_label
        return {
            'plant': plant,
            'condition': condition.replace('_', ' '),
            'confidence': round(confidence * 100, 2)
        }
    except Exception as e:
        print(f"[ERROR] {e}")
        return {'error': f"Unable to process prediction: {str(e)}"}
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        result = getResult(file_path)
        if 'error' in result:
            return jsonify(result), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True)
