from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import numpy as np
import skimage.feature as skfeat
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import uuid
import os
import random
from SupportVectorMachine import SVM
import joblib


app = Flask(__name__)

app.config.from_object(__name__)
CORS(app, resources={r"/*":{'origins':'*'}})

if not os.path.exists("uploads"):
    os.makedirs("uploads")

@app.route("/prediction", methods=['GET','POST'])
def main():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join("uploads", filename)
    file.save(filepath)
    
    flatten_features = preprocessing(filepath)
    if flatten_features.size == 0:
        os.remove(filepath)
        return jsonify({'error': 'Error processing image (preprocessing)'}), 500
    
    # Ekstrak fitur hanya dari GLCM
    glcm_features = glcm(flatten_features)
    if not glcm_features or np.nan in glcm_features:
        return jsonify({'error': 'Error processing image'}), 500

    # Pastikan fitur dalam bentuk array 2D
    glcm_features = np.array(glcm_features).reshape(1, -1)

    prediction = predict(glcm_features)

    result = "Positive" if prediction[0] == 1 else "Negative" if prediction[0] == -1 else "Unknown"
    os.remove(filepath)
    print(result)
    return jsonify({'prediction': result})


def predict(data):
    model = joblib.load("svm_rbf_smote_model.joblib")
    scaler = joblib.load("scaler.joblib")
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    return prediction


def preprocessing(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.array([])

    img_resized = cv2.resize(img, (128, 96))

    img_flatten = img_resized.flatten()

    return img_flatten

def glcm(flatten_image):
    if flatten_image.size == 0:
        return [np.nan] * 5

    # Reshape kembali ke (96, 128)
    img_reshaped = flatten_image.reshape(96, 128).astype(np.uint8)

    # Resize ke 64x64 untuk GLCM
    img = cv2.resize(img_reshaped, (64, 64))

    distances = [1]
    angles = [0]

    glcm_matrix = graycomatrix(img,
                               distances=distances,
                               angles=angles,
                               symmetric=True,
                               normed=True)

    contrast = graycoprops(glcm_matrix, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm_matrix, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm_matrix, 'homogeneity')[0, 0]
    energy = graycoprops(glcm_matrix, 'energy')[0, 0]
    correlation = graycoprops(glcm_matrix, 'correlation')[0, 0]

    return [contrast, dissimilarity, homogeneity, energy, correlation]

if __name__ == "__main__":
    app.run(debug=True)
