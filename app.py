from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/filtered'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained SVM classifier for soil type classification
with open('svm_classifier.pkl', 'rb') as file:
    svm_classifier = pickle.load(file)

# Load the label encoder for soil type classification
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Load the trained RandomForestRegressor model for nutrient prediction
model_file = 'models/rf_model.pkl'
rf_model = joblib.load(model_file)

# Define parameters for the Gabor filter (for soil type classification)
params = [
    {'ksize': (15, 15), 'sigma': 5, 'theta': np.pi/3, 'lambd': np.pi/4, 'gamma': 0.9, 'psi': 0.8},  # Filter 1
    {'ksize': (15, 15), 'sigma': 4, 'theta': np.pi/2, 'lambd': np.pi/4, 'gamma': 0.9, 'psi': 0.8},  # Filter 2
    {'ksize': (25, 25), 'sigma': 4, 'theta': np.pi/2, 'lambd': np.pi/4, 'gamma': 0.9, 'psi': 0.8},  # Filter 3
    {'ksize': (40, 40), 'sigma': 5, 'theta': np.pi/3, 'lambd': np.pi/4, 'gamma': 0.9, 'psi': 0.8}   # Filter 4
]

# Function to apply Gabor filter on an image for soil type classification
def apply_gabor_filter_on_channel(channel, ksize, sigma, theta, lambd, gamma, psi):
    kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered_channel = cv2.filter2D(channel, -1, kernel)
    return filtered_channel

def apply_gabor_filter_on_image(image, ksize, sigma, theta, lambd, gamma, psi):
    channels = cv2.split(image)
    filtered_channels = [apply_gabor_filter_on_channel(channel, ksize, sigma, theta, lambd, gamma, psi) for channel in channels]
    filtered_image = cv2.merge(filtered_channels)
    return filtered_image

# Function to preprocess image for soil type classification
def preprocess_image_for_classification(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Apply Gabor filters to the image
    filtered_images = [apply_gabor_filter_on_image(img, **param) for param in params]
    
    # Choose one of the filtered images, or combine them as needed for your model
    preprocessed_img = filtered_images[0]
    
    resized_img = cv2.resize(preprocessed_img, (224, 224))
    normalized_img = resized_img.astype('float32') / 255.0
    flattened_img = normalized_img.flatten().reshape(1, -1)
    
    return flattened_img

# Function to predict soil type based on an image
def predict_soil_type(image_path):
    flattened_img = preprocess_image_for_classification(image_path)
    if flattened_img is None:
        return None
    
    predicted_label = svm_classifier.predict(flattened_img)
    predicted_class = label_encoder.inverse_transform(predicted_label)[0]
    
    return predicted_class

# Function to extract RGB features from an image for nutrient prediction
def extract_rgb_features(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_rgb = np.mean(image_rgb, axis=(0, 1))
    std_rgb = np.std(image_rgb, axis=(0, 1))
    hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256]).flatten()
    rgb_features = np.concatenate([mean_rgb, std_rgb, hist_r, hist_g, hist_b])
    return rgb_features

# Function to predict nutrient levels
def predict_nutrients(image_path):
    rgb_features = extract_rgb_features(image_path)
    nutrients_prediction = rf_model.predict(rgb_features.reshape(1, -1))
    return nutrients_prediction.tolist()  # Convert to list for JSON serialization

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No file selected'})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(file_path)

    option = request.form.get('option')
    if option == 'classify':
        predicted_class = predict_soil_type(file_path)
        os.remove(file_path)
        if predicted_class is not None:
            return render_template('soil_result.html', predicted_class=predicted_class)
        else:
            return jsonify({'error': 'Error processing the image'})
    elif option == 'predict':
        nutrients_prediction = predict_nutrients(file_path)
        os.remove(file_path)
        return render_template('result.html', prediction=nutrients_prediction)
    else:
        os.remove(file_path)
        return jsonify({'error': 'Invalid option'})

if __name__ == '__main__':
    app.run(debug=True)
