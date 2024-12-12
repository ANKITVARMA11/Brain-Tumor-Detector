from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Load the trained model
model = load_model('Tumor Detector.h5')

def prepare_image(image):
    # Resizing the image and normalize it as per model's requirement
    image = image.resize(128, 128)  # Using the same size used in training
    image = np.array(image) / 255.0  # Normalization image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    # Getting the image from the POST request
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    
    # Preparing the image
    image = prepare_image(image)
    
    # Making prediction
    prediction = model.predict(image)
    print(f"Prediction: {prediction}")

    result = 'Tumor' if prediction[0][0] > 0.5 else 'No Tumor'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
