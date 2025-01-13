# from flask import Flask, request, jsonify, render_template
# from tensorflow.keras.preprocessing import image #type:ignore
# import numpy as np
# import os
# from tensorflow.keras.models import load_model #type:ignore

# app = Flask(__name__)
# model = load_model('models/disease_diagnosis_model.h5')

# # Route for uploading image and predicting disease
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # Save the image temporarily
#     img_path = os.path.join('static/uploaded_images', file.filename)
#     file.save(img_path)

#     # Preprocess the image
#     img = image.load_img(img_path, target_size=(150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     # Prediction
#     prediction = model.predict(img_array)
#     result = 'Pneumonia' if prediction[0] > 0.5 else 'Healthy'

#     # Return the result back to index.html
#     return render_template('index.html', result=result)

# # Route for displaying the upload form
# @app.route('/')
# def index():
#     return render_template('index.html', result=None)

# if __name__ == '__main__':
#     app.run(debug=False,host='0.0.0.0')


from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model  # type: ignore

app = Flask(__name__)

# Model file path
MODEL_PATH = 'models/disease_diagnosis_model.h5'

# Google Drive file ID for the model
FILE_ID = '1J1sjDUQPbRmRZ4ZfrDudKQvZZ3B-6LIj'  # Replace with your Google Drive file ID
DOWNLOAD_URL = f'https://drive.google.com/uc?id={FILE_ID}'

# Check if model exists locally, otherwise download it
if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Downloading from Google Drive...")
    os.makedirs('models', exist_ok=True)
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    print("Model downloaded successfully!")

# Load the model
model = load_model(MODEL_PATH)

# Route for uploading image and predicting disease
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image temporarily
    img_path = os.path.join('static/uploaded_images', file.filename)
    file.save(img_path)

    # Preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Prediction
    prediction = model.predict(img_array)
    result = 'Pneumonia' if prediction[0] > 0.5 else 'Healthy'

    # Return the result back to index.html
    return render_template('index.html', result=result)

# Route for displaying the upload form
@app.route('/')
def index():
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')