import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import mimetypes

def load_model(model_filename):
    try:
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except Exception as e:
        # Handle the error (e.g., log the error and return a default model)
        print(f"Error loading model: {str(e)}")
        return None

# Load models
decision_tree_model = load_model('decision_tree_model.pkl')
svm_classifier_model = load_model('svm_classifier_model.pkl')
random_forest_model = load_model('rfc_model.pkl')

app = Flask(__name__)

# Directory to store uploaded .wav files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# import numpy as np

import numpy as np

# Function to calculate audio features
def calculate_audio_features(wav_file_path):
    try:
        audio_data, sample_rate = librosa.load(wav_file_path)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)

        # Find the maximum length of all features
        max_length = max(chroma.shape[1], mfcc.shape[1], spectral_centroid.shape[1], spectral_contrast.shape[1], zero_crossing_rate.shape[1])

        # Create arrays to store features
        chroma_data = np.pad(chroma[0], (0, max_length - chroma.shape[1]))
        mfcc_data = np.pad(mfcc[0], (0, max_length - mfcc.shape[1]))
        spectral_centroid_data = np.pad(spectral_centroid[0], (0, max_length - spectral_centroid.shape[1]))
        spectral_contrast_data = np.pad(spectral_contrast[0], (0, max_length - spectral_contrast.shape[1]))
        zero_crossing_rate_data = np.pad(zero_crossing_rate[0], (0, max_length - zero_crossing_rate.shape[1]))

        # Combine the features into a list or tuple
        features = (chroma_data, mfcc_data, spectral_centroid_data, spectral_contrast_data, zero_crossing_rate_data)

        return features
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return None


# Apply Min-Max scaling for numeric columns
def apply_minmax_scaling(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data



@app.route('/')
def hello():
    return 'Hello, World!'



@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
     
        try:
            features = calculate_audio_features(file)

            if features is not None:
                chroma_data, mfcc_data, spectral_centroid_data, spectral_contrast_data, zero_crossing_rate_data = features
                # Convert NumPy arrays to lists for JSON serialization
                chroma_data = chroma_data.tolist()
                mfcc_data = mfcc_data.tolist()
                spectral_centroid_data = spectral_centroid_data.tolist()
                spectral_contrast_data = spectral_contrast_data.tolist()
                zero_crossing_rate_data = zero_crossing_rate_data.tolist()

                # Calculate the mean for each feature
                chroma_mean = np.mean(chroma_data)
                mfcc_mean = np.mean(mfcc_data)
                spectral_centroid_mean = np.mean(spectral_centroid_data)
                spectral_contrast_mean = np.mean(spectral_contrast_data)
                zero_crossing_rate_mean = np.mean(zero_crossing_rate_data)

                # Apply Min-Max scaling to the mean values
                features_to_predict = [zero_crossing_rate_mean,chroma_mean, spectral_centroid_mean]
                # features_to_predict = [chroma_mean]
                features_to_predict2 = [zero_crossing_rate_mean,chroma_mean, spectral_centroid_mean]
                features_to_predict3= [zero_crossing_rate_mean, spectral_contrast_mean,spectral_centroid_mean]
                features_to_predict = np.array(features_to_predict).reshape(1, -1)  # Reshape to match the model's input
                features_to_predict2 = np.array(features_to_predict2).reshape(1, -1)  # Reshape to match the model's input
                features_to_predict3 = np.array(features_to_predict3).reshape(1, -1)  # Reshape to match the model's input

                # print("Chromadaat=",chroma_data)
                # Predict using the loaded models
                try:
                    decision_tree_prediction = decision_tree_model.predict(features_to_predict)
                    svm_prediction = svm_classifier_model.predict(features_to_predict2)
                    random_forest_prediction = random_forest_model.predict(features_to_predict3)
               
                except Exception as prediction_error:
                    return jsonify({'error': f'Error in making predictions: {str(prediction_error)}'}), 400

          

            # Return the predictions for all models
             # Return the predictions for all models
                return jsonify({'message': 'Cry reason predicted',
                                'predictions': {
                                    'DecisionTree': int(decision_tree_prediction[0]),
                                    'SVMClassifier': int(svm_prediction[0]),
                                    'RandomForest': int(random_forest_prediction[0])
                                }})

        except Exception as feature_extraction_error:
            return jsonify({'error': f'Error in feature extraction: {str(feature_extraction_error)}'}), 400

    return jsonify({'error': 'Internal server error'}, 500)

   
if __name__ == '__main__':
    app.run()
