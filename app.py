from flask import Flask, request, jsonify
import os
import tensorflow as tf
import numpy as np
import librosa

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('model_audio.h5')

# Function to extract MFCC features from an audio file
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Function to predict emotion from an audio file
def predict_emotion(file_path):
    mfcc_features = extract_mfcc(file_path)
    mfcc_features = mfcc_features.reshape(1, 40, 1)
    predictions = model.predict(mfcc_features)
    predicted_emotion_class = np.argmax(predictions)
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Pleasant Surprise', 'Sadness', 'Neutral']
    return emotion_labels[predicted_emotion_class]

# Route to handle audio file upload and prediction
@app.route('/predict-emotion', methods=['POST'])
def predict_emotion_api():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    file = request.files['audio']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    predicted_emotion = predict_emotion(file_path)
    os.remove(file_path)
    return jsonify({'emotion': predicted_emotion})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, host='0.0.0.0', port=5000)
