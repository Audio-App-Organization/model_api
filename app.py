import csv
import os
import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from flask import Flask, request, jsonify

# Load your models
binary_model = tf.keras.models.load_model('Binary_model.h5')
multiclass_model = tf.keras.models.load_model('Multiclass_model.h5')


# Function to analyze audio by windowing
def analyze_audio(audio_file, window_size_ms=20, hop_size_ms=10):
    sample_rate, audio_data = wavfile.read(audio_file)

    # Convert milliseconds to samples
    window_size_samples = int((window_size_ms / 1000) * sample_rate)
    hop_size_samples = int((hop_size_ms / 1000) * sample_rate)

    signal_length = len(audio_data) / sample_rate

    # Initialize variables to store statistics
    average_amplitude = []
    average_frequency = []

    # Iterate over the audio data with overlapping windows
    for i in range(0, len(audio_data) - window_size_samples, hop_size_samples):
        # Extract the current window
        window = audio_data[i:i + window_size_samples]

        # Skip if the window is too short
        if len(window) < window_size_samples:
            continue

        # Calculate the average amplitude for this window
        avg_amp = np.mean(window)
        average_amplitude.append(avg_amp)

        # Calculate the average frequency for this window
        fft_result = np.fft.fft(window)
        frequencies = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
        fft_magnitudes = np.abs(fft_result)
        max_magnitude_index = np.argmax(fft_magnitudes)

        # Skip if the FFT result is too short
        if max_magnitude_index >= len(frequencies):
            continue

        max_frequency = frequencies[max_magnitude_index]
        average_frequency.append(max_frequency)

    return (
        np.mean(average_amplitude),
        np.mean(average_frequency)
    )


# Function to predict input based on features
def predictInput(audio_file):
    # Initial values for final output variables
    predicted_label_binary = "could_not_predict"
    predicted_label_multi = -1

    # Analyze audio to get frequency and amplitude
    frequency, amplitude = analyze_audio(audio_file)

    input_data = np.array([[amplitude, frequency]])

    # Prepare data to be fed into model
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    label_names = {
        0: 'Healthy',
        1: 'Pathology',
    }

    # Predict and calculate label
    prediction_binary = binary_model.predict(input_data)
    predicted_label_binary = np.argmax(prediction_binary, axis=1)[0]
    predicted_label_binary = label_names[predicted_label_binary]

    # If the label is Pathology, Execute multiclass model - Same as binary model
    if predicted_label_binary == 'Pathology':
        input_data_multi = np.array([[amplitude, frequency]])

        scaler = StandardScaler()
        input_data_multi = scaler.fit_transform(input_data_multi)

        prediction_multi = multiclass_model.predict(input_data)
        predicted_label_multi = np.argmax(prediction_multi, axis=1)[0]

    # Pack predictions into dictionary
    final_prediction = {
        "binary_prediction": predicted_label_binary,
        "multi_class_prediction": int(predicted_label_multi)
    }

    return final_prediction


# Web API
app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def predict():
    audio_file = request.files['file']

    if not audio_file or not audio_file.filename.endswith('.wav'):
        return jsonify({"message": "Invalid or no WAV file in request"}), 400

    script_dir = os.path.dirname(os.path.abspath(__file__))

    audio_file_path = os.path.join(script_dir, audio_file.filename)

    audio_file.save(audio_file_path)
    prediction = predictInput(audio_file_path)
    os.remove(audio_file_path)
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
