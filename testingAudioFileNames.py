import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import soundfile as sf
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, AutoProcessor
import numpy as np
from pydub import AudioSegment

letter_to_emotion_EmoDB = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fearful",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}

# Dictionary mapping the last capital letters in the filename to the corresponding emotion.
letter_to_emotion_CremaD = {
    "A": "angry",
    "D": "disgust",
    "F": "fearful",
    "H": "happy",
    "S": "sad",
    "N": "neutral",
}

# Dictionary for the classifier
id2label = {
        "0": "angry",
        "1": "calm",
        "2": "disgust",
        "3": "fearful",
        "4": "happy",
        "5": "neutral",
        "6": "sad",
        "7": "surprised"
    }

def compare_emotion(audio_file, mode):
    # Get the predicted emotion.
    predicted_emotion = "happy" #max(interp, key=interp.get)

    if mode==0:
        # If the predicted emotion is "calm", change it to "neutral", since we only have 7 classes on EmoDB.
        if predicted_emotion == "calm":
            predicted_emotion = "neutral"

        # Extract the last capital letter from the filename.
        last_capital_letter = audio_file.split('/')[-1][9]
        
        print(last_capital_letter)

        # Look up the expected emotion in the dictionary.
        expected_emotion = letter_to_emotion_EmoDB[last_capital_letter]

    elif mode==1:

        # Extract the last capital letter from the filename.
        last_capital_letter = audio_file.split('/')[-1][9]
        print(last_capital_letter)

        # Look up the expected emotion in the dictionary.
        expected_emotion = letter_to_emotion_CremaD[last_capital_letter]

    # Compare the expected emotion with the predicted emotion.
    is_correct = (expected_emotion == predicted_emotion)

    print(f"Expected emotion: {expected_emotion}, Predicted emotion: {predicted_emotion}, Is the result correct? {is_correct}")

    return is_correct, expected_emotion, predicted_emotion

#Define the mode (dataset that we are going to use)
# 0 = EmoDB, 1 = CremaD
mode = 1

# Define the directory where your .wav files are
if mode==0:
    directory = 'EmoDB/test'
elif mode==1:
    # directory = 'CremaD/test/sub'
    directory = "C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\AudioWAVsubset"


files_investigated = 0
score = 0
expected_emotions = []
predicted_emotions = []

# Load the dataset
for file in os.listdir(directory):
    print(file)
    if file.endswith('.wav'):
        files_investigated += 1
        audio_file = file # os.path.join(directory, file)
    #     interp = predict_emotion(audio_file)
    #     print(interp)
        is_correct, expected_emotion, predicted_emotion = compare_emotion(audio_file, mode)
    #     expected_emotions.append(expected_emotion)
    #     predicted_emotions.append(predicted_emotion)
    #     if is_correct:
    #         score += 1