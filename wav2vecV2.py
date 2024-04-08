import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import soundfile as sf
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, AutoProcessor
import numpy as np
from pydub import AudioSegment

# https://github.com/ehcalabres/EMOVoice
# the preprocessor was derived from https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english
# processor1 = AutoProcessor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
# ^^^ no preload model available for this model (above), but the `feature_extractor` works in place

#processor = AutoProcessor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
model1 = AutoModelForAudioClassification.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
model0 = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition") 

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")


# Dictionary mapping the last capital letter in the filename (German) to the corresponding emotion (English).
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

def compare_emotion(interp, audio_file, mode):
    # Get the predicted emotion.
    predicted_emotion = max(interp, key=interp.get)

    if mode==0:
        # If the predicted emotion is "calm", change it to "neutral", since we only have 7 classes on EmoDB.
        if predicted_emotion == "calm":
            predicted_emotion = "neutral"

        # Extract the last capital letter from the filename.
        last_capital_letter = audio_file.split('/')[-1][10]

        # Look up the expected emotion in the dictionary.
        expected_emotion = letter_to_emotion_EmoDB[last_capital_letter]

    elif mode==1:

        # Extract the last capital letter from the filename.
        last_capital_letter = audio_file.split('/')[-1][13]

        # Look up the expected emotion in the dictionary.
        expected_emotion = letter_to_emotion_CremaD[last_capital_letter]

    # Compare the expected emotion with the predicted emotion.
    is_correct = (expected_emotion == predicted_emotion)

    print(f"Expected emotion: {expected_emotion}, Predicted emotion: {predicted_emotion}, Is the result correct? {is_correct}")

    return is_correct, expected_emotion, predicted_emotion


def predict_emotion(audio_file):
    # Load the audio file
    speech, sampling_rate = sf.read(audio_file)

    # Preprocess the audio file
    input_values = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt").input_values

    # Forward pass through the model
    result = model1(input_values)
    
    interp = dict(zip(id2label.values(), list(round(float(i),4) for i in result[0][0])))
    return interp

#Define the mode (dataset that we are going to use)
# 0 = EmoDB, 1 = CremaD
mode = 1

# Define the directory where your .wav files are
if mode==0:
    directory = 'EmoDB/test'
elif mode==1:
    directory = 'CremaD/test/sub'


files_investigated = 0
score = 0
expected_emotions = []
predicted_emotions = []

# Load the dataset
for file in os.listdir(directory):
    if file.endswith('.wav'):
        files_investigated += 1
        audio_file = os.path.join(directory, file)
        interp = predict_emotion(audio_file)
        print(interp)
        is_correct, expected_emotion, predicted_emotion = compare_emotion(interp, audio_file, mode)
        expected_emotions.append(expected_emotion)
        predicted_emotions.append(predicted_emotion)
        if is_correct:
            score += 1

print("All files interpreted")
print(f"Out of {files_investigated} audio files {score} were correct. In total a {(score/files_investigated)*100}% accuracy")

cm = confusion_matrix(expected_emotions, predicted_emotions, labels=list(id2label.values()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id2label.values()))
disp.plot(include_values=True)
plt.show()
