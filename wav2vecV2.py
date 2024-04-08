import torch
import os
import soundfile as sf
from transformers import Wav2Vec2Processor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import numpy as np
from pydub import AudioSegment

# https://github.com/ehcalabres/EMOVoice
# the preprocessor was derived from https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english
# processor1 = AutoProcessor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
# ^^^ no preload model available for this model (above), but the `feature_extractor` works in place

model1 = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition") 

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")


# Dictionary mapping the last capital letter in the filename (German) to the corresponding emotion (English).
letter_to_emotion = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fearful",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}

def compare_emotion(interp, audio_file):
    # Get the predicted emotion.
    predicted_emotion = max(interp, key=interp.get)

    # If the predicted emotion is "calm", change it to "neutral", since we only have 7 classes on EmoDB.
    if predicted_emotion == "calm":
        predicted_emotion = "neutral"

    # Extract the last capital letter from the filename.
    last_capital_letter = audio_file.split('/')[-1][10]
    print(last_capital_letter)

    # Look up the expected emotion in the dictionary.
    expected_emotion = letter_to_emotion[last_capital_letter]

    # Compare the expected emotion with the predicted emotion.
    is_correct = (expected_emotion == predicted_emotion)

    print(f"Expected emotion: {expected_emotion}, Predicted emotion: {predicted_emotion}, Is the result correct? {is_correct}")

    return is_correct


def predict_emotion(audio_file):
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_frame_rate(16000)
    sound_array = np.array(sound.get_array_of_samples())
    # this model is VERY SLOW, so best to pass in small sections that contain 
    # emotional words from the transcript. like 10s or less.
    # how to make sub-chunk  -- this was necessary even with very short audio files 
    # test = torch.tensor(input.input_values.float()[:, :100000])

    input = feature_extractor(
        raw_speech=sound_array,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt")

    result = model1.forward(input.input_values.float())
    # making sense of the result 
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
    interp = dict(zip(id2label.values(), list(round(float(i),4) for i in result[0][0])))
    return interp


# Define the directory where your .wav files are
directory = 'EmoDB/test'

files_investigated = 0
score = 0

# Load the dataset
for file in os.listdir(directory):
    if file.endswith('.wav'):
        files_investigated += 1
        audio_file = os.path.join(directory, file)
        interp = predict_emotion(audio_file)
        print(interp)
        is_correct = compare_emotion(interp, audio_file)
        if is_correct:
            score += 1

print("All files interpreted")
print(f"Out of {files_investigated} audio files {score} were correct. In total a {(score/files_investigated)*100}% accuracy")

"""
A	anger	W	Ärger (Wut)
B	boredom	L	Langeweile
D	disgust	E	Ekel
F	anxiety/fear	A	Angst
H	happiness	F	Freude
S	sadness	T	Trauer
N = neutral version
"""