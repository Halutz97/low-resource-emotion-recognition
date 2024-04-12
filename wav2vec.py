
import os
import torch
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Processor

# Define the directory where your .wav files are
directory = 'EmoDB/test'

# Load the processor
processor = Wav2Vec2Processor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

# Load the model
model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

# Function to load and preprocess the audio files
def load_and_preprocess_audio(file_path):
    # Load the audio file
    audio_input, _ = sf.read(file_path)
    
    # Preprocess the audio file
    inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=True)

    return inputs

# Load the dataset
dataset = [load_and_preprocess_audio(os.path.join(directory, file)) for file in os.listdir(directory) if file.endswith('.wav')]

# Make predictions
for inputs in dataset:
    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    print(predicted_ids)

"""
A	anger	W	Ärger (Wut)
B	boredom	L	Langeweile
D	disgust	E	Ekel
F	anxiety/fear	A	Angst
H	happiness	F	Freude
S	sadness	T	Trauer
N = neutral version
"""