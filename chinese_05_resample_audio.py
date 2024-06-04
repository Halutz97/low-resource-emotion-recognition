# This script checks the sampling rate of the audio files that were extracted in the previous step.
# Import necessary libraries
import os
import numpy as np
import pandas as pd
import soundfile as sf
# import librosa
import torchaudio
from torchaudio.transforms import Resample


# Define the source directory
source_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\AudioWAV_testing"
destination_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\AudioWAV_resampled"
# Create the destination directory if it does not exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)
    print(f"Created directory: {destination_directory}")

# Get the list of audio files
audio_files = os.listdir(source_directory)

# Check the sampling rate of each audio file
expected_sr = True
for file in audio_files:
    audio_file_path = os.path.join(source_directory, file)
    # Load only the metadata of the audio file
    info = sf.info(audio_file_path)
    if info.samplerate != 44100:
        print(f"The sampling rate of {file} is {info.samplerate} Hz.")
        expected_sr = False
        break
    # print("Sampling Rate:", info.samplerate)
    # audio, sr = librosa.load(audio_file_path, sr=None)  # `sr=None` ensures original sr is used
    # print("Sampling Rate:", sr)
if expected_sr:
    print("All audio files have a sampling rate of 44100 Hz.")
else:
    print("There is a problem!")

for file in audio_files:
    audio_file_path = os.path.join(source_directory, file)
    waveform, sr = torchaudio.load(audio_file_path)
    resampler = Resample(orig_freq=sr, new_freq=16000)
    audio_resampled = resampler(waveform)
    # audio_resampled_np = audio_resampled.numpy()
    destination_file_path = os.path.join(destination_directory, file)
    torchaudio.save(destination_file_path, audio_resampled, 16000)

print("Resampling complete.")

# Check the sampling rate of each audio file

resampled_audio_files = os.listdir(destination_directory)
expected_sr = True
for file in resampled_audio_files:
    audio_file_path = os.path.join(destination_directory, file)
    # Load only the metadata of the audio file
    info = sf.info(audio_file_path)
    if info.samplerate != 16000:
        print(f"The sampling rate of {file} is {info.samplerate} Hz.")
        expected_sr = False
        break
    # print("Sampling Rate:", info.samplerate)
    # audio, sr = librosa.load(audio_file_path, sr=None)  # `sr=None` ensures original sr is used
    # print("Sampling Rate:", sr)
if expected_sr:
    print("All audio files have been resampled to 16000 Hz.")


