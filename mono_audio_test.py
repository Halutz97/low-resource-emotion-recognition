import librosa
import os
import soundfile as sf

# List of audio file paths
audio_files = ["dia10_utt1.wav", "dia10_utt2.wav", "dia10_utt3.wav"]
directory = "MELD_preprocess_test/MELD_fine_tune_v1_test_data"

for i, audio_file in enumerate(audio_files):
    # Load audio file and convert to mono
    file_to_load_path = os.path.join(directory, audio_file)
    y, sr = librosa.load(file_to_load_path, sr=16000, mono=True)

    # Save the mono audio
    sf.write(f"mono_audio{i+1}.wav", y, sr)