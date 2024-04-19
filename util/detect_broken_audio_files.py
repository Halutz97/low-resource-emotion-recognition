import librosa
import os
import wave

def create_silent_wav(input_file, output_file):
    # Open the original file
    with wave.open(input_file, 'rb') as in_wave:
        # Number of frames to read in one chunk
        n_frames = in_wave.getnframes()
        # Read audio data
        audio_data = in_wave.readframes(n_frames)

        # Parameters
        n_channels = in_wave.getnchannels()
        sample_width = in_wave.getsampwidth()
        framerate = in_wave.getframerate()

        # Create a silent audio data
        silent_data = bytearray(len(audio_data))

    # Write to a new WAV file
    with wave.open(output_file, 'wb') as out_wave:
        out_wave.setnchannels(n_channels)
        out_wave.setsampwidth(sample_width)
        out_wave.setframerate(framerate)
        out_wave.writeframes(silent_data)

def check_audio_files(directory, threshold=0.01):
    files = os.listdir(directory)
    # silent_files = []
    num_silent_files = 0
    num_files_checked = 0
    for file in files:
        # if num_files_checked >= 10:
            # break
        path = os.path.join(directory, file)
        try:
            # Load the audio file
            audio, sr = librosa.load(path, sr=None)
            # Calculate the absolute maximum of the audio signal
            max_amplitude = max(abs(audio))
            # Check if the maximum amplitude is below the threshold
            if max_amplitude < threshold:
                # silent_files.append(file)
                num_silent_files += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")
        num_files_checked += 1
    print(str(num_silent_files) + " out of " + str(num_files_checked) + " files are silent.")

def main():
    # Use the function
    # silent_files = check_audio_files('path_to_your_audio_files')
    directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\test\test_audio"
    # directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\dev\dev_audio"
    
    # file_to_mute = os.path.join(directory, "dia0000_utt01.wav")
    # muted_file = os.path.join(directory, "dia0000_utt01_silent.wav")
    # create_silent_wav(file_to_mute, os.path.join(directory, muted_file))

    check_audio_files(directory)

if __name__ == "__main__":
    main()