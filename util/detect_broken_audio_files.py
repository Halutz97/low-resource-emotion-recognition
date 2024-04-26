import librosa
import os
import wave
import shutil

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

def check_silent_audio_files(directory, copy_directory, threshold=0.01, delete_files=False, copy_files=True):
    if not os.path.exists(copy_directory):
        os.makedirs(copy_directory)
    files = os.listdir(directory)
    silent_files_list = []
    num_silent_files = 0
    num_files_checked = 0
    files_deleted = 0
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
                silent_files_list.append(file)
                num_silent_files += 1
                if copy_files:
                    shutil.copy(path, copy_directory)
                if delete_files:
                    os.remove(path)
                    files_deleted += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")
        num_files_checked += 1
    print(str(num_silent_files) + " out of " + str(num_files_checked) + " files are silent.")
    print("List of silent files:")
    print(silent_files_list)
    # if copy_files:
        # Copy silent files to a
    if delete_files:
        print("Deleted " + str(files_deleted) + " files.")

def check_short_audio_files(directory, copy_directory, cut_lower=0.55, cut_upper=0.65, lower_limit=True, copy_files=True, delete_files=False):
    if not os.path.exists(copy_directory):
        os.makedirs(copy_directory)
    files = os.listdir(directory)
    num_files_checked = 0
    num_files_deleted = 0
    for file in files:
        path = os.path.join(directory, file)
        try:
            audio, sr = librosa.load(path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            if (duration < cut_upper) and (lower_limit==False or (duration >= cut_lower)):
                if copy_files:
                    shutil.copy(path, copy_directory)
                if delete_files:
                    os.remove(path)
                    num_files_deleted += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")
        num_files_checked += 1
    print("Checked " + str(num_files_checked) + " files.")
    if delete_files:
        print("Deleted " + str(num_files_deleted) + " files.")

def check_long_audio_files(directory, copy_directory, cut_lower=10, cut_upper=12, upper_limit=True, copy_files=True, delete_files=False):
    if not os.path.exists(copy_directory):
        os.makedirs(copy_directory)
    files = os.listdir(directory)
    num_files_checked = 0
    num_files_deleted = 0
    for file in files:
        path = os.path.join(directory, file)
        try:
            audio, sr = librosa.load(path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            if (duration > cut_lower) and (upper_limit==False or (duration <= cut_upper)):
                if copy_files:
                    shutil.copy(path, copy_directory)
                if delete_files:
                    os.remove(path)
                    num_files_deleted += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")
        num_files_checked += 1
    print("Checked " + str(num_files_checked) + " files.")
    if delete_files:
        print("Deleted " + str(num_files_deleted) + " files.")

def main():
    # Use the function
    # silent_files = check_audio_files('path_to_your_audio_files')
    directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\dev\dev_audio"
    copy_silent_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\dev\silent"
    copy_short_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\dev\short"
    copy_long_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\dev\long"

    # directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\dev\dev_audio"
    
    # file_to_mute = os.path.join(directory, "dia0000_utt01.wav")
    # muted_file = os.path.join(directory, "dia0000_utt01_silent.wav")
    # create_silent_wav(file_to_mute, os.path.join(directory, muted_file))

    check_silent_audio_files(directory, copy_silent_directory, threshold=0.04, delete_files=True, copy_files=True)
    # check_short_audio_files(directory, copy_short_directory, cut_lower=0.60, cut_upper=0.70, lower_limit=False, copy_files=True, delete_files=True)
    # check_long_audio_files(directory, copy_long_directory, cut_lower=9, cut_upper=10, upper_limit=False, copy_files=True, delete_files=True)

if __name__ == "__main__":
    main()