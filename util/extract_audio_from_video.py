from moviepy.editor import VideoFileClip
import os
import logging
import sys
from contextlib import contextmanager

# This script extracts audio from video files and saves it as WAV files.

# Set the logging level to CRITICAL to silence console output from MoviePy
# logging.getLogger('moviepy').setLevel(logging.CRITICAL)

# Replace 'your_video_file.mp4' with the path to your MP4 file

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

if __name__ == "__main__":
    # video_directory = "C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_preprocess_test\MELD_preprocess_test_data"
    video_directory = "C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\dev\dev_splits_complete"
    destination_directory = "C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\dev\dev_audio"

    files_processed = 0
    num_files_to_process = 200
    # Only process 100 files at a time.
    # Only files that have not already been processed will be processed, 
    # so it is fine to just run the script multiple times.

    for file in os.listdir(video_directory):
        if files_processed >= num_files_to_process:
            break
        if file.endswith('.mp4'):
            video_file_path = os.path.join(video_directory, file)
            audio_file_path = os.path.join(destination_directory, file.replace('.mp4', '.wav'))
            if os.path.isfile(audio_file_path):
                continue
            else:
                with suppress_stdout_stderr():
                    # Load the video file
                    video = VideoFileClip(video_file_path)
                    # Extract the audio
                    audio = video.audio
                    # Write the audio to a file (in WAV format)
                    audio.write_audiofile(audio_file_path, verbose=False)
                    audio.close()
                    video.close()
                files_processed += 1
                if (files_processed % 5 == 0):
                    print(str(files_processed) + "/" + str(num_files_to_process) + " files processed.") # Optional status update

    print("Processed " + str(files_processed) + " files.")