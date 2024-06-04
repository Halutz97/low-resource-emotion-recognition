# Import necessary libraries
import os
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

def extract_files(video_files, video_directory, destination_directory, num_files_to_process):

    # Only files that have not already been processed will be processed, 
    # so it is fine to just run the script multiple times.
    
    files_processed = 0
    for file in video_files:
        if files_processed >= num_files_to_process:
            break
        video_file_path = os.path.join(video_directory, file + '.mp4')
        audio_file_path = os.path.join(destination_directory, file + '.wav')
        if os.path.isfile(audio_file_path):
            continue
        else:
            # Load the video file
            video = VideoFileClip(video_file_path)
            # Extract the audio
            audio = video.audio
            # Write the audio to a file (in WAV format)
            audio.write_audiofile(audio_file_path, logger=None, verbose=False)
            audio.close()
            video.close()
        files_processed += 1
        if (files_processed % 10 == 0):
            print(str(files_processed) + "/" + str(num_files_to_process) + " files processed.") # Optional status update
        
    print("Processed " + str(files_processed) + " files.")
          
if __name__ == "__main__":
    source_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\VideoMP4_testing"
    labels = pd.read_csv(r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\label_corrected_testing.csv")
    video_files = labels['filename'].tolist()
    # video_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\VideoMP4"
    destination_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\AudioWAV_testing"
    # Create the destination directory if it does not exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
        print(f"Created directory: {destination_directory}")
    num_files_to_process = 600
    extract_files(video_files, source_directory, destination_directory, num_files_to_process)


