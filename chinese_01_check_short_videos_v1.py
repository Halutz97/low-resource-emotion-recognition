# Import necessary libraries
import os
import numpy as np
import pandas as pd
import moviepy.editor

# Define the source and destination directories
source_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\VideoMP4"

# check how many videos are under 1 second
num_short_videos = 0
num_videos_checked = 0
for file in os.listdir(source_directory):
    video = moviepy.editor.VideoFileClip(os.path.join(source_directory, file))
    video.close()
    duration = video.duration
    if duration < 1:
        num_short_videos += 1
        print(f"{file} == Duration: {duration}")
    num_videos_checked += 1
    if num_videos_checked % 100 == 0:
        print(f"Checked {num_videos_checked} videos.")

print(f"Number of short videos: {num_short_videos}")