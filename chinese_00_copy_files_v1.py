# Preprocessing of the CH-SIMS dataset
# Import the necessary libraries
import os
import numpy as np
import pandas as pd
import shutil

source_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\Raw"
dest_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\VideoMP4"
# create dest_directory if it doesn't exist
if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
    print(f"Directory {dest_directory} created.")

folders = os.listdir(source_directory)
print(folders)
# Use list comprehension to get a full list of filenames in each folder
# If they don't start with "._"
all_files = []
files_copied = 0
for folder in folders:
    files = [file for file in os.listdir(os.path.join(source_directory, folder)) if not file.startswith("._")]
    for file in files:
        if file.endswith(".mp4"):
            shutil.copy2(os.path.join(source_directory, folder, file), os.path.join(dest_directory, f"{folder}_{file}"))
            files_copied += 1

print(f"Files copied: {files_copied}")
# dataframe = pd.DataFrame(all_files, columns=["filename"])
# print(dataframe.head(50))    