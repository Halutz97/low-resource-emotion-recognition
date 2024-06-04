# Import necessary libraries
import os
import numpy as np
import pandas as pd
import moviepy.editor

# Define the source and destination directories
source_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\VideoMP4"

# Load the CSV file
labels = pd.read_csv(r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\label.csv", dtype={'file': str})

# Print the column names
print(labels.columns)

# add 'folder' column and 'file' column to create a new column 'filename'

folders_list = labels['folder'].tolist()
files_list = labels['file'].tolist()

# Add two lists of strings together element-wise
filenames = [f"{folder}_{file}" for folder, file in zip(folders_list, files_list)]
# for name in filenames:
    # print(name)

# Add the filename_list to the dataframe
labels['filename'] = filenames

# get the first value in the 'file' column
print(type(labels['file'][0]))

# Drop the 'folder' and 'file' columns
labels = labels.drop(columns=['folder', 'file', 'chinese_text'])

# Print the first 5 rows of the dataframe
print(labels.head())

# Drop columns 'val1' to 'val4'
labels = labels.drop(columns=['val1', 'val2', 'val3', 'val4'])

# Use binary indexing to keep only rows wher 'data_split' is 'train'
labels = labels[labels['data_split'] == 'train']

# Drop the 'data_split' column
labels = labels.drop(columns=['data_split'])
# Reorder the columns to have 'filename' as the first column
labels = labels[['filename', 'Emotion']]

# Print the first 5 rows of the dataframe
print(labels.head())
# Print shape of the dataframe
print(labels.shape)

# Get emotion distribution
emotion_distribution = labels['Emotion'].value_counts()
print(emotion_distribution)
