import numpy as np
import pandas as pd
import os

# Read csv file with list of all files
sentence_filenames = pd.read_csv(r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\SentenceFilenames.csv")

# Check the column names
print(sentence_filenames.columns)

keys = []
IDs = []
emotions = []
activations = []

# Iterate over the rows in the csv file
for index, row in sentence_filenames.iterrows():
    # For each filename, take a substring
    ID = row['Filename'][0:4]
    key = row['Filename'][5:8]
    emotion = row['Filename'][9:12]
    activation = row['Filename'][13:15]
    # Add to keys
    keys.append(key)
    IDs.append(ID)
    emotions.append(emotion)
    activations.append(activation)

# Get frequency of each key
keys_freq = np.unique(keys, return_counts=True)
print(keys_freq)

keys_unique = np.unique(keys)

print(keys_unique)

# Get frequency of each ID
IDs_freq = np.unique(IDs, return_counts=True)
print(IDs_freq)

# Get unique IDs
IDs_unique = np.unique(IDs)
print(IDs_unique)

# Get frequency of each emotion
emotions_freq = np.unique(emotions, return_counts=True)
print(emotions_freq)

# Get unique emotions
emotions_unique = np.unique(emotions)
print(emotions_unique)

# Get frequency of each activation
activations_freq = np.unique(activations, return_counts=True)
print(activations_freq)

# Get unique activations
activations_unique = np.unique(activations)
print(activations_unique)

