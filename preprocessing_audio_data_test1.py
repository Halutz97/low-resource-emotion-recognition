import librosa
import numpy as np
import pandas as pd
import os

# Assuming you have a DataFrame with columns "filename" and "emotion"
data = pd.read_csv("C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_preprocess_test\pre_process_test.csv")

directory = "C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_preprocess_test\MELD_preprocess_test_data"

files = []

# Get a list of all files in the directory
for file in os.listdir(directory):
    if file.endswith('.wav'):
        files.append(file)

# Add filenames to a new column in the DataFrame
data['filename'] = files

# Number of entries in dataframe:
# print(len(data))

print(data.head())

# print(data.Emotion)

# Placeholder for features and labels
features = []
labels = []

for index, row in data.iterrows():
    print(str(row))
    print()
    print(row['Emotion'])

    # Load audio file
#     file_to_load = data.loc[index, 'filename']
#     audio, sr = librosa.load(row['filename'], sr=16000)
    
#     # Extract features (e.g., MFCCs)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
#     # Mean across time
#     mfccs_processed = np.mean(mfccs.T, axis=0)
    
#     features.append(mfccs_processed)
    
#     # Encode label
#     labels.append(label_encoder.transform([row['emotion']]))

# # Convert to arrays
# features = np.array(features)
# labels = np.array(labels).flatten()

# # Now, `features` and `labels` can be used for training your model
# # Optionally, save them to disk
# np.save('features.npy', features)
# np.save('labels.npy', labels)
