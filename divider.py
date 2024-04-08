import os
import shutil
import random

# Define the directory where your .wav files are
directory = 'CremaD'

# Define the directories for your train and test sets
train_dir = 'CremaD/train'
test_dir = 'CremaD/test'

# Create the train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get a list of all .wav files in the directory
wav_files = [file for file in os.listdir(directory) if file.endswith('.wav')]

# Shuffle the list to ensure randomness
random.shuffle(wav_files)

# Calculate the number of files to put in the train set (70% of total)
num_train = int(len(wav_files) * 0.7)

# Move the appropriate files to the train and test directories
for i, file in enumerate(wav_files):
    if i < num_train:
        shutil.move(os.path.join(directory, file), os.path.join(train_dir, file))
    else:
        shutil.move(os.path.join(directory, file), os.path.join(test_dir, file))