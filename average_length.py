import json
import numpy as np

directory = r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\train.json'

# Load train.json
with open(directory) as f:
    data = json.load(f)

lengths = []

# Get length of each sequence
for utt_id, utt_data in data.items():
    length = float(utt_data['length'])
    lengths.append(length)

print("Max length: ", max(lengths))
print("Min length: ", min(lengths))

average = np.mean(lengths)
print("Average length: ", average)

