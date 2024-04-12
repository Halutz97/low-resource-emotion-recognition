import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Assuming you have a DataFrame with columns "filename" and "emotion"
# data = pd.read_csv("C:/MyDocs/DTU/MSc/Thesis/Data/MELD/MELD_preprocess_test/pre_process_test.csv")
data = pd.read_csv(r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_fine_tune_v1_train_data\train_labels.csv")
# Audio files directory
directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_fine_tune_v1_train_data"

# iterate through dataframe:
for index, row in data.iterrows():
    dialogue_id = str(row['Dialogue_ID'])
    utterance_id = str(row['Utterance_ID'])
    if int(dialogue_id) >= 1000:
            dialogue_id = dialogue_id
    elif int(dialogue_id) >= 100:
        dialogue_id = "0" + dialogue_id
    elif int(dialogue_id) >= 10:
        dialogue_id = "00" + dialogue_id
    else:
        dialogue_id = "000" + dialogue_id

    if int(utterance_id) >= 10:
        utterance_id = utterance_id
    else:
        utterance_id = "0" + utterance_id

    row['Expected filename'] = "dia" + dialogue_id + "_utt" + utterance_id + ".wav"
    print(row['Expected filename'])
    data.at[index, 'Expected filename'] = row['Expected filename']


print(data['Expected filename'])

files = []

# Get a list of all files in the directory
for file in os.listdir(directory):
    if file.endswith('.wav'):
        files.append(file)

# sort 'files' alphabetically
files.sort()

# remove dataframe rows if file is not in 'files'
data = data[data['Expected filename'].isin(files)]

# Add filenames to a new column in the DataFrame
data['filename'] = files

# Check if the filenames match the expected filenames and store in new column 'Match'
data['Match'] = np.where(data['filename'] == data['Expected filename'], True, False)

# Iterate through dataframe and check if any filenames do not match the expected filenames
num_missmatches = 0
for index, row in data.iterrows():
    if row['Match'] == False:
        print(row['filename'])
        print(row['Expected filename'])
        print()
        num_missmatches += 1

print("Number of missmatches: " + str(num_missmatches))

# Export dataframe to csv
# data.to_csv(r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_fine_tune_v1_train_data\export_csv.csv", index=False)

# Number of entries in dataframe:
# print(len(data))

# print(data.head())

# print(data.Emotion)

# Placeholder for features and labels
features = []
labels = []

label_encoder = LabelEncoder()

raw_labels = data['Emotion'].values
labels = label_encoder.fit_transform(raw_labels)

# # Show the label-encoding pairs:
print(label_encoder.classes_)
print("[0,         1,       2,       3,         4,         5]")

print(labels)

# Test manually if encoding is correct
my_encoding_dict = {'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 'disgust': 4, 'fear': 5}

# max_length = 16000 * 10  # 10 seconds

# for index, row in data.iterrows():

#     # Load audio file
#     file_to_load = row['filename']
#     file_to_load_path = os.path.join(directory, file_to_load)
#     # print()
#     # print(index)
#     # print(file_to_load)
#     # print()

#     audio, sr = librosa.load(file_to_load_path, sr=16000)
    
#     if len(audio) > max_length:
#         audio = audio[:max_length]
#     else:
#         padding = max_length - len(audio)
#         offset = padding // 2
#         audio = np.pad(audio, (offset, padding - offset), 'constant')
    
#     # Extract features (e.g., MFCCs)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
#     # Mean across time
#     mfccs_processed = np.mean(mfccs.T, axis=0)
    
#     features.append(mfccs_processed)
    
#     # Encode label
#     # labels.append(label_encoder.transform([row['Emotion']]))

# # Convert to arrays
# features = np.array(features)
# labels = np.array(labels).flatten()

# # Now, `features` and `labels` can be used for training your model
# # Optionally, save them to disk
# # np.save('features.npy', features)
# # np.save('labels.npy', labels)

# print(features.shape)
# print(labels.shape)

# # Convert features and labels into PyTorch tensors
# features_tensor = torch.tensor(features).float()
# labels_tensor = torch.tensor(labels).long()  # Use .long() for integer labels, .float() for one-hot

# # Choose train indices and validation indices
# train_indices = np.random.choice(len(features), int(0.8 * len(features)), replace=False)
# val_indices = np.array([i for i in range(len(features)) if i not in train_indices])


# # Create dataset and dataloader for both training and validation sets
# train_dataset = TensorDataset(features_tensor[train_indices], labels_tensor[train_indices])
# val_dataset = TensorDataset(features_tensor[val_indices], labels_tensor[val_indices])

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# # Define a simple neural network for classification
# class EmotionClassifier(nn.Module):
#     def __init__(self, num_features, num_classes):
#         super(EmotionClassifier, self).__init__()
#         self.layer1 = nn.Linear(num_features, 512)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self.layer2 = nn.Linear(512, num_classes)
        
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.layer2(x)
#         return x
    

# # Initialize the model, loss function, and optimizer
# model = EmotionClassifier(num_features=features.shape[1], num_classes=len(np.unique(labels)))
# criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # # Freeze early layers, fine-tune the rest
# # for name, param in model.named_parameters():
# #     if name in ['layer2.weight', 'layer2.bias']:
# #         param.requires_grad = True
# #     else:
# #         param.requires_grad = False

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     for inputs, targets in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
    
#     # Validation loop
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             val_loss += loss.item()
    
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss / len(val_loader)}")

# # Save the model
# torch.save(model.state_dict(), 'emotion_recognition_model.pth')
