import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForPreTraining, TrainingArguments, Trainer
import evaluate

# Assuming you have a DataFrame with columns "filename" and "emotion"
# data = pd.read_csv("C:/MyDocs/DTU/MSc/Thesis/Data/MELD/MELD_preprocess_test/pre_process_test.csv")
data = pd.read_csv("C:/Users/DANIEL/Desktop/thesis/low-resource-emotion-recognition/MELD_preprocess_test/pre_process_test.csv")

# directory = "C:/MyDocs/DTU/MSc/Thesis/Data/MELD/MELD_preprocess_test/MELD_preprocess_test_data"
directory = "C:/Users/DANIEL/Desktop/thesis/low-resource-emotion-recognition/MELD_preprocess_test/MELD_preprocess_test_data"

files = []

# Get a list of all files in the directory
for file in os.listdir(directory):
    if file.endswith('.wav'):
        files.append(file)

# Add filenames to a new column in the DataFrame
data['filename'] = files

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

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

# Show the label-encoding pairs:
print(label_encoder.classes_)
print("[0,         1,       2,       3,         4,         5]")

print(labels)

max_length = 16000 * 10  # 10 seconds

for index, row in data.iterrows():

    # Load audio file
    file_to_load = row['filename']
    file_to_load_path = os.path.join(directory, file_to_load)
    # print()
    # print(index)
    # print(file_to_load)
    # print()

    audio, sr = librosa.load(file_to_load_path, sr=16000)
    
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        padding = max_length - len(audio)
        offset = padding // 2
        audio = np.pad(audio, (offset, padding - offset), 'constant')
    
    # Extract features (e.g., MFCCs)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Mean across time
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    features.append(mfccs_processed)
    
    # Encode label
    # labels.append(label_encoder.transform([row['Emotion']]))

# Convert to arrays
features = np.array(features)
labels = np.array(labels).flatten()


# Now, `features` and `labels` can be used for training your model
# Optionally, save them to disk
# np.save('features.npy', features)
# np.save('labels.npy', labels)

print(features.shape)
print(labels.shape)

# Convert features and labels into PyTorch tensors
features_tensor = torch.tensor(features).float()
labels_tensor = torch.tensor(labels).long()  # Use .long() for integer labels, .float() for one-hot

# Choose train indices and validation indices
train_indices = np.random.choice(len(features), int(0.8 * len(features)), replace=False)
val_indices = np.array([i for i in range(len(features)) if i not in train_indices])


# Create dataset and dataloader for both training and validation sets
train_dataset = TensorDataset(features_tensor[train_indices], labels_tensor[train_indices])
val_dataset = TensorDataset(features_tensor[val_indices], labels_tensor[val_indices])

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Load a pre-trained model for pretrained
model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53", num_labels=5)

# Define training arguments
training_args = TrainingArguments(output_dir="test_trainer")

# Initialize the trainer
metric = evaluate.load("accuracy")

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# Prepare the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()



# Save the model
torch.save(model.state_dict(), 'emotion_recognition_model.pth')
