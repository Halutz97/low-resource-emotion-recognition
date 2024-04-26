# Multitasking speech emotion recognition and speech recognition

import librosa
import zipfile
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
import evaluate
from datasets import Dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        class Sample:
            pass
        sample = Sample()
        sample.input_ids = self.features[idx]
        sample.labels = self.labels[idx]
        return sample


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Assuming you have a DataFrame with columns "filename" and "emotion"
# data = pd.read_csv("C:/MyDocs/DTU/MSc/Thesis/Data/MELD/MELD_preprocess_test/pre_process_test.csv")
# data = pd.read_csv("C:/Users/DANIEL/Desktop/thesis/low-resource-emotion-recognition/MELD_preprocess_test/pre_process_test.csv")
data = pd.read_csv('/train_labels_corrected.csv')

# directory = "C:/MyDocs/DTU/MSc/Thesis/Data/MELD/MELD_preprocess_test/MELD_preprocess_test_data"
zip_path = '/train_audio-002.zip'
extract_to = '/data'
# os.makedirs(extract_to, exist_ok=True)
# directory = '/content/drive/My Drive/Thesis_Data/MELD/Run3/data/train_audio.zip'

if not os.listdir(extract_to):
    # If the directory is empty, extract the files
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Files extracted successfully!")
else:
    print("Directory is not empty. Extraction skipped to avoid overwriting.")


files = []

directory = os.path.join(extract_to, "train_audio")

# Get a list of all files in the directory
for file in os.listdir(directory):
    if file.endswith('.wav'):
        files.append(file)

# Add filenames to a new column in the DataFrame
data['filename'] = files

features = []
labels = []

label_encoder = LabelEncoder()

raw_labels = data['Emotion'].values
labels = label_encoder.fit_transform(raw_labels)

# Show the label-encoding pairs:
print(label_encoder.classes_)
print("[0,         1,       2,       3,         4,         5,   6]")

print(labels)

max_length = 16000 * 9  # 9 seconds

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


    features.append(audio)

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

# Convert features to a float tensor and transpose the last two dimensions
features_tensor = torch.tensor(features).float()
labels_tensor = torch.tensor(labels).long()  # Use .long() for integer labels, .float() for one-hot

# Choose train indices and validation indices
train_indices = np.random.choice(len(features), int(0.8 * len(features)), replace=False)
val_indices = np.array([i for i in range(len(features)) if i not in train_indices])


# Convert the TensorDatasets to Datasets
train_dataset = Dataset.from_dict({
    'input_values': features_tensor[train_indices],
    'labels': labels_tensor[train_indices]
})
val_dataset = Dataset.from_dict({
    'input_values': features_tensor[val_indices],
    'labels': labels_tensor[val_indices]
})

# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load a pre-trained model for pretrained
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53", num_labels=7)

# Define training arguments
# training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# Initialize the trainer
metric = evaluate.load("accuracy")

# Prepare the trainer

training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=20,             # Number of training epochs
    per_device_train_batch_size=4,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
    save_strategy='steps',               # Saving model checkpoint strategy
    save_steps=500,                      # Save checkpoint every 500 steps
    save_total_limit=3
)

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

save_path = '/content/drive/My Drive/Thesis_Data/MELD/Run2/model/emotion_recognition_model.pth'
torch.save(model.state_dict(), save_path)