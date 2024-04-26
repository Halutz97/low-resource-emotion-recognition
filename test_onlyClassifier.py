import librosa
import zipfile
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
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

# Load the processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

for index, row in data.iterrows():

    # Load audio file
    file_to_load = row['filename']
    file_to_load_path = os.path.join(directory, file_to_load)

    # Load and preprocess the audio
    audio, sr = librosa.load(file_to_load_path, sr=16000)
    audio = librosa.util.normalize(audio)

    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        padding = max_length - len(audio)
        offset = padding // 2
        audio = np.pad(audio, (offset, padding - offset), 'constant')

    # Process the audio
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")


    features.append(inputs.input_values[0])

    # Encode label
    # labels.append(label_encoder.transform([row['Emotion']]))



# Convert labels to tensors
labels_tensor = torch.tensor(labels).long()  # Use .long() for integer labels, .float() for one-hot

# Print the dimensions of the labels tensor
print(f"Labels tensor dimensions: {labels_tensor.shape}")

# Choose train indices and validation indices
indices = torch.randperm(len(features))
train_indices = indices[:int(0.8 * len(features))]
val_indices = indices[int(0.8 * len(features)):]

# Print the number of training and validation samples
print(f"Number of training samples: {len(train_indices)}")
print(f"Number of validation samples: {len(val_indices)}")


# Convert the TensorDatasets to Datasets
train_dataset = Dataset.from_dict({
    'input_values': [features[i] for i in train_indices],
    'labels': labels_tensor[train_indices]
})
val_dataset = Dataset.from_dict({
    'input_values': [features[i] for i in val_indices],
    'labels': labels_tensor[val_indices]
})

# Print the dimensions of the first feature in the training and validation dataset
print(f"First training sample dimensions: {len(train_dataset['input_values'][0])}")
print(f"First validation sample dimensions: {len(val_dataset['input_values'][0])}")


# Load a pre-trained model for pretrained
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53", num_labels=7)

# Freeze all the parameters of the model
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the classifier parameters
for param in model.classifier.parameters():
    param.requires_grad = True


# Initialize the trainer
metric = evaluate.load("accuracy")

# Prepare the trainer

training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    learning_rate=1e-4,              # Learning rate
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=4,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    gradient_accumulation_steps=2,   # Number of updates steps to accumulate before performing a backward/update pass
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
    save_strategy='steps',               # Saving model checkpoint strategy
    save_steps=500,                      # Save checkpoint every 500 steps
    save_total_limit=3,
    fp16=True                        # Enable mixed precision training
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