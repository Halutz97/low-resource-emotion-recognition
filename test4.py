import librosa
import wandb
import zipfile
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
import evaluate
from datasets import Dataset
from google.colab import drive, files
drive.mount('/content/drive')

# Set environment variables
os.environ["WANDB_PROJECT"] = "emotion-recognition-IEMOCAP"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Initialize wandb
wandb.init()

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

# !pip install accelerate -U

zip_path = '/content/drive/My Drive/Thesis_Data/IEMOCAP_runs/Run1/IEMOCAP_full_release.zip'
extract_to = '/content/extracted_data'


if os.path.exists(extract_to):
    if not os.listdir(extract_to):
        # If the directory is empty, extract the files
        # os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Files extracted successfully!")
    else:
        print("Directory is not empty. Extraction skipped to avoid overwriting.")
else:
    print("Directory does not exist. Creating...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Files extracted successfully!")




if not os.listdir(extract_to):
    # If the directory is empty, extract the files
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Files extracted successfully!")
else:
    print("Directory is not empty. Extraction skipped to avoid overwriting.")


data = pd.read_csv('/content/extracted_data/labels_corrected.csv')


files = []

directory = os.path.join(extract_to, "audio")

# Get a list of all files in the directory
for file in os.listdir(directory):
    if file.endswith('.wav'):
        files.append(file)

# Add filenames to a new column in the DataFrame
data['filename'] = files

features = []
labels = []

my_encoding_dict = {'ang': 0, 'dis': 1, 'fea': 2, 'hap': 3, 'neu': 4, 'sad': 5, 'sur': 6, 'fru': 7, 'exc': 8, 'oth': 9}

labels = data['Emotion'].map(my_encoding_dict).values

# Print the classes in the order they were encountered
print(my_encoding_dict)


max_length = 16000 * 9  # 9 seconds

# Load the processor
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

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
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53", num_labels=10)

# Log model parameters and gradients
wandb.watch(model)


# Initialize the trainer
metric = evaluate.load("accuracy")

# Prepare the trainer

training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=3,              # Number of training epochs
    per_device_eval_batch_size=8,    # Batch size for evaluation
    gradient_accumulation_steps=2,   # Number of updates steps to accumulate before performing a backward/update pass
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
    save_strategy='steps',               # Saving model checkpoint strategy
    save_steps=500,                      # Save checkpoint every 500 steps
    save_total_limit=3,
    fp16=True,                        # Enable mixed precision training
    report_to="wandb"                # Report the results to Weights & Biases
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

sweep_config = {
    'method': 'bayes',  # grid, random
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate': {
            'min': 1e-6,
            'max': 1e-3
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="my-first-sweep")

wandb.agent(sweep_id, function=trainer, count=10)

# Save the model
torch.save(model.state_dict(), 'emotion_recognition_model.pth')

save_path = '/content/drive/My Drive/Thesis_Data/IEMOCAP_runs/Run1/model/emotion_recognition_model.pth'
torch.save(model.state_dict(), save_path)