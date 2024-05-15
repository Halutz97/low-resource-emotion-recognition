# !pip install wandb
# !pip install accelerate -U
# !pip install transformers[torch] -U
import wandb
import zipfile
import numpy as np
import pandas as pd
import os
import time
import ast
from transformers import Wav2Vec2Processor, Wav2Vec2Model, TrainingArguments, Trainer, AutoConfig
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from google.colab import drive
drive.mount('/content/drive')

torch.cuda.empty_cache()

# Set environment variables
os.environ["WANDB_PROJECT"] = "emotion-recognition-att-IEMOCAP"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ['WANDB_API_KEY'] = '00e7b3e4cf2d54995e43ef45df8a0ec3767a3e91'


class AudioDataset(Dataset):
    def __init__(self, annotations, audio_dir, processor, max_length=160000):
        self.annotations = annotations
        self.audio_dir = audio_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        audio_filename = self.annotations.iloc[index]['filename']
        attributes = self.annotations.iloc[index]['Attributes']
        valence, activation, _ = ast.literal_eval(attributes)

        audio_file_path = os.path.join(self.audio_dir, audio_filename+'.wav')
        audio, sr = torchaudio.load(audio_file_path)
        audio = audio.squeeze(0)
        if audio.shape[0] > self.max_length:
            audio = audio[:self.max_length]
        else:
            padding = self.max_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding), "constant", 0)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")

        return {'input_values': inputs.input_values.squeeze(0), 
                'valence': torch.tensor(valence, dtype=torch.float),
                'activation': torch.tensor(activation, dtype=torch.float)}
    
class CustomWav2Vec2ForRegression(nn.Module):
    def __init__(self, config):
        super(CustomWav2Vec2ForRegression, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53", config=config)
        
        # This assumes config.hidden_size is available and set appropriately.
        self.regression_head = nn.Linear(config.hidden_size, 2)  # Outputting two continuous values


    def forward(self, input_values, attention_mask=None):

        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        predictions = self.regression_head(pooled_output)

        return predictions
    
def wait_for_file(filename, timeout=60):
    """Wait for a file to exist."""
    start_time = time.time()
    while not os.path.exists(filename) or not os.path.getsize(filename) > 0:
        time.sleep(1)
        if time.time() - start_time > timeout:
            raise FileNotFoundError(f"File {filename} not found after {timeout} seconds")
        
def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    valence = [item['valence'] for item in batch]
    activation = [item['activation'] for item in batch]
    input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=0)
    valence = torch.stack(valence).to(device)
    activation = torch.stack(activation).to(device)
    

    return {
        'input_values': input_values_padded,
        'labels': torch.stack([valence, activation], dim=1)
    }

# Pearson Correlation Function
def pearson_correlation(y_true, y_pred):
    y_true = y_true - torch.mean(y_true)
    y_pred = y_pred - torch.mean(y_pred)
    corr = torch.sum(y_true * y_pred) / (torch.sqrt(torch.sum(y_true ** 2)) * torch.sqrt(torch.sum(y_pred ** 2)))
    if torch.isnan(corr) or torch.isinf(corr):
        return torch.tensor(0.0)
    return corr

# Define the MAE metric function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    mae = nn.L1Loss()(preds, labels.float())
    return {"mae": mae.item()}

# Define the paths to use
zip_path = '/content/drive/My Drive/Thesis_Data/IEMOCAP_runs/Run5/IEMOCAP_full_release3.zip'
extract_to = '/content/extracted_data'
directory = os.path.join(extract_to, "audio_training")

# Define basic parameters of the model and dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 16000 * 10  # 9 seconds


# Extract the files from the drive directory
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

print(f"There are {len(os.listdir(directory))} elements in the folder.")


# Wait until it has confirmed that the csv is there
csv_file = '/content/extracted_data/labels_training.csv'
wait_for_file(csv_file)

# Obtain the data and its labels
data = pd.read_csv(csv_file)


# Load the processor and dataset
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
dataset = AudioDataset(data, directory, processor, max_length=max_length)

# Split the dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Prepare DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)


# Prepare the trainer
def train_model():
    run = wandb.init(project="emotion-recognition-IEMOCAP", reinit=True, config={
        "learning_rate": 1e-4,  # Default learning rate
        "batch_size": 4,        # Adjusted to match DataLoader
    })

    # Load a pre-trained model for pretrained
    config = AutoConfig.from_pretrained("facebook/wav2vec2-large-xlsr-53", num_labels=2)
    model = CustomWav2Vec2ForRegression(config)
    model.to(device)


    # Log model parameters and gradients
    wandb.watch(model, log='all', log_freq=100)  # Configure as needed

    learning_rate=wandb.config.learning_rate
    batch_size=wandb.config.batch_size

    training_args = TrainingArguments(
        output_dir='./results',          # Output directory
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=3,              # Number of training epochs
        per_device_eval_batch_size=batch_size,    # Batch size for evaluation
        gradient_accumulation_steps=8,   # Number of updates steps to accumulate before performing a backward/update pass
        warmup_steps=500,                # Number of warmup steps for learning rate scheduler
        save_total_limit=1,                    # Only keep the best model
        weight_decay=0.01,               # Strength of weight decay
        logging_dir='./logs',            # Directory for storing logs
        logging_steps=10,
        evaluation_strategy="steps",            # Evaluate every `eval_steps`
        eval_steps=500,                         # Evaluation and save interval
        save_strategy='steps',               # Saving model checkpoint strategy
        save_steps=500,                      # Save checkpoint every 500 steps
        load_best_model_at_end=True,           # Load the best model at the end of training
        metric_for_best_model="mae",      # Use MAE to evaluate the best model
        greater_is_better=False,                # Lower error is better
        fp16=True,                        # Enable mixed precision training
        report_to="wandb"                # Report the results to Weights & Biases
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        compute_metrics=compute_metrics,
    )

    try:
      trainer.train()
    except Exception as e:
      print(f"An error occurred: {e}")
      # Optionally, re-raise the exception if you want to stop the process
      raise e

    model_path = f'/content/drive/My Drive/Thesis_Data/IEMOCAP_runs/Run5/model/emotion_recognition_model_{run.id}.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    run.finish()

sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'MAE',
      'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 1e-4},
        'batch_size': {'values': [4 , 8]},
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="emotion-recognition-att-IEMOCAP")

wandb.agent(sweep_id, function=train_model, count=8)