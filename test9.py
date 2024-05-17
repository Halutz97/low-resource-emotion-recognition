# !pip install wandb
# !pip install evaluate
# !pip install accelerate -U
# !pip install transformers[torch] -U
import wandb
import zipfile
import numpy as np
import pandas as pd
import os
import time
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer, AutoConfig, AdamW
from transformers.modeling_outputs import SequenceClassifierOutput
import evaluate
from sklearn. metrics import f1_score
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from google.colab import drive
drive.mount('/content/drive')

torch.cuda.empty_cache()

# Set environment variables
os.environ["WANDB_PROJECT"] = "emotion-recognition-IEMOCAP"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ['WANDB_API_KEY'] = '00e7b3e4cf2d54995e43ef45df8a0ec3767a3e91'

# Define your augmentation function
def augment_audio(audio, sr):
    # Ensure the audio tensor is in the correct shape (1, num_samples)
    audio = audio.unsqueeze(0) if len(audio.shape) == 1 else audio

    # Apply FrequencyMasking and TimeMasking directly to the audio waveform
    freq_masking = T.FrequencyMasking(freq_mask_param=30)
    time_masking = T.TimeMasking(time_mask_param=50)
    audio = freq_masking(audio)
    audio = time_masking(audio)

    return audio

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

class CombinedAudioDataset(Dataset):
    def __init__(self, annotations_list, audio_dir, max_length=160000, augment=False):
        self.annotations_list = annotations_list
        self.audio_dir = audio_dir
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return sum([len(annotations) for annotations in self.annotations_list])

    def __getitem__(self, index):
        for annotations in self.annotations_list:
            if index < len(annotations):
                sample = annotations.iloc[index]
                break
            index -= len(annotations)
        
        audio_filename = sample['filename']
        emotion_label = sample['Label']
        audio_file_path = os.path.join(self.audio_dir, audio_filename + '.wav')
        audio, sr = torchaudio.load(audio_file_path)

        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            audio = resampler(audio)

        # Ensure audio is a 1D tensor (sequence_length)
        audio = audio.squeeze(0)

        if audio.shape[0] > self.max_length:
            audio = audio[:self.max_length]
        else:
            padding = self.max_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding), "constant", 0)

        if self.augment:
            audio = augment_audio(audio, sr)
        
        label = torch.tensor(emotion_label, dtype=torch.long)
        return {'input_values': audio, 'labels': label}



def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Ensure input_values is a 2D tensor [batch_size, sequence_length]
    input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)

    # Move data to the appropriate device
    input_values_padded = input_values_padded.to(device)
    labels = labels.to(device)

    return {
        'input_values': input_values_padded,
        'labels': labels
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='weighted')
    return metric.compute(predictions=predictions, references=labels)
    # return {"f1":f1}


def wait_for_file(filename, timeout=60):
    """Wait for a file to exist."""
    start_time = time.time()
    while not os.path.exists(filename) or not os.path.getsize(filename) > 0:
        time.sleep(1)
        if time.time() - start_time > timeout:
            raise FileNotFoundError(f"File {filename} not found after {timeout} seconds")

def augment_class(df_class, num_samples, audio_dir, class_label, max_length=160000, chunk_size=10):
    augmented_samples = []
    num_chunks = (num_samples // chunk_size) + 1
    chunk_count = 0
    
    for _ in range(num_chunks):
        samples_to_augment = df_class.sample(n=min(chunk_size, num_samples - len(augmented_samples)), replace=True)
        
        for idx, sample in samples_to_augment.iterrows():
            audio_filename = sample['filename']
            audio_file_path = os.path.join(audio_dir, audio_filename + '.wav')
            
            # Load the audio file
            audio, sr = torchaudio.load(audio_file_path)
            
            if sr != 16000:
                resampler = T.Resample(sr, 16000)
                audio = resampler(audio)
            
            # Ensure audio is a 1D tensor (sequence_length)
            audio = audio.squeeze(0)
            
            if audio.shape[0] > max_length:
                audio = audio[:max_length]
            else:
                padding = max_length - audio.shape[0]
                audio = torch.nn.functional.pad(audio, (0, padding), "constant", 0)
            
            audio = augment_audio(audio, sr)
            
            # Convert back to dataframe format
            augmented_sample = sample.copy()
            augmented_sample['audio'] = audio.numpy()  # Save the augmented audio as numpy array
            augmented_samples.append(augmented_sample)
        
        # Save intermediate results to disk
        if len(augmented_samples) >= chunk_size:
            augmented_samples_df = pd.DataFrame(augmented_samples)
            augmented_samples_df.to_csv(f'/content/extracted_data/augmented_{class_label}_chunk_{chunk_count}.csv', index=False)
            chunk_count += 1
            augmented_samples = []  # Clear memory
    
    # Save any remaining samples
    if augmented_samples:
        augmented_samples_df = pd.DataFrame(augmented_samples)
        augmented_samples_df.to_csv(f'/content/extracted_data/augmented_{class_label}_chunk_{chunk_count}.csv', index=False)

    print(f"Augmentation for class {class_label} completed.")



# Define the paths to use
zip_path = '/content/drive/My Drive/Thesis_Data/IEMOCAP_runs/Run6/IEMOCAP_full_release_E3.zip'
extract_to = '/content/extracted_data'
directory = os.path.join(extract_to, "audio_training")

# Define basic parameters of the model and dataset
# my_encoding_dict = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3, 'sur': 4, 'fru': 5, 'exc': 6}
my_encoding_dict = {'ang': 0, 'hap': 1, 'neu': 2}
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
labels = data['Emotion'].map(my_encoding_dict).values

class_counts = data['Label'].value_counts()
max_samples = class_counts.max()

chunk_size = 10  # Adjust the chunk size based on your memory limits

# Split data by class
classes = data['Label'].unique()

for cls in classes:
    df_class = data[data.Label == cls]
    class_count = df_class.shape[0]
    if class_count < max_samples:
        augment_class(df_class, max_samples - class_count, directory, cls, max_length=max_length, chunk_size=chunk_size)
    else:
        # Save original data in chunks as well to manage memory
        for i in range(0, class_count, chunk_size):
            df_class.iloc[i:i + chunk_size].to_csv(f'/content/extracted_data/original_{cls}_chunk_{i//chunk_size}.csv', index=False)
        print(f"Original data for class {cls} saved in chunks.")

# Combine saved files
combined_annotations_list = []

for cls in classes:
    augmented_files = [f for f in os.listdir('/content/extracted_data') if f.startswith(f'augmented_{cls}_chunk')]
    original_files = [f for f in os.listdir('/content/extracted_data') if f.startswith(f'original_{cls}_chunk')]
    
    for file in augmented_files + original_files:
        combined_annotations_list.append(pd.read_csv(f'/content/extracted_data/{file}'))

combined_annotations = pd.concat(combined_annotations_list, ignore_index=True)

print("Dataset balanced:")
print(combined_annotations['Label'].value_counts())


# Initialize the trainer
metric = evaluate.load("accuracy")

import wandb
import time

def train_model():
    for _ in range(3):  # Retry mechanism
        try:
            run = wandb.init(project="emotion-recognition-IEMOCAP", reinit=True, config={
                "learning_rate": 1e-5,  # Default learning rate
                "batch_size": 4,        # Adjusted to match DataLoader
                "num_linear_layers": 1  # Default number of linear layers
            })

            batch_size = wandb.config.batch_size  # Ensure this is used consistently
            learning_rate = wandb.config.learning_rate

            # Load your dataset here
            annotations_files = [f for f in os.listdir('/content/extracted_data') if f.startswith('augmented') or f.startswith('original')]
            annotations_list = [pd.read_csv(f'/content/extracted_data/{file}') for file in annotations_files]
            dataset = CombinedAudioDataset(annotations_list, directory, max_length=max_length)

            # Split the dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # Create DataLoader with num_workers=0 to avoid BrokenPipeError
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

            # Load a pre-trained model
            model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53", num_labels=3)
            model.to(device)

            # Log model parameters and gradients
            wandb.watch(model, log='all', log_freq=100)  # Configure as needed

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
                metric_for_best_model="f1",      # Use F1 score to evaluate the best model
                greater_is_better=True,           # Higher F1 score is better
                max_grad_norm=1.0,                # Gradient clipping value
                fp16=True,                        # Enable mixed precision training
                report_to="wandb"                # Report the results to Weights & Biases
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_loader.dataset,  # Pass the dataset directly
                eval_dataset=val_loader.dataset,     # Pass the dataset directly
                compute_metrics=compute_metrics
            )

            trainer.train()

            model_path = f'/content/drive/My Drive/Thesis_Data/IEMOCAP_runs/Run6/model/emotion_recognition_model_{run.id}.pth'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

            run.finish()
            break  # Exit the retry loop if successful

        except Exception as e:
            print(f"An error occurred: {e}")
            wandb.finish(exit_code=1)
            time.sleep(5)  # Wait before retrying

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'f1',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {'values': [1e-6]},
        'batch_size': {'values': [4, 8]},
        # 'num_linear_layers': {'values': [1]}  # Uncomment if you want to sweep over linear layers
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="emotion-recognition-IEMOCAP")

wandb.agent(sweep_id, function=train_model, count=1)

