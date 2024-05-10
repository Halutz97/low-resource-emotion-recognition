import librosa
#!pip install wandb
#!pip install evaluate
import wandb
import zipfile
import numpy as np
import pandas as pd
import os
import time
import torch
#!pip install accelerate -U
#!pip install transformers[torch] -U
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import evaluate
import torch
import torch.nn as nn
from datasets import Dataset
from google.colab import drive, files
drive.mount('/content/drive')

# Set environment variables
os.environ["WANDB_PROJECT"] = "emotion-recognition-IEMOCAP"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ['WANDB_API_KEY'] = '00e7b3e4cf2d54995e43ef45df8a0ec3767a3e91'


class CustomWav2Vec2ForSequenceClassification(Wav2Vec2ForSequenceClassification):
    def __init__(self, config, num_linear_layers=1):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # Redefine the classifier to have multiple linear layers
        layers = [nn.Linear(config.hidden_size, config.hidden_size) for _ in range(num_linear_layers - 1)]
        layers += [nn.Linear(config.hidden_size, config.num_labels)]
        self.classifier = nn.Sequential(
            *layers,
            nn.Dropout(config.hidden_dropout_prob)
        )
        
    def forward(self, input_values, attention_mask=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure class_weights are on the right device:
        self.class_weights = class_weights.to(self.model.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Check if class_weights were provided and use them
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    

def custom_compute_loss(model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = nn.CrossEntropyLoss(weight=weights_tensor)
    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def wait_for_file(filename, timeout=60):
    """Wait for a file to exist."""
    start_time = time.time()
    while not os.path.exists(filename) or not os.path.getsize(filename) > 0:
        time.sleep(1)
        if time.time() - start_time > timeout:
            raise FileNotFoundError(f"File {filename} not found after {timeout} seconds")


zip_path = '/content/drive/My Drive/Thesis_Data/IEMOCAP_runs/Run1/IEMOCAP_full_release.zip'
extract_to = '/content/extracted_data'
directory = os.path.join(extract_to, "audio")


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
csv_file = '/content/extracted_data/labels_corrected.csv'
wait_for_file(csv_file)


data = pd.read_csv(csv_file)


files = []


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

labels_np = np.array(labels)
class_counts = np.bincount(labels)
class_weights = 1. / class_counts  # Inverse of class counts
class_weights = class_weights / class_weights.sum() * len(np.unique(labels))  # Normalize to keep the same scale
weights_tensor = torch.tensor(class_weights, dtype=torch.float32) # This works on CPU, add .to(device) if using GPU



max_length = 16000 * 10  # 9 seconds

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

# Initialize the trainer
metric = evaluate.load("accuracy")

# Prepare the trainer
def train_model():
    run = wandb.init(project="emotion-recognition-IEMOCAP", reinit=True, config={
        "learning_rate": 1e-4,  # Default learning rate
        "batch_size": 4,        # Default batch size
        "num_linear_layers": 1  # Default number of linear layers
    })

    # Load a pre-trained model for pretrained
    config = AutoConfig.from_pretrained("facebook/wav2vec2-large-xlsr-53", num_labels=10)
    model = CustomWav2Vec2ForSequenceClassification(config, num_linear_layers=run.config.num_linear_layers)

    # Log model parameters and gradients
    wandb.watch(model, log='all', log_freq=100)  # Configure as needed

    learning_rate=wandb.config.learning_rate
    batch_size=wandb.config.batch_size

    training_args = TrainingArguments(
        output_dir='./results',          # Output directory
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=3,              # Number of training epochs
        per_device_eval_batch_size=4,    # Batch size for evaluation
        gradient_accumulation_steps=4,   # Number of updates steps to accumulate before performing a backward/update pass
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
        metric_for_best_model="acurracy",      # Use accuracy to evaluate the best model
        greater_is_better=True,                # Higher accuracy is better
        fp16=True,                        # Enable mixed precision training
        report_to="wandb"                # Report the results to Weights & Biases
    )


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=weights_tensor  # Pass your computed class weights here
    )

    trainer.train()

    model_path = f'/content/drive/My Drive/Thesis_Data/IEMOCAP_runs/Run2/model/emotion_recognition_model_{run.id}.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    run.finish()

sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 1e-4},
        'batch_size': {'values': [4, 6]},
        'num_linear_layers': {'values': [1, 2, 3]}  # Sweep over 1, 2, or 3 linear layers
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="emotion-recognition-IEMOCAP")

wandb.agent(sweep_id, function=train_model, count=6)