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
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from google.colab import drive
drive.mount('/content/drive')

torch.cuda.empty_cache()

# Set environment variables
os.environ["WANDB_PROJECT"] = "emotion-recognition-IEMOCAP"
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
        emotion_label = self.annotations.iloc[index]['Label']

        audio_file_path = os.path.join(self.audio_dir, audio_filename+'.wav')
        audio, sr = torchaudio.load(audio_file_path)
        audio = audio.squeeze(0)
        if audio.shape[0] > self.max_length:
            audio = audio[:self.max_length]
        else:
            padding = self.max_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding), "constant", 0)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        label = torch.tensor(emotion_label, dtype=torch.long)
        return {'input_values': inputs.input_values.squeeze(0), 'labels': label}


class CustomWav2Vec2ForSequenceClassification(Wav2Vec2ForSequenceClassification):
    def __init__(self, config, num_linear_layers=1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.class_weights = weights_torch

        # Redefine the classifier to have multiple linear layers
        layers = [nn.Linear(config.hidden_size, config.hidden_size) for _ in range(num_linear_layers - 1)]
        layers += [nn.Linear(config.hidden_size, config.num_labels)]

        # Use a fixed dropout value if hidden_dropout_prob is not available
        dropout_rate = getattr(config, 'hidden_dropout_prob', 0.1)  # Default to 0.1 if not specified

        self.classifier = nn.Sequential(*layers,nn.Dropout(dropout_rate))

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

        # Average pooling across the sequence length (dim=1)
        logits = logits.mean(dim=1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights) if self.class_weights is not None else nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)  # No need to view since logits and labels should match
        

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

    def setup_optimizers(self):
        if self.optimizer is None:
          self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        # print(f"Logits shape: {outputs.logits.shape}, Labels shape: {labels.shape}")  # Debugging line
        # Assume class weights are handled within the model
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        # Override to directly return the DataLoader
        return self.train_dataset

    def get_eval_dataloader(self, eval_dataset=None):
        # Override to directly return the DataLoader for evaluation
        return self.eval_dataset if eval_dataset is None else eval_dataset

    def train(self, model_path=None, trial=None):
        """
        Main training entry point.
        """
        # Ensure optimizer is setup
        self.setup_optimizers()

        train_dataloader = self.get_train_dataloader()

        for epoch in range(int(self.args.num_train_epochs)):
            self.model.train()
            for step, inputs in enumerate(train_dataloader):
                # Ensure all inputs are on the same device.
                inputs = self._prepare_inputs(inputs)

                loss, outputs = self.compute_loss(self.model, inputs, return_outputs=True)
                self.model.zero_grad()
                loss.backward()
                if self.optimizer is None:
                    raise ValueError("Optimizer not initialized")
                self.optimizer.step()
                if step % self.args.logging_steps == 0:
                    print(f"Step {step}: Loss {loss.item()}")

        torch.cuda.empty_cache()
        print("Cleared GPU cache after all training epochs.")


def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    labels = [item['labels'] for item in batch]
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
    return metric.compute(predictions=predictions, references=labels)


def wait_for_file(filename, timeout=60):
    """Wait for a file to exist."""
    start_time = time.time()
    while not os.path.exists(filename) or not os.path.getsize(filename) > 0:
        time.sleep(1)
        if time.time() - start_time > timeout:
            raise FileNotFoundError(f"File {filename} not found after {timeout} seconds")

# Define the paths to use
zip_path = '/content/drive/My Drive/Thesis_Data/IEMOCAP_runs/Run5/IEMOCAP_full_release3.zip'
extract_to = '/content/extracted_data'
directory = os.path.join(extract_to, "audio_training")

# Define basic parameters of the model and dataset
my_encoding_dict = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3, 'sur': 4, 'fru': 5, 'exc': 6}
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

# Obtain the wights for the classes
labels_np = np.array(labels)
class_counts = np.bincount(labels)
class_weights = 1. / class_counts  # Inverse of class counts
class_weights = class_weights / class_weights.sum() * len(np.unique(labels))  # Normalize to keep the same scale
weights_torch = torch.tensor(class_weights, dtype=torch.float).to(device)

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

# Initialize the trainer
metric = evaluate.load("accuracy")

# Prepare the trainer
def train_model():
    run = wandb.init(project="emotion-recognition-IEMOCAP", reinit=True, config={
        "learning_rate": 1e-4,  # Default learning rate
        "batch_size": 4,        # Adjusted to match DataLoader
        "num_linear_layers": 1  # Default number of linear layers
    })

    # Load a pre-trained model for pretrained
    config = AutoConfig.from_pretrained("facebook/wav2vec2-large-xlsr-53", num_labels=7)
    model = CustomWav2Vec2ForSequenceClassification(config, num_linear_layers=run.config.num_linear_layers)
    model.to(device)

    # Freeze all parameters in the base wav2vec2 model
    for param in model.wav2vec2.parameters():
      param.requires_grad = False

    # Unfreeze the classifier parameters in your custom classifier
    for param in model.classifier.parameters():
      param.requires_grad = True

    # Check which parameters are frozen and which are not
    for name, param in model.named_parameters():
      print(f"{name} is {'trainable' if param.requires_grad else 'frozen'}")

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
        metric_for_best_model="accuracy",      # Use accuracy to evaluate the best model
        greater_is_better=True,                # Higher accuracy is better
        fp16=True,                        # Enable mixed precision training
        report_to="wandb"                # Report the results to Weights & Biases
    )


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        compute_metrics=compute_metrics,
        class_weights=weights_torch  # Pass your computed class weights here
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
      'name': 'accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 1e-4},
        'num_linear_layers': {'values': [1, 2, 3]}  # Sweep over 1, 2, or 3 linear layers
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="emotion-recognition-IEMOCAP")

wandb.agent(sweep_id, function=train_model, count=6)