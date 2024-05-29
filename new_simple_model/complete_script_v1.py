# !pip install transformers[torch] -U
import os
import zipfile
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AdamW
from tqdm import tqdm
from google.colab import drive
drive.mount('/content/drive')

torch.cuda.empty_cache()

class AudioDataset(Dataset):
    def __init__(self, directory, processor):
        """
        Args:
            directory (string): Directory with all the .wav files.
        """
        self.directory = directory
        self.file_names = [file for file in os.listdir(directory) if file.endswith('.wav')]
        self.processor = processor

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.file_names[idx])
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample the waveform if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Process the waveform through the wav2vec2 processor
        inputs = self.processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values

        # Optionally, you could add other transformations here (e.g., feature extraction)
        # return waveform, 16000  # Return the resampled waveform and the new sample rate
        return inputs
        

# # Usage
# audio_dataset = AudioDataset('/path/to/your/wav/files')
# data_loader = DataLoader(audio_dataset, batch_size=4, shuffle=True)

# Load pre-trained Wav2Vec 2.0 model
pretrained_model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name)
wav2vec2_model = Wav2Vec2Model.from_pretrained(pretrained_model_name)

# Define the number of emotion classes
num_classes = 7

# Create a new model class that adds a linear layer for emotion classification
class Wav2Vec2ForEmotionRecognition(nn.Module):
    def __init__(self, base_model, num_classes):
        super(Wav2Vec2ForEmotionRecognition, self).__init__()
        self.wav2vec2 = base_model
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_classes)
        
        # Initialize weights of the classifier
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, input_values):
        # Get the output from Wav2Vec 2.0 model
        outputs = self.wav2vec2(input_values)
        
        # Get the last hidden state
        hidden_states = outputs.last_hidden_state
        
        # Pooling: Mean pooling over the time dimension
        pooled_output = hidden_states.mean(dim=1)
        
        # Classify pooled output
        logits = self.classifier(pooled_output)
        
        return logits

# Instantiate the new model
model = Wav2Vec2ForEmotionRecognition(wav2vec2_model, num_classes)

# Print model architecture
print(model)

# # Usage
audio_dataset = AudioDataset('/path/to/your/wav/files')

# Split into train and validation sets?
train_size = int(0.8 * len(audio_dataset))
val_size = len(audio_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(audio_dataset, [train_size, val_size])
# Create train and validation data loaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# train_dataloader = DataLoader(audio_dataset, batch_size=4, shuffle=True)

# Optimizer and Loss Function
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

# Training Loop
num_epochs = 3

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Training
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        input_values = batch["input_values"].squeeze(1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        labels = batch["labels"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        optimizer.zero_grad()
        logits = model(input_values)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss}")
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_values = batch["input_values"].squeeze(1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = batch["labels"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            logits = model(input_values)
            _, predicted = torch.max(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy}")
