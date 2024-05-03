import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.utils.data import DataLoader
from audio_dataset import AudioDataset

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')

# Initialize the dataset
audio_dataset = AudioDataset(directory='path_to_your_audio_files', processor=processor)

# Create DataLoader
data_loader = DataLoader(audio_dataset, batch_size=10, shuffle=True)

# Tell it to disregard inbuilt classifier?

features = []
model.eval()
with torch.no_grad():
    for batch in data_loader:
        input_values = batch.squeeze(1)  # Adjust dimensions if necessary
        outputs = model(input_values).last_hidden_state
        features.append(outputs)

# `features` now contains the extracted features for each audio file



