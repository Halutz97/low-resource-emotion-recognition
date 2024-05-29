import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model

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
