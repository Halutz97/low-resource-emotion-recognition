import numpy as np
import pandas as pd
import os
import sklearn.metrics
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from speechbrain.inference.interfaces import Pretrained
from hyperpyyaml import load_hyperpyyaml

class EmotionRecognition(Pretrained):
    HPARAMS_NEEDED = ["modules", "pretrainer"]
    MODULES_NEEDED = ["encoder", "pooling", "regressor"]

    def __init__(self, modules=None, hparams=None, run_opts=None, freeze_params=True):
        super().__init__(modules, hparams, run_opts, freeze_params)
        
    def forward(self, wavs):
        """Forward pass for inference."""
        wavs, wav_lens = wavs.to(self.device), torch.tensor([1.0]).to(self.device)
        features = self.mods.encoder(wavs, wav_lens)
        pooled_features = self.hparams.avg_pool(features, wav_lens)  # Ensure pooling is included
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)
        predictions = self.mods.regressor(pooled_features)
        return predictions

    def predict_file(self, wav_path):
        """Predict emotion from an audio file."""
        wavs = self.load_audio(wav_path)
        wavs = wavs.squeeze(0) # Remove batch dimension if present
        wavs = torch.tensor(wavs).unsqueeze(0)
        predictions = self.forward(wavs)
        return predictions.squeeze().cpu().numpy()

    @staticmethod
    def load_audio(wav_path, sample_rate=16000):
        import torchaudio
        sig, fs = torchaudio.load(wav_path)
        if fs != sample_rate:
            sig = torchaudio.transforms.Resample(fs, sample_rate)(sig)
        return sig
    
def load_checkpoint_with_renamed_keys(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    renamed_checkpoint = {}
    
    for key, value in checkpoint.items():
        new_key = key.replace('0.w.', 'w.')  # Renaming logic
        renamed_checkpoint[new_key] = value
    
    model.load_state_dict(renamed_checkpoint, strict=False)
    return model


# Custom function to load the model from local files
def load_local_model(model_dir, hparams_file):
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Add device to hparams
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams["device"] = device

    modules = hparams["modules"]
    for module in modules.values():
        if module is not None:
            module.to(device)

    pretrainer = hparams.get("pretrainer", None)
    if pretrainer is not None:
        pretrainer.set_collect_in(model_dir)
        pretrainer.load_collected()

    # Load the regressor with renamed keys
    model = EmotionRecognition(modules, hparams)
    load_checkpoint_with_renamed_keys(os.path.join(model_dir, 'model.ckpt'), model.mods.regressor, device)
    
    return model

# Load the data
data = pd.read_csv(r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_attributes_corrected.csv")
directory = r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio"


files = []

# Get a list of all files in the directory
for file in os.listdir(directory):
    if file.endswith('.wav'):
        files.append(file)

# Define the true regression values
true_valence = data['Valence']
true_arousal = data['Arousal']
true_valence = [float(label) for label in true_valence]
true_arousal = [float(label) for label in true_arousal]

# Define the paths for the pretrained model and hparams file
model_dir = r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\CKPT_2024-05-26_10-59-44_00'
hparams_file = r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\old_hparams_inference.yaml'

# Load the model using the custom function
emotion_model = load_local_model(model_dir, hparams_file)


# Set the model to evaluation mode
emotion_model.mods.encoder.eval()
emotion_model.mods.pooling.eval()
emotion_model.mods.regressor.eval()

predicted_valence = []
predicted_arousal = []

for i, file in enumerate(files):

    predictions = emotion_model.predict_file(os.path.join(directory,file))
    predicted_valence.append(predictions[0])
    predicted_arousal.append(predictions[1])
    print(f"File: {i}")
    print(f"Predicted Valence: {predictions[0]}, Real Valence: {true_valence[i]}")
    print(f"Predicted Arousal: {predictions[1]}, Real Arousal: {true_arousal[i]}")


# Calculate the mean squared error 
mse_valence = sklearn.metrics.mean_squared_error(true_valence, predicted_valence)
mse_arousal = sklearn.metrics.mean_squared_error(true_arousal, predicted_arousal)

print(f"MSE Valence: {mse_valence}")
print(f"MSE Arousal: {mse_arousal}")

