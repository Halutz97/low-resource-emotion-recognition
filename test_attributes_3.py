import numpy as np
import pandas as pd
import os
import sklearn.metrics
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import json
from speechbrain.inference.interfaces import Pretrained
from hyperpyyaml import load_hyperpyyaml

# class EmotionRecognition(Pretrained):
#     HPARAMS_NEEDED = ["modules", "pretrainer"]
#     MODULES_NEEDED = ["encoder", "pooling", "regressor"]

#     def __init__(self, modules=None, hparams=None, run_opts=None, freeze_params=True):
#         super().__init__(modules, hparams, run_opts, freeze_params)
        
#     def forward(self, wavs):
#         """Forward pass for inference."""
#         wavs, wav_lens = wavs.to(self.device), torch.tensor([1.0]).to(self.device)
#         features = self.mods.encoder(wavs, wav_lens)
#         pooled_features = self.hparams.avg_pool(features, wav_lens)  # Ensure pooling is included
#         pooled_features = pooled_features.view(pooled_features.shape[0], -1)
#         predictions = self.mods.regressor(pooled_features)
#         return predictions

#     def predict_file(self, wav_path):
#         """Predict emotion from an audio file."""
#         wavs = self.load_audio(wav_path)
#         wavs = wavs.squeeze(0) # Remove batch dimension if present
#         wavs = torch.tensor(wavs).unsqueeze(0)
#         predictions = self.forward(wavs)
#         return predictions.squeeze().cpu().numpy()

#     @staticmethod
#     def load_audio(wav_path, sample_rate=16000):
#         import torchaudio
#         sig, fs = torchaudio.load(wav_path)
#         if fs != sample_rate:
#             sig = torchaudio.transforms.Resample(fs, sample_rate)(sig)
#         return sig
    
# def load_checkpoint_with_renamed_keys(checkpoint_path, model, device):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     renamed_checkpoint = {}
    
#     for key, value in checkpoint.items():
#         new_key = key.replace('0.w.', 'w.')  # Renaming logic
#         renamed_checkpoint[new_key] = value
    
#     model.load_state_dict(renamed_checkpoint, strict=False)
#     return model


# # Custom function to load the model from local files
# def load_local_model(model_dir, hparams_file):
#     with open(hparams_file) as fin:
#         hparams = load_hyperpyyaml(fin)

#     # Add device to hparams
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     hparams["device"] = device

#     modules = hparams["modules"]
#     for module in modules.values():
#         if module is not None:
#             module.to(device)

#     pretrainer = hparams.get("pretrainer", None)
#     if pretrainer is not None:
#         pretrainer.set_collect_in(model_dir)
#         pretrainer.load_collected()

#     # Load the regressor with renamed keys
#     model = EmotionRecognition(modules, hparams)
#     load_checkpoint_with_renamed_keys(os.path.join(model_dir, 'model.ckpt'), model.mods.regressor, device)
    
#     return model

# # Load the data
# directory = r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio"


# files = []

# # Get a list of all files in the directory
# for file in os.listdir(directory):
#     if file.endswith('.wav'):
#         files.append(file)


# # Define the paths for the pretrained model and hparams file
# model_dir = r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\CKPT_2024-05-26_10-59-44_00'
# hparams_file = r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\old_hparams_inference.yaml'

# # Load the model using the custom function
# emotion_model = load_local_model(model_dir, hparams_file)


# # Set the model to evaluation mode
# emotion_model.mods.encoder.eval()
# emotion_model.mods.pooling.eval()
# emotion_model.mods.regressor.eval()

# # Load the test.json file
# with open(r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\test.json') as f:
#     test_data = json.load(f)


# predicted_valence = []
# predicted_arousal = []
# true_valence = []
# true_arousal = []

# # Iterate through the test data
# for utt_id, utt_data in test_data.items():
#     old_wav_path = utt_data['wav']
#     wav_path = os.path.join(directory, os.path.basename(old_wav_path))
#     val = float(utt_data['val'])
#     aro = float(utt_data['aro'])

#     # Predict the emotions
#     predictions = emotion_model.predict_file(wav_path)
#     predicted_valence.append(predictions[0])
#     predicted_arousal.append(predictions[1])
#     true_valence.append(val)
#     true_arousal.append(aro)

#     print(f"Utterance: {utt_id}")
#     print(f"Predicted Valence: {predictions[0]}, Real Valence: {val}")
#     print(f"Predicted Arousal: {predictions[1]}, Real Arousal: {aro}")

# Load the results
results = pd.read_csv(r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\results.csv')
true_valence = results['True Valence']
true_arousal = results['True Arousal']
predicted_valence = results['Predicted Valence']
predicted_arousal = results['Predicted Arousal']

# Calculate the mean squared error 
mse_valence = sklearn.metrics.mean_squared_error(true_valence, predicted_valence)
mse_arousal = sklearn.metrics.mean_squared_error(true_arousal, predicted_arousal)

print(f"MSE Valence: {mse_valence}")
print(f"MSE Arousal: {mse_arousal}")

total_mse = (mse_valence + mse_arousal) / 2
print(f"Total MSE: {total_mse}")

# # Save the results
# results = pd.DataFrame({'True Valence': true_valence, 'Predicted Valence': predicted_valence, 'True Arousal': true_arousal, 'Predicted Arousal': predicted_arousal})
# results.to_csv(r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\results.csv', index=False)

# Obtain maximums errors
max_error_valence = max(abs(np.array(true_valence) - np.array(predicted_valence)))
max_error_arousal = max(abs(np.array(true_arousal) - np.array(predicted_arousal)))
print()
print(f"Max Error Valence: {max_error_valence}")
print(f"Max Error Arousal: {max_error_arousal}")

# Calculate probability of getting an error of less than 0.1 and 0.5 for arousal and valence
valence_error = np.array(true_valence) - np.array(predicted_valence)
arousal_error = np.array(true_arousal) - np.array(predicted_arousal)

valence_prob = sum(abs(valence_error) < 0.1) / len(valence_error)
arousal_prob = sum(abs(arousal_error) < 0.1) / len(arousal_error)

print()
print(f"Probability of getting an error of less than 0.1 for valence: {valence_prob}")
print(f"Probability of getting an error of less than 0.1 for arousal: {arousal_prob}")

valence_prob = sum(abs(valence_error) < 0.5) / len(valence_error)
arousal_prob = sum(abs(arousal_error) < 0.5) / len(arousal_error)

print()
print(f"Probability of getting an error of less than 0.5 for valence: {valence_prob}")
print(f"Probability of getting an error of less than 0.5 for arousal: {arousal_prob}")

# Calculate mean and standard deviation of the errors
valence_mean = np.mean(valence_error)
valence_std = np.std(valence_error)
arousal_mean = np.mean(arousal_error)
arousal_std = np.std(arousal_error)

print()
print(f"Mean Valence Error: {valence_mean}")
print(f"Standard Deviation Valence Error: {valence_std}")
print(f"Mean Arousal Error: {arousal_mean}")
print(f"Standard Deviation Arousal Error: {arousal_std}")


# Plot the results

# Plot density of true labels
plt.figure(figsize=(12, 5))

# Density plot for true labels
plt.subplot(1, 3, 1)
sns.kdeplot(x=true_valence, y=true_arousal, cmap="Blues", fill=True, thresh=0)
plt.title('Density Plot of True Labels')
plt.xlabel('Valence')
plt.ylabel('Arousal')
plt.xlim(1, 5)
plt.ylim(1, 5)

# Density plot for true labels
plt.subplot(1, 3, 2)
sns.kdeplot(x=predicted_valence, y=predicted_arousal, cmap="Oranges", fill=True, thresh=0)
plt.title('Density Plot of Predicted Labels')
plt.xlabel('Valence')
plt.ylabel('Arousal')
plt.xlim(1, 5)
plt.ylim(1, 5)

# Scatter plot for true and predicted labels
plt.subplot(1, 3, 3)
plt.scatter(true_valence, true_arousal, label='True', alpha=0.6)
plt.scatter(predicted_valence, predicted_arousal, label='Predicted', alpha=0.6)
plt.legend()
plt.title('Valence-Arousal Prediction')
plt.xlabel('Valence')
plt.ylabel('Arousal')
plt.xlim(1, 5)
plt.ylim(1, 5)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
# Plot error distribution for valence
plt.subplot(1, 2, 1)
sns.histplot(np.array(true_valence) - np.array(predicted_valence), kde=True, stat='density')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Valence Error Distribution')

# Plot error distribution for arousal
plt.subplot(1, 2, 2)
sns.histplot(np.array(true_arousal) - np.array(predicted_arousal), kde=True, stat='density')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Arousal Error Distribution')

plt.tight_layout()
plt.show()





