import numpy as np
import pandas as pd
import os
import sklearn.metrics
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import json
import torchaudio
from speechbrain.inference.interfaces import Pretrained
from hyperpyyaml import load_hyperpyyaml

class EmotionRecognition(Pretrained):
    HPARAMS_NEEDED = ["modules", "pretrainer"]
    MODULES_NEEDED = ["encoder", "pooling", "regressor", "classifier"]

    def __init__(self, modules=None, hparams=None, run_opts=None, freeze_params=True):
        super().__init__(modules, hparams, run_opts, freeze_params)
        
    def forward(self, wavs):
        """Forward pass for inference."""
        wavs, wav_lens = wavs.to(self.device), torch.tensor([1.0]).to(self.device)
        features = self.mods.encoder(wavs, wav_lens)
        pooled_features = self.hparams.avg_pool(features, wav_lens)  # Ensure pooling is included
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)
        regressor_predictions = self.mods.regressor(pooled_features)
        classifier_predictions = self.mods.classifier(pooled_features)
        return regressor_predictions, classifier_predictions

    def predict_file(self, wav_path):
        """Predict emotion from an audio file."""
        wavs = self.load_audio(wav_path)
        wavs = wavs.squeeze(0) # Remove batch dimension if present
        wavs = torch.tensor(wavs).unsqueeze(0)
        regression_predictions, classifier_predictions = self.forward(wavs)
        classifier_predictions = torch.log_softmax(classifier_predictions, dim=-1)
        return regression_predictions.squeeze().cpu().numpy(), classifier_predictions.squeeze().cpu().numpy()

    @staticmethod
    def load_audio(wav_path, sample_rate=16000):
        """Load an audio file and resample if needed."""
        print(wav_path)
        sig, fs = torchaudio.load(wav_path)
        if fs != sample_rate:
            sig = torchaudio.transforms.Resample(fs, sample_rate)(sig)
        return sig
    
def load_checkpoint_with_renamed_keys(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    renamed_checkpoint = {}

    # Rename keys according to the expected names in the model's state_dict
    for key, value in checkpoint.items():
        new_key = key.replace('0.w.', 'w.')  # Adjust this based on your specific needs
        renamed_checkpoint[new_key] = value

    # Load the state_dict with the renamed keys
    model.load_state_dict(renamed_checkpoint, strict=False)
    return model


def load_local_model(model_dir, hparams_file):
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams["device"] = device

    modules = hparams["modules"]

    # Instantiate the model
    model = EmotionRecognition(modules, hparams)

    # Load checkpoint of the heads
    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
    checkpoint_heads = torch.load(checkpoint_path, map_location=device)
    
    # Prepare a new state dictionary for the regressor and classifier
    new_state_dict = {}
    for key, param in checkpoint_heads.items():
        if key.startswith('0.'):  # Weights for the regressor
            new_key = key.replace('0.', '')  # Remove the '0.' prefix
            new_state_dict['mods.regressor.' + new_key] = param
        elif key.startswith('1.'):  # Weights for the classifier
            new_key = key.replace('1.', '')  # Remove the '1.' prefix
            new_state_dict['mods.classifier.' + new_key] = param

    # Load the new state dict into the model
    model.load_state_dict(new_state_dict, strict=False)

    # Load checkpoint of the encoder
    checkpoint_path = os.path.join(model_dir, 'wav2vec2.ckpt')
    checkpoint_encoder = torch.load(checkpoint_path, map_location=device)

    # Load the encoder state dict
    model.mods.encoder.load_state_dict(checkpoint_encoder, strict=False)

    missing_keys, unexpected_keys = model.mods.encoder.load_state_dict(checkpoint_encoder, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    print()
    for name, param in model.named_parameters():
        print(name)


    return model



# Main code
# Define the dataset
dataset = "CREMA-D"

# Variables
my_encoding_dict_model = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3}
label_names = ['neu', 'ang', 'hap', 'sad']
label_names_model = ['ang', 'hap', 'neu', 'sad']

# Load the data
if dataset != "IEMOCAP-test" and dataset != "IEMOCAP":
    dimensions = False

    if dataset == "CREMA-D":
        data = pd.read_csv(r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\labels_testing.csv")
        directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\audio_testing"
        my_encoding_dict_dataset = {'NEU': 0, 'ANG': 1, 'HAP': 2, 'SAD': 3}

    elif dataset == "CREMA-D-voted":
        data = pd.read_csv(r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\labels_v_testing.csv")
        directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\audio_v_testing"
        my_encoding_dict_dataset = {'N': 0, 'A': 1, 'H': 2, 'S': 3}

    elif dataset == "EMO-DB":
        data = pd.read_csv(r"C:\Users\DANIEL\Desktop\thesis\EmoDB\labels_testing.csv")
        directory = r"C:\Users\DANIEL\Desktop\thesis\EmoDB\audio_testing"
        my_encoding_dict_dataset = {'N': 0, 'W': 1, 'F': 2, 'T': 3}

    elif dataset == "ShEMO":
        data = pd.read_csv(r"C:\Users\DANIEL\Desktop\thesis\ShEMO\labels_testing.csv")
        directory = r"C:\Users\DANIEL\Desktop\thesis\ShEMO\audio_testing"
        my_encoding_dict_dataset = {'N': 0, 'A': 1, 'H': 2, 'S': 3}

    else:
        raise ValueError("Invalid dataset")


    # Get a list of all files in the directory
    files = []
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            files.append(os.path.join(directory,file))

    true_labels = data['Emotion']
    true_valence = np.zeros(len(true_labels))
    true_arousal = np.zeros(len(true_labels))
    emotion_keys = true_labels.map(my_encoding_dict_dataset).values

elif dataset == "IEMOCAP":
    dimensions = True
    data = pd.read_csv(r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_testing.csv")
    directory = r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_testing"
    my_encoding_dict_dataset = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3}

    # Get a list of all files in the directory
    files = []
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            files.append(os.path.join(directory,file))

    true_labels = data['Emotion']
    true_valence = data['Valence']
    true_arousal = data['Arousal']
    emotion_keys = true_labels.map(my_encoding_dict_dataset).values

else:
    dimensions = True
    with open(r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\multiobjectif\test.json') as f:
        test_data = json.load(f)
    directory = r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio"
    my_encoding_dict_dataset = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3}

    # Get a list of all files in the directory
    files = []
    true_valence = []
    true_arousal = []
    true_labels = []
    emotion_keys = []
    for utt_id, utt_data in test_data.items():
        old_wav_path = utt_data['wav']
        files.append(os.path.join(directory, os.path.basename(old_wav_path)))
        true_valence.append(float(utt_data['val']))
        true_arousal.append(float(utt_data['aro']))
        true_labels.append(utt_data['emo'])
        emotion_keys.append(my_encoding_dict_model[utt_data['emo']])



# Define the paths for the pretrained model and hparams file
model_dir = r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\multiobjectif'
hparams_file = r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\hparams_inference.yaml'

# Load the model using the custom function
emotion_model = load_local_model(model_dir, hparams_file)


# Set the model to evaluation mode
emotion_model.mods.encoder.eval()
emotion_model.mods.pooling.eval()
emotion_model.mods.regressor.eval()
emotion_model.mods.classifier.eval()


predicted_valence = []
predicted_arousal = []
predicted_emotions = []
predicted_probabilites = []
score_emotions = []

# Iterate through the test data
for index, file in enumerate(files):

    # Predict the emotions
    regression_predictions, classifier_predictions = emotion_model.predict_file(file)
    predicted_valence.append(regression_predictions[0])
    predicted_arousal.append(regression_predictions[1])

    # Get the predicted emotion label
    predicted_emotion = classifier_predictions.argmax()
    
    # map the predicted emotion to the correct label
    predicted_emotion = label_names_model[predicted_emotion]
    predicted_emotion = my_encoding_dict_model[predicted_emotion]

    score_emotions.append(classifier_predictions)
    predicted_emotions.append(predicted_emotion)

    print(f"Utterance: {index}")
    print(f"Predicted Valence: {regression_predictions[0]}, True Valence: {true_valence[index]}")
    print(f"Predicted Arousal: {regression_predictions[1]}, True Arousal: {true_arousal[index]}")
    print(f"Predicted Emotion: {predicted_emotion}, True Emotion: {emotion_keys[index]}")

# Save the results
results = pd.DataFrame({'Predicted Valence': predicted_valence, 
                        'True Valence': true_valence,
                        'Predicted Arousal': predicted_arousal, 
                        'True Arousal': true_arousal,
                        'Predicted Emotion': predicted_emotions,
                        'True Emotion': true_labels,
                        'Emotion Labels': emotion_keys,
                        'Score Emotions': score_emotions,
                        })
results.to_csv(r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\results_multi_' + dataset +'csv', index=False)



# Testing the model
label_names = ['neu', 'ang', 'hap', 'sad']
dimensions = False
dataset = "CREMA-D"

# Load the results
results = pd.read_csv(r'C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition\speechbrain_model\results_multi_' + dataset +'csv')
true_valence = results['True Valence']
true_arousal = results['True Arousal']
predicted_valence = results['Predicted Valence']
predicted_arousal = results['Predicted Arousal']
true_labels = results['True Emotion']
emotion_keys = results['Emotion Labels']
predicted_keys = results['Predicted Emotion']
score_emotions = results['Score Emotions']

# Pass all emotion scores as float arrays
score_emotions = np.array([np.array(x[1:-1].split()).astype(float) for x in score_emotions])

# Calculate the accuracy and nll of the emotion classification
emotion_keys = np.array(emotion_keys)
emotion_scores = np.array(emotion_keys)
predicted_keys = np.array(predicted_keys)



if dimensions:
    # Calculate the mean squared error 
    mse_valence = sklearn.metrics.mean_squared_error(true_valence, predicted_valence)
    mse_arousal = sklearn.metrics.mean_squared_error(true_arousal, predicted_arousal)

    print(f"MSE Valence: {mse_valence}")
    print(f"MSE Arousal: {mse_arousal}")

    total_mse = (mse_valence + mse_arousal) / 2
    print(f"Total MSE: {total_mse}")

    nll = sklearn.metrics.log_loss(emotion_keys, score_emotions)
    print(f"NLL: {nll}")


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


    # Calculate the correlation between true and predicted valence and arousal
    valence_corr = np.corrcoef(true_valence, predicted_valence)[0, 1]
    arousal_corr = np.corrcoef(true_arousal, predicted_arousal)[0, 1]

    print()
    print(f"Correlation between true and predicted valence: {valence_corr}")
    print(f"Correlation between true and predicted arousal: {arousal_corr}")


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

    plt.figure(figsize=(12, 5))
    # Plot correlation between true and predicted valence
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=true_valence, y=predicted_valence, alpha=0.6)
    plt.xlabel('True Valence')
    plt.ylabel('Predicted Valence')
    plt.title('True vs Predicted Valence')
    plt.xlim(1, 5)
    plt.ylim(1, 5)

    # Plot correlation between true and predicted arousal
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=true_arousal, y=predicted_arousal, alpha=0.4)
    plt.xlabel('True Arousal')
    plt.ylabel('Predicted Arousal')
    plt.title('True vs Predicted Arousal')
    plt.xlim(1, 5)
    plt.ylim(1, 5)


    plt.tight_layout()
    plt.show()


# Calculate accuracy
accuracy = (predicted_keys == emotion_keys).mean()
print()
print()
print()
print("Accuracy:", accuracy)

# Calculate F1 Scores
f1_micro = sklearn.metrics.f1_score(emotion_keys, predicted_keys, average='micro')
print("F1 Score (Micro):", f1_micro)

f1_macro = sklearn.metrics.f1_score(emotion_keys, predicted_keys, average='macro')
print("F1 Score (Macro):", f1_macro)

f1_weighted = sklearn.metrics.f1_score(emotion_keys, predicted_keys, average='weighted')
print("F1 Score (Weighted):", f1_weighted)


# Generate confusion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(emotion_keys, predicted_keys)

confusion_matrix_full = np.zeros((len(label_names), len(label_names)), dtype=int)

# Fill the confusion matrix with the values from the actual confusion matrix
for i, label in enumerate(true_labels):
    confusion_matrix_full[emotion_keys[i], predicted_keys[i]] +=1

# Create a DataFrame for the confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix_full, index=label_names, columns=label_names)

# Normalize each row by its sum to get the percentage
confusion_matrix_percent = confusion_matrix_df.div(confusion_matrix_df.sum(axis=1), axis=0) * 100

# Create a DataFrame for the normalized confusion matrix
confusion_matrix_df_percent = pd.DataFrame(confusion_matrix_percent, index=label_names, columns=label_names)

print("Confusion Matrix:")
print(confusion_matrix_df_percent)

# Calculate the maximum value for the heatmap color scale
max_value = 100

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix_df_percent, annot=True, fmt='.2f', cmap='Blues', xticklabels=label_names, yticklabels=label_names, vmin=0, vmax=max_value)
plt.title('Confusion Matrix - ' + dataset)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



