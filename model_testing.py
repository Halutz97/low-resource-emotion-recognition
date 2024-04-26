import librosa
import numpy as np
import pandas as pd
import os
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import Wav2Vec2ForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

# Assuming you have a DataFrame with columns "filename" and "emotion"
# data = pd.read_csv("C:/MyDocs/DTU/MSc/Thesis/Data/MELD/MELD_preprocess_test/pre_process_test.csv")
data = pd.read_csv("C:/Users/DANIEL/Desktop/thesis/low-resource-emotion-recognition/MELD_preprocess_test/pre_process_test.csv")

# directory = "C:/MyDocs/DTU/MSc/Thesis/Data/MELD/MELD_preprocess_test/MELD_preprocess_test_data"
directory = "C:/Users/DANIEL/Desktop/thesis/low-resource-emotion-recognition/MELD_preprocess_test/MELD_preprocess_test_data"

files = []

# Get a list of all files in the directory
for file in os.listdir(directory):
    if file.endswith('.wav'):
        files.append(file)

# Add filenames to a new column in the DataFrame
data['filename'] = files



features = []
labels = []

label_encoder = LabelEncoder()

raw_labels = data['Emotion'].values
labels = label_encoder.fit_transform(raw_labels)


max_length = 16000 * 9  # 10 seconds

for index, row in data.iterrows():

    # Load audio file
    file_to_load = row['filename']
    file_to_load_path = os.path.join(directory, file_to_load)
    # print()
    # print(index)
    # print(file_to_load)
    # print()

    audio, sr = librosa.load(file_to_load_path, sr=16000)

    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        padding = max_length - len(audio)
        offset = padding // 2
        audio = np.pad(audio, (offset, padding - offset), 'constant')

    # Append raw audio data
    features.append(audio)

    # Encode label
    # labels.append(label_encoder.transform([row['Emotion']]))

# Convert to arrays
features = np.array(features)
labels = np.array(labels).flatten()


# Now, `features` and `labels` can be used for training your model
# Optionally, save them to disk
# np.save('features.npy', features)
# np.save('labels.npy', labels)

print(features.shape)
print(labels.shape)

# Convert features and labels into PyTorch tensors
features_tensor = torch.tensor(features).float()
labels_tensor = torch.tensor(labels).long()  # Use .long() for integer labels, .float() for one-hot

# Reshape features_tensor to 2D (batch_size, sequence_length)
features_tensor = features_tensor.view(features_tensor.shape[0], -1)


dataset = TensorDataset(features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=16)  # Adjust batch size as needed

# Initialize the model
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53", num_labels=7)

# Load the saved weights
model.load_state_dict(torch.load('emotion_recognition_model.pth', map_location=torch.device('cpu')))

model.eval()  # Set the model to evaluation mode
outputs = []
with torch.no_grad():  # Disable gradient calculations
    for features, labels in dataloader:
        inputs = {'input_values': features, 'labels': labels}
        output = model(**inputs)  # Get model outputs for a batch
        outputs.append(output)

# Print one of the logits as an example
print(outputs[0].logits)


# The outputs are logits, convert them to probabilities using softmax
probabilities = [torch.nn.functional.softmax(output.logits, dim=-1) for output in outputs]

# Print one of the probabilities as an example
print(probabilities[0])

# Get the predicted class
predicted_classes = [torch.argmax(prob, dim=-1) for prob in probabilities]
# Convert predicted_classes to a numpy array
predicted_classes = torch.cat(predicted_classes).numpy()

# Calculate accuracy
accuracy = (predicted_classes == labels_tensor.numpy()).mean()
print("Accuracy:", accuracy)

# Get the label names from the label encoder
label_names = label_encoder.classes_


# Generate confusion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(labels_tensor.numpy(), predicted_classes)

# Create a DataFrame for the confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix, index=label_names, columns=label_names)

# Add a row and column for the total counts
confusion_matrix_df['Total'] = confusion_matrix_df.sum(axis=1)
confusion_matrix_df.loc['Total'] = confusion_matrix_df.sum()

print("Confusion Matrix:")
print(confusion_matrix_df)
