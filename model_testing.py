import librosa
import numpy as np
import pandas as pd
import os
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset

# Assuming you have a DataFrame with columns "filename" and "emotion"
# data = pd.read_csv("C:/MyDocs/DTU/MSc/Thesis/Data/MELD/MELD_preprocess_test/pre_process_test.csv")
# data = pd.read_csv(r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\custom_test\custom_labels_corrected.csv")
data = pd.read_csv(r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_testing.csv")

# directory = "C:/MyDocs/DTU/MSc/Thesis/Data/MELD/MELD_preprocess_test/MELD_preprocess_test_data"
# directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\custom_test\custom_test_data"
directory = r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_testing"

files = []

# Get a list of all files in the directory
for file in os.listdir(directory):
    if file.endswith('.wav'):
        files.append(file)

# Add filenames to a new column in the DataFrame
data['filename'] = files



features = []
labels = []

#my_encoding_dict = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
my_encoding_dict = {'ang': 0, 'dis': 1, 'fea': 2, 'hap': 3, 'neu': 4, 'sad': 5, 'sur': 6, 'fru': 7, 'exc': 8, 'oth': 9}
#my_encoding_dict = {'ang': 0, 'cal': 1, 'dis': 2, 'fea': 3, 'hap': 4, 'neu': 5, 'sad': 6, 'sur': 7}

labels = data['Emotion'].map(my_encoding_dict).values

# Print the classes in the order they were encountered
print(my_encoding_dict)


max_length = 16000 * 9  # 10 seconds

# Load the processor
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

for index, row in data.iterrows():

    # Load audio file
    file_to_load = row['filename']
    file_to_load_path = os.path.join(directory, file_to_load)
    # print()
    # print(index)
    # print(file_to_load)
    # print()

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

    print(type(inputs.input_values[0]))
    features.append(inputs.input_values[0])



# Convert labels to tensors
features_tensor = torch.stack(features)
labels_tensor = torch.tensor(labels).long()  # Use .long() for integer labels, .float() for one-hot

# Print the dimensions of the labels tensor
print(f"Labels tensor dimensions: {labels_tensor.shape}")

# Convert the TensorDatasets to Datasets
dataset = Dataset.from_dict({
    'input_values': features_tensor,
    'labels': labels_tensor
})

# Specify the batch size
batch_size = 10

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size)

# Initialize the model
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53",
                                                           num_labels=10
                                                          )
print("model loaded")

# Load the saved weights
model.load_state_dict(torch.load('emotion_recognition_model.pth', map_location=torch.device('cpu')))
print("model weights loaded")



model.eval()  # Set the model to evaluation mode
outputs = []
with torch.no_grad():  # Disable gradient calculations
    for batch in dataloader:
        # Get the input values and labels from the batch
        input_values = torch.stack(batch['input_values']).float()
        labels = batch['labels']

        
        print(f"Input values size: {input_values.size()}")  # Add this line
        input_values = input_values.transpose(0, 1)

        # Forward pass: compute the model outputs
        output = model(input_values, labels=labels)
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
label_names = list(my_encoding_dict.keys())

print(f"Predicted classes size:", predicted_classes.shape)
print(f"Labels tensor size:", labels_tensor.numpy().size)

# Print the predicted classes and the actual labels
for i, label in enumerate(labels_tensor.numpy()):
    print("Predicted:", label_names[predicted_classes[i]], "Actual:", label_names[label])

# Generate confusion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(labels_tensor.numpy(), predicted_classes)

confusion_matrix_full = np.zeros((len(label_names), len(label_names)), dtype=int)

# Fill the confusion matrix with the values from the actual confusion matrix
for i, label in enumerate(labels_tensor.numpy()):
    confusion_matrix_full[label, predicted_classes[i]] +=1

# Create a DataFrame for the confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix_full, index=label_names, columns=label_names)

# Add a row and column for the total counts
confusion_matrix_df['Total'] = confusion_matrix_df.sum(axis=1)
confusion_matrix_df.loc['Total'] = confusion_matrix_df.sum()

print("Confusion Matrix:")
print(confusion_matrix_df)


# Generate confusion matrix
# confusion_matrix = sklearn.metrics.confusion_matrix(labels_tensor.numpy(), predicted_classes)

# # Create a confusion matrix of shape (7, 7) filled with zeros
# confusion_matrix_full = np.zeros((8, 8), dtype=int)

# # Get the unique labels in the test data
# unique_labels = np.unique(labels_tensor.numpy())

# # Fill the confusion matrix with the values from the actual confusion matrix
# for i, label in enumerate(unique_labels):
#     confusion_matrix_full[label, unique_labels] = confusion_matrix[i, :len(unique_labels)]

# # Create a DataFrame for the confusion matrix
# confusion_matrix_df = pd.DataFrame(confusion_matrix_full, index=label_names, columns=label_names)

# # Add a row and column for the total counts
# confusion_matrix_df['Total'] = confusion_matrix_df.sum(axis=1)
# confusion_matrix_df.loc['Total'] = confusion_matrix_df.sum()

# print("Confusion Matrix:")
# print(confusion_matrix_df)
