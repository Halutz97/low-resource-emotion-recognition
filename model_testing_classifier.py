import librosa
import numpy as np
import pandas as pd
import os
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2Model
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset

    
class CustomWav2Vec2ForSequenceClassification(Wav2Vec2ForSequenceClassification):
    def __init__(self, config, num_linear_layers=1):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Ensure the custom classifier matches the trained model
        layers = [nn.Linear(config.hidden_size, config.hidden_size) for _ in range(num_linear_layers - 1)]
        layers += [nn.Linear(config.hidden_size, config.num_labels)]
        dropout_rate = getattr(config, 'hidden_dropout_prob', 0.1)  # Match training setup
        self.classifier = nn.Sequential(*layers, nn.Dropout(dropout_rate))

    def forward(self, input_values, attention_mask=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output.mean(dim=1))  # Applying mean pooling as in training

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)  # Assuming weight setup as in training
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 

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


#my_encoding_dict = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3, 'sur': 4, 'fru': 5, 'exc': 6}
my_encoding_dict = {'ang': 0, 'hap': 1}

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


# Initialize the configuration
config = Wav2Vec2Config.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    hidden_size=1024,  # Ensure this matches the trained model
    num_labels=2,      # Ensure this matches the trained model
    hidden_dropout_prob=0.1
)

# Initialize the model
model = CustomWav2Vec2ForSequenceClassification(config, num_linear_layers=1)
print("model loaded")


# Load the saved weights
model.load_state_dict(torch.load('model/emotion_recognition_model_y6n1q2p1.pth', map_location=torch.device('cpu')))
print("model weights loaded")

# # Dummy data for demonstration; replace with actual evaluation data
# dummy_input = torch.rand(1, 16000)  # Sample input; adjust size and content as needed

# # Evaluate the model
# with torch.no_grad():
#     output = model(dummy_input)
#     print("Output from model:", output.logits)

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
        output = model(input_values)
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

# Get the label names from the label encoder
label_names = list(my_encoding_dict.keys())

true_labels = labels_tensor.numpy()

# Print the predicted classes and the actual labels
for i, label in enumerate(true_labels):
    print("Predicted:", label_names[predicted_classes[i]], "Actual:", label_names[label])


# Calculate accuracy
accuracy = (predicted_classes == true_labels).mean()
print("Accuracy:", accuracy)

# Calculate F1 Scores
f1_micro = sklearn.metrics.f1_score(true_labels, predicted_classes, average='micro')
print("F1 Score (Micro):", f1_micro)

f1_macro = sklearn.metrics.f1_score(true_labels, predicted_classes, average='macro')
print("F1 Score (Macro):", f1_macro)

f1_weighted = sklearn.metrics.f1_score(true_labels, predicted_classes, average='weighted')
print("F1 Score (Weighted):", f1_weighted)


# Generate confusion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, predicted_classes)

confusion_matrix_full = np.zeros((len(label_names), len(label_names)), dtype=int)

# Fill the confusion matrix with the values from the actual confusion matrix
for i, label in enumerate(true_labels):
    confusion_matrix_full[label, predicted_classes[i]] +=1

# Create a DataFrame for the confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix_full, index=label_names, columns=label_names)

# Add a row and column for the total counts
confusion_matrix_df['Total'] = confusion_matrix_df.sum(axis=1)
confusion_matrix_df.loc['Total'] = confusion_matrix_df.sum()

print("Confusion Matrix:")
print(confusion_matrix_df)