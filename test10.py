import numpy as np
import pandas as pd
import os
import torch
import sklearn.metrics
from speechbrain.inference.interfaces import foreign_class

data = pd.read_csv(r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_testing.csv")

# directory = "C:/MyDocs/DTU/MSc/Thesis/Data/MELD/MELD_preprocess_test/MELD_preprocess_test_data"
# directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\custom_test\custom_test_data"
# directory = r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_testing"
directory = r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_testing"
# directory = r"C:\Users\DANIEL\Desktop\thesis\low-resource-emotion-recognition"

files = []

# Get a list of all files in the directory
for file in os.listdir(directory):
    if file.endswith('.wav'):
        files.append(file)

my_encoding_dict = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3}
label_names = ['neu', 'ang', 'hap', 'sad']
true_labels = data['Emotion']
label_keys = true_labels.map(my_encoding_dict).values

predicted_classes=[]
classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

for i, file in enumerate(files):

    out_prob, score, index, text_lab = classifier.classify_file(os.path.join(directory,file))
    predicted_classes.append(text_lab[0])

# print("out prob:", out_prob)
# print("score:", score)
# print("index:", index)
# print("text lab:", text_lab)
# print("type of text lab:", type(text_lab))

predicted_keys = [my_encoding_dict[keys] for keys in predicted_classes]

# Print the predicted classes and the actual labels
for i, prediction in enumerate(predicted_classes):
    print("Predicted:", prediction, "Actual:", true_labels[i])


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
    confusion_matrix_full[label_keys[i], predicted_keys[i]] +=1

# Create a DataFrame for the confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix_full, index=label_names, columns=label_names)

# Add a row and column for the total counts
confusion_matrix_df['Total'] = confusion_matrix_df.sum(axis=1)
confusion_matrix_df.loc['Total'] = confusion_matrix_df.sum()

print("Confusion Matrix:")
print(confusion_matrix_df)
