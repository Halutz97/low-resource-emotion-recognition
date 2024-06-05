import numpy as np
import pandas as pd
import os
import sklearn.metrics
import seaborn as sns
import matplotlib.pyplot as plt
from speechbrain.inference.interfaces import foreign_class

dataset = "CREMA-D"

if dataset == "IEMOCAP":
    # data = pd.read_csv(r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_testing.csv")
    # directory = r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_testing"
    data = pd.read_csv(r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_testing.csv")
    directory = r"C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_testing"
    my_encoding_dict_dataset = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3}

elif dataset == "CREMA-D":
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


# directory = "C:/MyDocs/DTU/MSc/Thesis/Data/MELD/MELD_preprocess_test/MELD_preprocess_test_data"
# directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\custom_test\custom_test_data"


files = []

# Get a list of all files in the directory
for file in os.listdir(directory):
    if file.endswith('.wav'):
        files.append(file)

my_encoding_dict_model = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3}
label_names = ['neu', 'ang', 'hap', 'sad']
true_labels = data['Emotion']
label_keys = true_labels.map(my_encoding_dict_dataset).values

predicted_classes=[]
classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

for i, file in enumerate(files):

    out_prob, score, index, text_lab = classifier.classify_file(os.path.join(directory,file))
    predicted_classes.append(text_lab[0])
    print("Utterance: ", i,"/",len(files)," Predicted: ", text_lab[0], " Actual: ", true_labels[i])

print("out prob:", out_prob)
print("score:", score)
print("index:", index)
print("text lab:", text_lab)
print("type of text lab:", type(text_lab))

predicted_keys = [my_encoding_dict_model[keys] for keys in predicted_classes]


# Calculate accuracy
accuracy = (predicted_keys == label_keys).mean()
print("Accuracy:", accuracy)

# Calculate F1 Scores
f1_micro = sklearn.metrics.f1_score(label_keys, predicted_keys, average='micro')
print("F1 Score (Micro):", f1_micro)

f1_macro = sklearn.metrics.f1_score(label_keys, predicted_keys, average='macro')
print("F1 Score (Macro):", f1_macro)

f1_weighted = sklearn.metrics.f1_score(label_keys, predicted_keys, average='weighted')
print("F1 Score (Weighted):", f1_weighted)


# Generate confusion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(label_keys, predicted_keys)

confusion_matrix_full = np.zeros((len(label_names), len(label_names)), dtype=int)

# Fill the confusion matrix with the values from the actual confusion matrix
for i, label in enumerate(true_labels):
    confusion_matrix_full[label_keys[i], predicted_keys[i]] +=1

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
