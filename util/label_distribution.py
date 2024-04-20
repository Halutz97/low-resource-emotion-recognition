# Script to plot the label distribution
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from csv file and plot the label distribution
data = pd.read_csv(r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\dev_labels_corrected.csv")

# plot label distribution
plt.figure(figsize=(10, 5))  # Set the figure size as needed
data['Emotion'].value_counts().plot(kind='bar')
plt.xlabel('Emotion')
plt.ylabel('Frequency')
plt.title('Label Distribution')
plt.savefig('label_distribution.png')
plt.show()
# save as png
