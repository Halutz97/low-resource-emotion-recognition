import os
# import torch
import pandas as pd

directory = "C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_fine_tune_v1_train_data"
annotations = pd.read_csv(r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_fine_tune_v1_train_data\train_labels.csv")
# annotations['filename'] = annotations['filename'].apply(lambda x: os.path.join(directory, x))

print(annotations.head(10))