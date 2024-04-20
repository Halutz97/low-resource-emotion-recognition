import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Script to plot the training loss curve
# Load data from csv file and plot the training loss curve
data = pd.read_csv(r"C:\MyDocs\DTU\MSc\Thesis\Fine tuning\training_loss_test.csv")

# show data
print(data.head())

# Load dataframe columns into np arrays:
steps = data['Step'].values
losses = data['Training Loss'].values

# Plot step vs loss
plt.figure(figsize=(10, 5))  # Set the figure size as needed
plt.plot(steps, losses, label='Training Loss', color='blue')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True)
plt.title('Training Loss Curve')
plt.savefig('training_loss.png')
plt.show()
# save as png


