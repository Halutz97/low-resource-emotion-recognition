import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FusionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FusionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
    
class EnhancedFusionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedFusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def fuse_score(train_dataset, val_dataset, model_save_path="/content/fusion_model.pth", batch_size=32, num_epochs=100, learning_rate=0.0001):
    # Create DataLoaders for the training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_dataset[0][0].shape[0]  # Number of features in combined input
    output_dim = 6  # Number of emotional classes

    # Initialize the fusion model
    fusion_model = EnhancedFusionModel(input_dim, output_dim)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        fusion_model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = fusion_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        training_losses.append(train_loss)

        fusion_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = fusion_model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        validation_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(fusion_model.state_dict(), model_save_path)
            print(f"Model weights saved to {model_save_path} with validation loss: {val_loss:.4f}")

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Load the best model weights
    fusion_model.load_state_dict(torch.load(model_save_path))

    return fusion_model, training_losses, validation_losses

# Example usage:
# Prepare your training and validation datasets
# train_dataset = TensorDataset(train_features, train_labels)
# val_dataset = TensorDataset(val_features, val_labels)
# model_save_path = "path/to/save/fusion_model.pth"
# fusion_model, training_losses, validation_losses = fuse_score(train_dataset, val_dataset, model_save_path, batch_size=32, num_epochs=100, learning_rate=0.0001)

