import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, make_scorer
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import TensorDataset, DataLoader  # Added missing imports
from pytorch_lightning.callbacks import EarlyStopping  # Added missing import
from pytorch_tabnet.tab_model import TabNetRegressor  # Import TabNet

# Loading Dataset
#data = pd.read_csv("/content/drive/MyDrive/ECE324_Project/Model/dataset.csv") # change path for your env
data = pd.read_csv("SmartStudy\\notebooks\\database.csv") # change path for your env
data.head()

# Data Splitting & Normalization
scaler = StandardScaler()
input = data.drop(columns=['GPA'], errors='ignore')
input = scaler.fit_transform(input)
labels = data['GPA']
X_train, X_temp, Y_train, Y_temp = train_test_split(input, labels, test_size=0.3, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)


class LightweightTabPFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_heads=2, num_layers=2, output_dim=1):
        super(LightweightTabPFN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Input embedding layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output regression layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input embedding
        x = self.input_layer(x)

        # Add a sequence dimension for the transformer
        x = x.unsqueeze(1)  # Shape: (batch_size, seq_len=1, hidden_dim)

        # Transformer encoder
        x = self.transformer(x)

        # Remove sequence dimension
        x = x.squeeze(1)  # Shape: (batch_size, hidden_dim)

        # Output regression
        x = self.output_layer(x)
        return x

# Example usage
def train_lightweight_tabpfn(X_train, Y_train, X_val, Y_val, input_dim, epochs=50, batch_size=32):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    # Convert labels to numpy arrays before creating PyTorch tensors
    Y_train_tensor = torch.tensor(Y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    # Convert labels to numpy arrays before creating PyTorch tensors
    Y_val_tensor = torch.tensor(Y_val.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss, and optimizer
    model = LightweightTabPFN(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, Y_batch)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}")

    return model

# Example call (replace X_train, Y_train, X_val, Y_val with actual data)
# model = train_lightweight_tabpfn(X_train, Y_train, X_val, Y_val, input_dim=10)
# Train the LightweightTabPFN model
input_dim = X_train.shape[1]
model = train_lightweight_tabpfn(X_train, Y_train, X_val, Y_val, input_dim=input_dim, epochs=50, batch_size=32)

# Test the model
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze().numpy()

# Evaluate the model
mse = mean_squared_error(Y_test, predictions)
mae = mean_absolute_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R2 Score: {r2:.4f}")
