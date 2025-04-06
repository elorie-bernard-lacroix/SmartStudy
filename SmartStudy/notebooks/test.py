import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score

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

# Loading Dataset
data = pd.read_csv("/content/drive/MyDrive/ECE324_Project/Model/dataset.csv") # change path for your env
#data = pd.read_csv("database.csv") # change path for your env
data.head()

# Data Splitting & Normalization
scaler = StandardScaler()
input = data.drop(columns=['GPA'], errors='ignore')
input = scaler.fit_transform(input)
labels = data['GPA']
X_train, X_temp, Y_train, Y_temp = train_test_split(input, labels, test_size=0.3, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

class SoftOrdering1DCNN(pl.LightningModule):

    def __init__(self, input_dim, output_dim=1, sign_size=32, cha_input=16, cha_hidden=32, 
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2, learning_rate=1e-3):
        super().__init__()

        hidden_size = sign_size * cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size // 2
        output_size = (sign_size // 4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output
        self.learning_rate = learning_rate

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = nn.Conv1d(
            cha_input, 
            cha_input * K, 
            kernel_size=5, 
            stride=1, 
            padding=2,  
            groups=cha_input, 
            bias=False)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input * K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input * K, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer (Output layer)
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_output)
        self.dense2 = nn.Linear(output_size, output_dim)
        
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = torch.relu(self.dense1(x))
        
        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1) 
        
        x = self.batch_norm_c1(x)
        x = torch.relu(self.conv1(x))
        
        x = self.ave_po_c1(x)
        
        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = torch.relu(self.conv2(x))
        
        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = x.view(x.size(0), -1) 
        x = self.dense2(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)  # Using MSE loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_mse', self.mse(y_hat, y))
        self.log('val_mae', self.mae(y_hat, y))
        self.log('val_r2', self.r2(y_hat, y))
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

# Assuming X_train, Y_train, X_test, Y_test are your data

# 1. Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).reshape(-1, 1) 
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).reshape(-1, 1) 

# 2. Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=32) 

# 3. Instantiate the model
input_dim = X_train_tensor.shape[1]  
model = SoftOrdering1DCNN(input_dim=input_dim)

# 4. Configure Trainer and callbacks
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)  
trainer = pl.Trainer(max_epochs=50, callbacks=[early_stopping]) 

# 5. Train the model
trainer.fit(model, train_loader, test_loader)  # Use train and validation loaders

# 6. Make predictions and evaluate
# Get the actual output size after convolutions and reshaping
# Pass a sample input through the model to get the output shape
sample_input = X_test_tensor[0].unsqueeze(0)  # Take one sample from X_test_tensor
output_shape = model(sample_input).shape

# The output size you need is the number of features in the output
output_size = output_shape[1]  

# Modify the `dense2` layer in your model to match the correct input size
model.dense2 = nn.Linear(output_size, 1)  # 1 output for regression

predictions = torch.cat([model(x) for x, _ in test_loader]).detach().numpy() 

# 7. Calculate and print evaluation metrics
mse = mean_squared_error(Y_test, predictions)
mae = mean_absolute_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)
kendall_tau_corr, _ = kendalltau(Y_test, predictions)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R2 Score:', r2)
print('Kendall Tau:', kendall_tau_corr)