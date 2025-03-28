# neural_model/train.py

"""
This script trains a neural network (SocialPolicyPredictor) to predict social policy values —
namely meritocracy, fairness, and efficiency — based on demographic and socioeconomic
features from the UCI Adult dataset.

Each row in the dataset represents an individual. For each person, we compute a vector of
three social scores that describe how their life conditions relate to societal values. The model
learns to map input features (like age, education, gender) to this policy vector.

The trained model is used later in the project to generate proposed social policies that are
evaluated by simulated LLM agents.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from neural_model.transformer import SocialPolicyPredictor
from neural_model.metrics import compute_metrics

# Load UCI Adult Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]
data = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
data.dropna(inplace=True)

# Encode categorical variables
data_encoded = data.copy()
for col in columns:
    data_encoded[col] = pd.Categorical(data_encoded[col]).codes

# Compute societal metrics per row
y = compute_metrics(data_encoded)  # shape: (n_samples, 3)

# Feature selection
X = data_encoded.drop("income", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
torch_X_train = torch.tensor(X_train, dtype=torch.float32)
torch_y_train = torch.tensor(y_train, dtype=torch.float32)
torch_X_test = torch.tensor(X_test, dtype=torch.float32)
torch_y_test = torch.tensor(y_test, dtype=torch.float32)

# DataLoader
train_dataset = TensorDataset(torch_X_train, torch_y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = SocialPolicyPredictor(input_dim, output_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    avg_loss = epoch_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "./neural_model/policy_model.pt")
print("Policy model saved.")
