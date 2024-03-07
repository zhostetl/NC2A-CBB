import os
import glob
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn

df = pd.read_excel(r'C:\Users\zhostetl\Documents\11_CBB\99_git\NC2A-CBB\03_modelfitting\adjusted_data.xlsx',index_col=0)

# print(df.head())

# Check if there are any NaN values in the 
nan_indices = df[df['Distance_Traveled'].isna()].index
# print(df.loc[nan_indices, ['Team','Opponent','Distance_Traveled']])
df = df.drop(nan_indices)

params = ['Distance_Traveled',
          'adj_Raw_Off_Eff','adj_off_eFG','adj_off_TOV','adj_off_ORB','adj_off_FTR',
          'adj_Raw_Def_Eff','adj_def_eFG','adj_def_TOV','adj_def_ORB','adj_def_FTR']
y = ['Team_score']
# Prepare the data
X = df[params].values
y = df[y].values


# Normalize the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Convert the data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(params), 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the neural network
net = Net()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Train the neural network
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Predict the score for a new input
new_input = torch.tensor([[100, 1.10, 0.90, 0.35, 0.1, 0.4, 0.4, 0.3, 0.5, 0.13, 0.7]], dtype=torch.float32)
predicted_score = net(new_input)
print(f'Predicted Score: {predicted_score.item():.2f}')