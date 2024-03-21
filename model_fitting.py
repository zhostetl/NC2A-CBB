import os
import glob
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


files = [r'C:\Users\zhostetl\Documents\11_CBB\99_git\NC2A-CBB\03_modelfitting\2022_2023_adjusted_data_EM.xlsx',
         r'C:\Users\zhostetl\Documents\11_CBB\99_git\NC2A-CBB\03_modelfitting\2023_2024_adjusted_data_EM.xlsx']

df = pd.concat([pd.read_excel(f,index_col=0) for f in files])

# df = pd.read_excel(r'C:\Users\zhostetl\Documents\11_CBB\99_git\NC2A-CBB\03_modelfitting\2022_2023_adjusted_data.xlsx',index_col=0)
# new_season = pd.read_excel(r'C:\Users\zhostetl\Documents\11_CBB\99_git\NC2A-CBB\03_modelfitting\2023_2024_adjusted_data.xlsx',index_col=0)
# print(df.head())

# Check if there are any NaN values in the 
nan_indices = df[df['Distance_Traveled'].isna()].index

# print(df.loc[nan_indices, ['Team','Opponent','Distance_Traveled']])
df = df.drop(nan_indices)

params = ['Distance_Traveled',
          'adj_Raw_Off_Eff','adj_off_eFG','adj_off_TOV','adj_off_ORB','adj_off_FTR',
          'adj_Raw_Def_Eff','adj_def_eFG','adj_def_TOV','adj_def_ORB','adj_def_FTR',
          'Pace','Total Turnovers','Fouls','FG_attempted','3PT_attempted',
          'Team_Possessions','Home','Away','adj_EM']
# params = ['Distance_Traveled','adj_Raw_Off_Eff','adj_Raw_Def_Eff']
y = ['Team_score']
# Prepare the data
X = df[params].values
y = df[y].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, r'03_modelfitting\scaler_EM.pkl')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
# add_layers = 40
add_layers = 60
#build the model
ANNreg = nn.Sequential(
    nn.Linear(X.shape[1], add_layers),
    # nn.ReLU(),
    nn.Sigmoid(),
    nn.Linear(add_layers, add_layers),
    # nn.ReLU(),
    nn.Sigmoid(),
    nn.Linear(add_layers, add_layers),
    nn.Sigmoid(),
    nn.Linear(add_layers, 1)
)
# print(ANNreg)
learning_rate = 0.005

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(ANNreg.parameters(), lr=learning_rate)

# Train the model
num_epochs = 7000
# num_epochs = 50000
losses = torch.zeros(num_epochs)

# y = scaler.fit_transform(y)
# Convert X and y to Tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

for epoch in range(num_epochs):

    # Forward pass
    y_hat= ANNreg(X_train)
    loss = criterion(y_hat, y_train)
    losses[epoch] = loss
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print (f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

#final forward pass
predictions = ANNreg(X_test)
# print(predictions.detach().numpy())
# print(X_test)
#final loss 
# testloss = (predictions - y_test).pow(2).mean()
testloss = criterion(predictions, y_test)
print(f"final loss: {testloss:0.2f}")

torch.save(ANNreg.state_dict(), r'03_modelfitting\ANNreg_EM_train.pth')
#rescale the predictions
# predictions = scaler.inverse_transform(predictions.detach().numpy())
# y = scaler.inverse_transform(y.detach().numpy())

# plt.plot(losses.detach().numpy())
# plt.plot(num_epochs, testloss.detach(), 'ro')
# plt.title(f"Final Loss: {testloss:0.2f}")

# plt.figure()
# plt.plot(predictions.detach().numpy(), y_test, 'ro')
# plt.plot([0, 100], [0, 100], 'k-')
# plt.title('Predictions vs. Actual')

# plt.show()


