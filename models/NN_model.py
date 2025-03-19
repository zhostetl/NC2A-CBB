import os
import numpy as np
import torch 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pickle
import datetime
import json

from database.database import *

class NN_Model(): 

    def __init__(self):
        self.model = None

    def build_stats_model(self,input_features=None,output_features=None):
        self.params = input_features
        add_layers = 60
        # add_layers = 60
        #build the model
        self.model = nn.Sequential(
        nn.Linear(len(input_features), add_layers),
        # nn.ReLU(),
        nn.Sigmoid(),
        nn.Linear(add_layers, 100),
        # nn.ReLU(),
        nn.Sigmoid(),
        nn.Linear(100, add_layers),
        nn.Sigmoid(),
        nn.Linear(add_layers, len(output_features))
        )

    
    def build_model(self, input_features = None): 
        self.params = input_features
        add_layers = 40
        # add_layers = 60
        #build the model
        self.model = nn.Sequential(
        nn.Linear(len(input_features), add_layers),
        # nn.ReLU(),
        nn.Sigmoid(),
        nn.Linear(add_layers, 100),
        # nn.ReLU(),
        nn.Sigmoid(),
        nn.Linear(100, add_layers),
        nn.Sigmoid(),
        nn.Linear(add_layers, 1)
        )

    def train_model(self, num_epochs = 500, learning_rate = 0.05, criterion = nn.MSELoss(),
                    X = None, y = None, test_size = 0.2, random_state = 42, normalize_output = False): 
        # Loss and optimizer
        self.criterion = criterion
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.train_date = datetime.datetime.now()
        # Train the model
        losses = torch.zeros(num_epochs)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        if normalize_output:
            self.output_scalar = StandardScaler()
            y = self.output_scalar.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        print(f"Size of training data: {X.shape}\nX_train shape: {X_train.shape}\nX_test shape: {X_test.shape}\ny_train shape: {y_train.shape}\ny_test shape: {y_test.shape}")

        # Convert X and y to Tensors
        X_train = torch.from_numpy(X_train.astype(np.float32))
        X_test = torch.from_numpy(X_test.astype(np.float32))
        y_train = torch.from_numpy(y_train.astype(np.float32))
        y_test = torch.from_numpy(y_test.astype(np.float32))

        for epoch in range(num_epochs):

            # Forward pass
            y_hat= self.model(X_train)
            loss = self.criterion(y_hat, y_train)
            losses[epoch] = loss
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                print (f"Epoch [{epoch}/{num_epochs}] Loss: {loss.item():.2f} (pts) ")
    
        # final forward pass
        predictions = self.predict(X_test)
        testloss = self.criterion(predictions, y_test)
        print(f"final loss: {testloss:0.2f} pts")
    
    def save_model(self, session, name = 'ANNreg_EM_train', description = 'ANNreg model trained on the NCAA Basketball data'):

        pp_scaler = pickle.dumps(self.scaler)
       
        state_dict = self.model.state_dict()
        state_dict_blob = pickle.dumps(state_dict)

        json_params = json.dumps(self.params)

        new_model = NNModel(name=name, description=description, scalar_data=pp_scaler, model_data=state_dict_blob,
                             params=json_params, train_date=self.train_date)

        session.add(new_model)
        session.commit()
    
    def predict(self, X):
     
        return self.model(X)
    
    def load_model(self, session): 
        #get the latest model
        retrieved_model = session.query(NNModel).order_by(NNModel.train_date.desc()).first()
        # retrieved_model = session.query(NNModel).filter_by(name='ANNreg_EM_train').first()
        self.train_date = retrieved_model.train_date
        # Deserialize the state dictionary
        loaded_state_dict = pickle.loads(retrieved_model.model_data)

        # print(loaded_state_dict)

        num_params = json.loads(retrieved_model.params)
        self.params = num_params
        self.build_model(input_features=num_params)
        
        # Load the state dictionary into the PyTorch model
        self.model.load_state_dict(loaded_state_dict)

        # Load the scaler
        self.scaler = pickle.loads(retrieved_model.scalar_data)