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
import pickle


from database.database import *
from models.NN_model import *
import time

# get the data extracted from the database... will need the Games table and the AdjustedMetrics table
t1 = time.time()

engine = create_engine('sqlite:///database/ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

compiled_df = pd.DataFrame()

seasons = [2015, 2016, 2017]
NUM_EPOCHS = 10000
LEARNING_RATE = 0.05
# seasons = [2015, 2016]
for season in seasons: 


    # season_year = session.query(Season).filter(Season.year == season).first()
    season_id = session.query(Season).filter(Season.year == season).first()
    # Join the Games table with the AdjustedMetrics table and filter by season_id
    # season_data = session.query(Games, AdjustedMetrics).join(AdjustedMetrics, Games.game_id == AdjustedMetrics.game_id).all()
    season_data = session.query(Games, AdjustedMetrics).join(AdjustedMetrics, (Games.game_id == AdjustedMetrics.game_id) & (Games.team_id == AdjustedMetrics.team_id)).filter(Games.season_id == season_id.id).all()
    # print(season_data)
    # Convert the query results to a DataFrame

    df = pd.DataFrame([{**game.__dict__, **metrics.__dict__} for game, metrics in season_data])
    df = df.drop(columns=['_sa_instance_state'])
    
    #check for NaN values
    for col in df.columns: 
        if df[col].isna().sum() > 0:
            print(f"Column {col} has {df[col].isna().sum()} NaN values")

    compiled_df = pd.concat([compiled_df, df])



#these are the parameters that we want to train the model with 
params = ['distance_traveled',
          'adj_offensive_efficiency', 'adj_defensive_efficiency', 'adj_efficiency_margin',
          'adj_efg_percentage','adj_turnover_percentage','adj_offensive_rebound_percentage','adj_free_throw_rate',
          'opp_adj_efg_percentage','opp_adj_turnover_percentage','adj_def_rebound_percentage','opp_adj_free_throw_rate',
          'pace','total_turnovers','fouls', 'steals','blocks','rebounds','assists',
          'two_point_field_goal_percentage','three_point_field_goal_percentage', 'free_throw_percentage',
          'possessions','home','away']

y = ['points']
# Prepare the data
X = compiled_df[params].values
y = compiled_df[y].values


# call the NN model from the modeling module 

nn_model = NN_Model()

nn_model.build_model(input_features=params)

nn_model.train_model(num_epochs = NUM_EPOCHS, learning_rate = LEARNING_RATE, criterion = nn.MSELoss(),
                    X = X, y = y, test_size = 0.2, random_state = 42)


nn_model.save_model(session, name = f'NN_model_trained with {seasons[0]}-{seasons[-1]}', description = f'This model was trained with data from the {seasons[0]}-{seasons[-1]} seasons for {NUM_EPOCHS} epochs with a learning rate of {LEARNING_RATE}')


