from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, aliased
from datetime import datetime
import pandas as pd
import numpy as np
import time 

from webscraping.web_scrapper import Scraper
from database.database import *
from models.NN_model import *
from single_prediction import ModelMatchup, ModelStats, predict_game
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

t1 = time.time()

engine = create_engine('sqlite:///database/ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

season = 2025

season_id = session.query(Season).filter(Season.year == season).first()


team_of_interest = 'Duke Blue Devils'

def generate_training_data(exclude):

    season_id = session.query(Season).filter(Season.year == season).first()
    OpponentAdjMetrics = aliased(AdjustedMetrics)
    OpponentMetrics = aliased(Games)

    # training_data = session.query(Games, Teams, OpponentMetrics, OpponentAdjMetrics).join(Teams, Games.team_id == Teams.id).join(OpponentMetrics, (Games.game_id == OpponentMetrics.game_id) & (Games.opponent_id == OpponentMetrics.team_id)).join(OpponentAdjMetrics, (OpponentMetrics.game_id == OpponentAdjMetrics.game_id) & (OpponentMetrics.opponent_id == OpponentAdjMetrics.team_id)).filter(Games.season_id == season_id.id, Games.season_id != exclude).all()

    training_data = session.query(Games, Teams, OpponentAdjMetrics).join(Teams, Games.team_id == Teams.id).join(OpponentAdjMetrics, (Games.game_id == OpponentAdjMetrics.game_id) & (Games.opponent_id == OpponentAdjMetrics.team_id)).filter(Games.season_id < season_id.id).all()

    td = pd.DataFrame([{**game.__dict__, **teams.__dict__, **opponent.__dict__} for game, teams, opponent, in training_data])

    return td

def season_trends(team_name):

    team_id = session.query(Teams).filter(Teams.espn_name == team_name).first().id

    OpponentAdjMetrics = aliased(AdjustedMetrics)
    

    season_games = session.query(Games, Teams, OpponentAdjMetrics).join(Teams, Games.team_id == Teams.id).join(OpponentAdjMetrics, (Games.game_id == OpponentAdjMetrics.game_id) & (Games.opponent_id == OpponentAdjMetrics.team_id)).filter(Games.season_id == season_id.id, Games.team_id == team_id).all()

    # print(len(season_games))
    
    #unpack each tuple to dataframe
    season_df = pd.DataFrame([{**game.__dict__, **teams.__dict__, **opponent.__dict__} for game, teams, opponent, in season_games])


    return season_df 

# training_df = generate_training_data(exclude=2025)

# NUM_EPOCHS = 1000
# LEARNING_RATE = 0.5

# points_model = NN_Model()
# points_model.load_model(session)

# #these are opponent stats
# input_params = ['adj_offensive_efficiency','adj_defensive_efficiency','adj_efficiency_margin','adj_efg_percentage','adj_turnover_percentage','two_point_field_goals_attempted','three_point_field_goals_attempted','free_throws_attempted','possessions','pace']
# output_params = ['three_point_field_goal_percentage','two_point_field_goal_percentage','free_throw_percentage','assists']
# # output_params = points_model.params

# X = training_df[input_params].values
# y = training_df[output_params].values

# nn_model = NN_Model()

# nn_model.build_stats_model(input_features=input_params,output_features=output_params)

# nn_model.train_model(num_epochs = NUM_EPOCHS, learning_rate = LEARNING_RATE, criterion = nn.MSELoss(),
#                     X = X, y = y, test_size = 0.2, random_state = 42, normalize_output = True)


season_df = season_trends(team_of_interest)

#find nearest neighbors for opponent efficiency margin for a given game
search_cols = ['adj_offensive_efficiency','adj_defensive_efficiency','adj_efficiency_margin']
X = season_df[search_cols].values
neigh = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
neigh.fit(X)
game_input = season_df.iloc[10][search_cols].values.reshape(1,-1)
distances, indices = neigh.kneighbors(game_input)

fig, ax = plt.subplots()
sns.histplot(data=season_df, x='three_point_field_goal_percentage', bins=20, kde=True, ax=ax)
ax.axvline(x = season_df.iloc[10]['three_point_field_goal_percentage'], color='red', linestyle='--')
for idx, i in enumerate(indices[0]):
    if idx == 0 :
        continue
    ax.axvline(x = season_df.iloc[i]['three_point_field_goal_percentage'], color='blue', linestyle='--')

plt.show()
#test the model prediction 
# x_data = season_df[input_params].values
# x_scaled = nn_model.scaler.transform(x_data)
# x_scaled = torch.from_numpy(x_scaled.astype(np.float32))

# predicted = nn_model.predict(x_scaled)
# #inverse transform for the outcome because we normalized the output
# predicted = nn_model.output_scalar.inverse_transform(predicted.detach().numpy())
# for idx, param in enumerate(output_params):
#     print(f"{param}: {predicted[0][idx]:0.2f} vs actual: {season_df.iloc[0][param]}")

# diff = predicted - season_df[output_params].values
# print(f"difference: {diff}")
# print(f"predicted: {predicted[0]} vs actual: {season_df.iloc[0][output_params].values}")

# #predict points now 
# x_points = predicted[0].reshape(1,-1)
# x_points_scaled = points_model.scaler.transform(x_points)
# x_points_scaled = torch.from_numpy(x_points_scaled.astype(np.float32))
# point_pred = points_model.predict(x_points_scaled)
# point_pred = point_pred.detach().numpy()
# print(f"predicted points: {point_pred} vs actual points: {season_df.iloc[0]['points']}")
# fig, ax = plt.subplots(2,2, figsize=(10,10))

# ax[0,0].scatter(season_df['three_point_field_goal_percentage'], season_df['adj_defensive_efficiency'])
# ax[0,0].set_xlabel('3 Point Field Goal Percentage')
# ax[0,1].scatter(season_df['two_point_field_goal_percentage'], season_df['adj_defensive_efficiency'])
# ax[0,1].set_xlabel('2 Point Field Goal Percentage')
# ax[1,0].scatter(season_df['free_throw_percentage'], season_df['adj_defensive_efficiency'])
# ax[1,0].set_xlabel('Free Throw Percentage')
# ax[1,1].scatter(season_df['possessions'], season_df['adj_defensive_efficiency'])
# ax[1,1].set_xlabel('Possessions')

t2 = time.time()

print(f"time elapsed: {t2-t1:0.2f} seconds") 
plt.show()