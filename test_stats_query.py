from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, aliased
from datetime import datetime
import pandas as pd
import numpy as np
import time 

from webscraping.web_scrapper import Scraper
from database.database import *
import matplotlib.pyplot as plt
import seaborn as sns

t1 = time.time()

engine = create_engine('sqlite:///database/ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()


season = 2025

season_id = session.query(Season).filter(Season.year == season).first()

games = session.query(Games).filter(Games.season_id == season_id.id).all()

game_ids = [game.game_id for game in games]

print(f"length of games: {len(games)}")

adjusted_game_ids = session.query(AdjustedMetrics.game_id).all()

agi = [game.game_id for game in adjusted_game_ids]



# season_data = session.query(Games, AdjustedMetrics).join(AdjustedMetrics, Games.game_id == AdjustedMetrics.game_id and Games.team_id == AdjustedMetrics.team_id).filter(Games.season_id == season_id.id).all()

season_data = session.query(Games, AdjustedMetrics, Teams).join(AdjustedMetrics, (Games.game_id == AdjustedMetrics.game_id) & (Games.team_id == AdjustedMetrics.team_id)).join(Teams, Games.team_id==Teams.id).filter(Games.season_id == season_id.id).all()

game_df = pd.DataFrame([game.__dict__ for game in games])

# Count the number of times each game_id occurs in the DataFrame
game_id_counts = game_df['game_id'].value_counts()

dup_games = game_id_counts[game_id_counts > 2]
if len(dup_games) > 0:
    print(f"duplicate games in games table: {dup_games}")
else:
    print("No duplicate games in games table")

# for game in game_ids:
#     if game not in agi:
#         print(f"game_id {game} not in AdjustedMetrics table")



# print(f"Number of games with duplicate entries: {dup_games}")


print(f"Joined data length: {len(season_data)}")

off_by = len(season_data) - len(games)

print(f"off by: {off_by}")

df = pd.DataFrame([{**game.__dict__, **metrics.__dict__, **teams.__dict__} for game, metrics, teams, in season_data])

df = df.drop(columns=['_sa_instance_state','game_location','game_state','over_under','betting_line','date','referee1','referee2','referee3','espn_name','name','location'])


sdf = df.groupby('team_id').mean()

#now map the team id back to the team name for display purposes 
teams = session.query(Teams).all()
team_dict = {team.id: team.espn_name for team in teams}

sdf['team_name'] = sdf.index.map(team_dict)
sdf = sdf.set_index('team_name')

keep_cols = ['adj_offensive_efficiency','adj_defensive_efficiency','adj_efficiency_margin','pace']

sdf = sdf[keep_cols]
print(sdf.sort_values(by='adj_efficiency_margin',ascending=False))
# print(sdf['adj_efficiency_margin'].sort_values(ascending=False))

#steps to do nearest neighbor search: 
#1. identify the game of interest (this will give you team and also the opponent)
#2. get the adjusted metrics for the team and opponent
#3. Calculate season averages for the team and opponent
#4. use this as input to search all of the other games to find similar games and matchups

# game_id = 401721474
game_id = 401731884

game_of_interest = session.query(Games).filter(Games.game_id == game_id).first()

team1_id = game_of_interest.team_id
team2_id = game_of_interest.opponent_id

team1_name = session.query(Teams).filter(Teams.id == team1_id).first().espn_name
team2_name = session.query(Teams).filter(Teams.id == team2_id).first().espn_name

print(f"analyzing game between {team1_name} and {team2_name}")

# #get season averages for the team and opponent
team1_avg = sdf.loc[team1_name]
team2_avg = sdf.loc[team2_name]

print(f"here are the season averages:\n{team1_avg}\n{team2_avg}")

# # print(f"team1 shape: {team1_avg.shape}")
# #now search for similar games
# all_data = session.query(Games, AdjustedMetrics, Teams).join(AdjustedMetrics, (Games.game_id == AdjustedMetrics.game_id) & (Games.team_id == AdjustedMetrics.team_id)).join(Teams, Games.team_id==Teams.id).all()

TeamMetrics = aliased(AdjustedMetrics)
OpponentMetrics = aliased(AdjustedMetrics)
Opponent = aliased(Teams)
all_data = session.query(Games,TeamMetrics.adj_offensive_efficiency.label('team_off_eff'),
    OpponentMetrics.adj_offensive_efficiency.label('opp_off_eff'),
    TeamMetrics.adj_defensive_efficiency.label('team_def_eff'),
    OpponentMetrics.adj_defensive_efficiency.label('opp_def_eff'),
    TeamMetrics.adj_efficiency_margin.label('team_eff_margin'),
    OpponentMetrics.adj_efficiency_margin.label('opp_eff_margin'),
    Teams.espn_name.label('team_name'),
    Opponent.espn_name.label('opponent_name')
).join(
    TeamMetrics, (Games.game_id == TeamMetrics.game_id) & (Games.team_id == TeamMetrics.team_id)
).join(
    OpponentMetrics, (Games.game_id == OpponentMetrics.game_id) & (Games.opponent_id == OpponentMetrics.team_id)
).join(
    Teams, Games.team_id == Teams.id
).join(
    Opponent, Games.opponent_id == Opponent.id
).all()

# combined = pd.DataFrame([{**game.__dict__, **metrics.__dict__, **teams.__dict__} for game, metrics, teams, in all_data])
combined = pd.DataFrame(all_data)
print(combined.columns)
# combined = combined[keep_cols]


search_cols = ['team_off_eff','team_def_eff','team_eff_margin','opp_off_eff','opp_def_eff','opp_eff_margin']
all_df = combined[search_cols]
print(all_df.tail())
# #nearest neighbor search
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# #drop the columns that are not needed for the nearest neighbor search
# combined = combined.drop(columns=['_sa_instance_state','game_location','game_state','date','over_under','betting_line','date','referee1','referee2','referee3','espn_name','name','location','team_id'])

# #scale the data
scaler = StandardScaler()
X = scaler.fit_transform(all_df)
print(f"X shape: {X.shape}")
# #fit the model
nn = NearestNeighbors(n_neighbors=500,algorithm='kd_tree')
nn.fit(X)

#now format the current team1 averages to be used as input to the model
team1_input = pd.DataFrame([{
    'team_off_eff': team1_avg['adj_offensive_efficiency'],
    'team_def_eff': team1_avg['adj_defensive_efficiency'],
    'team_eff_margin': team1_avg['adj_efficiency_margin'],
    'opp_off_eff': team2_avg['adj_offensive_efficiency'],
    'opp_def_eff': team2_avg['adj_defensive_efficiency'],
    'opp_eff_margin': team2_avg['adj_efficiency_margin']
}])

team1_scaled = scaler.transform(team1_input)
distance, indicies = nn.kneighbors(team1_scaled)
# print(f"distance: {distance}")
# print(f"indicies: {indicies}")

rdf = combined.loc[indicies[0],search_cols]
print(f"nearest neighbor comparison:")
print(f"eff margin: {rdf.mean()['team_eff_margin']:0.2f} vs {team1_avg['adj_efficiency_margin']:0.2f}")
print(f"off eff: {rdf.mean()['team_off_eff']:0.2f} vs {team1_avg['adj_offensive_efficiency']:0.2f}")
print(f"def eff: {rdf.mean()['team_def_eff']:0.2f} vs {team1_avg['adj_defensive_efficiency']:0.2f}")

# # Untransform the data
# rdf_unscaled = pd.DataFrame(scaler.inverse_transform(rdf), columns=search_cols)

# print(f"nearest neighbor comparison:")
# print(f"eff margin: {rdf_unscaled.mean()['team_eff_margin']:0.2f} vs {team1_avg['adj_efficiency_margin']:0.2f}")
# print(f"off eff: {rdf_unscaled.mean()['team_off_eff']:0.2f} vs {team1_avg['adj_offensive_efficiency']:0.2f}")
# print(f"def eff: {rdf_unscaled.mean()['team_def_eff']:0.2f} vs {team1_avg['adj_defensive_efficiency']:0.2f}")

# sdf['adj_efficiency_margin'].mean().sort_values(ascending=False).to_csv('adj_efficiency_margin.csv')

t2 = time.time()
print(f"Time elapsed: {t2-t1:0.2f} seconds")