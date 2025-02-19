from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, aliased
from datetime import datetime
import pandas as pd
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

df = df.drop(columns=['_sa_instance_state'])


sdf = df.groupby('espn_name')


print(sdf['adj_efficiency_margin'].mean().sort_values(ascending=False))

# sdf['adj_efficiency_margin'].mean().sort_values(ascending=False).to_csv('adj_efficiency_margin.csv')

t2 = time.time()
print(f"Time elapsed: {t2-t1:0.2f} seconds")