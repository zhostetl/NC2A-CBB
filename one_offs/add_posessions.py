from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, aliased
from datetime import datetime
import pandas as pd
import time 


# Use relative import to import from the database module
from database.database import *
import matplotlib.pyplot as plt
import seaborn as sns
import os

t1 = time.time()

# Construct the relative path to the database file
db_folder = os.path.join(os.path.dirname(__file__), '..', 'database')
db_name = 'ncaa_basketball.db'
db_path = os.path.join(db_folder, db_name)

# Create the SQLAlchemy engine using the constructed database path
engine = create_engine(f'sqlite:///{db_path}')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

season_data = session.query(Games).all()

print(f"len(season_data): {len(season_data)}")

missing = []

for idx, game in enumerate(season_data):
    # if idx > 5: 
    #     break
    game_id = game.game_id
    team_id = game.team_id
    possessions = 0.96 * (game.field_goals_attemped + game.total_turnovers + 0.44 * game.free_throws_attempted - game.offensive_rebounds)
    # print(f"game_id: {game_id}, team_id: {team_id}, possessions: {possessions:0.2f}")
    game.possessions = round(possessions,5)
    # game.opp_eFG = round(opp_eFG,5)
    # game.opp_TO_rate = round(opp_TO_per,5)
    # game.DREB_per = round(dreb_per,5)
    # game.opp_FT_rate = round(opp_FT_rate,5)
    # game.TO_rate = round(TO_rate,5)
    # session.commit()
    
    # print(f"updated game_id: {game_id}")
# session.commit()
print("***--------***\nFinished updating possessions\n***--------***")