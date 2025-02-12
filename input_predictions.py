from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
import pandas as pd
import numpy as np 
import os
import glob
import time 
from datetime import date, timedelta


from webscraping.web_scrapper import Scraper
from webscraping.scrape_gamescores import Matchup
from database.database import *

engine = create_engine('sqlite:///database/ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

def query_teamname(team_name, session):
    team = session.query(Teams).filter(Teams.espn_name.ilike(f'%{team_name}%')).first()
    return team


prediction_path = r'C:\Users\zhostetl\Documents\11_CBB\99_git\NC2A-CBB\predictions'

prediction_files = glob.glob(prediction_path + r'\*.csv')

for file in prediction_files:
    prediction_date = os.path.basename(file).split('_')[0]
    if prediction_date!= '2025-02-12':
        continue
    p_date = datetime.strptime(prediction_date, '%Y-%m-%d')

    predictions = pd.read_csv(file)
    print(f"predictions for {prediction_date}")
    for idx, row in predictions.iterrows(): 
        t1 = row['winning_team']
        t2 = row['losing_team']
        team1 = query_teamname(t1, session)
        team2 = query_teamname(t2, session)
        winner = team1.id
        # print(team1.espn_name)
        # print(f"winner: {winner}")
        model_prediction = Predictions(date = p_date, team1_id = team1.id, team2_id = team2.id, team1_pts = row['winning_team_pts'], team2_pts = row['losing_team_pts'], win_pct = row['win_pct'], win_margin = row['win_margin'], total_pts = row['total_pts'],winner = team1.id)
        session.add(model_prediction)
    session.commit()