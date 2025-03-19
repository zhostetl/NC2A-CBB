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
import matplotlib.pyplot as plt
import seaborn as sns

from webscraping.web_scrapper import Scraper
from webscraping.scrape_gamescores import Matchup
from database.database import *

engine = create_engine('sqlite:///database/ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()


def get_model_performance():
    model_performance = session.query(ModelPerformance).all()
    df = pd.DataFrame([(p.date, p.correct, p.total, p.accuracy, p.error) for p in model_performance], columns = ['date', 'correct', 'total', 'accuracy', 'error']).sort_values(by='date')
    return df

df = get_model_performance()

# total_accuracy = df['correct'].sum()/df['total'].sum()*100

# print(f"Total accuracy: {total_accuracy:0.2f}%")

# doi = date(2025, 1, 25)

# doi = date.today()

# dates = ['yesterday','total']
# summary_df = pd.DataFrame(columns = ['correct','total','accuracy','avg_margin_error'], index = dates)
margin_error = {}

model_predictions = session.query(Predictions).all()
#get unique dates from model_predictions
dates = list(set([p.date for p in model_predictions]))


for d in dates:
    
    if d in df['date'].values:
        # print(f"already have predictions for {d}")
        continue
    print(f"gettting predictions for {d}")
    # if d == date.today():
#         continue
    model_predictions = session.query(Predictions).filter(Predictions.date == d).all()


# #     if d == 'yesterday':
# #         continue
# #         # doi = date.today() - timedelta(days=1)
# #         # model_predictions = session.query(Predictions).filter(Predictions.date == doi).all()
# #     else:
# #         doi = date.today()
# #         model_predictions = session.query(Predictions).filter(Predictions.date < doi).all()


    # print(f"number of predictions for {doi}: {len(model_predictions)}")

#     #get the games table data to compare the actual outcome: 
    season_id = session.query(Season).filter(Season.year == d.year).first()

    win_count = 0 
    margin_error[d] = np.array([])

    total_games = len(model_predictions)

    for game in model_predictions: 
        
        team_name = session.query(Teams).filter(Teams.id == game.team1_id).first()
        opponent_name = session.query(Teams).filter(Teams.id == game.team2_id).first()

        game_outcome = session.query(Games).filter(Games.season_id == season_id.id).filter(Games.team_id == game.team1_id).filter(Games.opponent_id == game.team2_id).filter(Games.date == game.date).first()

        opponent_points = session.query(Games).filter(Games.season_id == season_id.id).filter(Games.team_id == game.team2_id).filter(Games.opponent_id == game.team1_id).filter(Games.date == game.date).first()
        
        if not opponent_points:
            print(f"game on {game.date} for team id {game.team2_id} not found in database")
            total_games-=1
            continue
        win_team_name = session.query(Teams).filter(Teams.id == game.team1_id).first()

        if game_outcome is None:
            print(f"game not found in database")
        else:
            if game_outcome.win == 1:
                
                # print(f"{win_team_name.espn_name} won {game_outcome.points} to {opponent_points}")
                win_count+=1
                real_win_margin = game_outcome.points - opponent_points.points
                predicted_win_margin = game.win_margin
                margin_error[d] = np.append(margin_error[d], real_win_margin - predicted_win_margin)
            else:
                # print(f"**{win_team_name.espn_name} lost {game_outcome.points} to {opponent_points}**")
                real_win_margin = opponent_points.points - game_outcome.points
                predicted_win_margin = game.win_margin
                margin_error[d] = np.append(margin_error[d], real_win_margin - predicted_win_margin)
    
    # summary_df.loc[d] = [win_count, total_games, win_count/total_games*100, np.mean(margin_error[d])]

    model_performance = ModelPerformance(date = d, correct = win_count, total = total_games, accuracy = win_count/total_games*100, error = np.mean(margin_error[d]))
    print(f"predicted {win_count} games correctly out of {total_games}")
    print(f"Accuracy: {(win_count/total_games*100):0.2f}%")
    session.add(model_performance)

session.commit()

model_performance = get_model_performance()

# yesterday = date.today() - timedelta(days=1)
# sdf = model_performance[model_performance['date'] == yesterday]
# print(f"Yesterday's performance: {sdf['correct'].values[0]} out of {sdf['total'].values[0]}")
# yesterday_accuracy = sdf['accuracy'].values[0]
# print(f"Yesterday's accuracy: {yesterday_accuracy:0.2f}%")
total_accuracy = model_performance['correct'].sum()/model_performance['total'].sum()*100
print(f"{model_performance['correct'].sum()} out of {model_performance['total'].sum()}\nTotal accuracy: {total_accuracy:0.2f}%")

#set the color of the bar depending on the accuracy
sns.barplot(x='date', y='accuracy', data=model_performance, palette=['red' if x < total_accuracy else 'green' for x in model_performance['accuracy']])
# sns.barplot(x='date', y='accuracy', data=model_performance)
plt.title('Accuracy of model predictions')
plt.xticks(rotation=45)
plt.show()


