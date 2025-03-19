import datetime
from datetime import date, timedelta
from webscraping.web_scrapper import Scraper
from database.database import *
from single_prediction import predict_game
from webscraping.scrape_gamescores import Matchup 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


engine = create_engine('sqlite:///database/ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()


todays_date = date.today()

future_date = todays_date + timedelta(days=1)

NUM_GAMES = 10000

ws = Scraper()

ws.scrape_future_games(todays_date)

game_dict = {'winning_team':[], 'winning_team_pts':[], 'win_pct':[], 'losing_team':[], 'losing_team_pts':[], 'over_perc':[], 'win_margin':[],'total_pts':[]}

# print(ws.future_df)

for idx, row in ws.future_df.iterrows():
    away_team = row['away_team']
    home_team = row['home_team']
    game_location = row['location']
    game_location = game_location.lstrip()
    game_location+=' '
    gloc = game_location
    vegas_over_under = row['over_under']
    if vegas_over_under =='None':
        vegas_over_under = None
    
    
    team1 = session.query(Teams).filter(Teams.espn_name.ilike(f'%{away_team}%')).first()
    team2 = session.query(Teams).filter(Teams.espn_name.ilike(f'%{home_team}%')).first()
    game_location = session.query(GameLocations).filter(GameLocations.location.ilike(f'%{game_location}%')).first()
    if game_location is None:
        print(f"Game location {row['location'].lstrip()} not found in database")
        game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts = predict_game(team1 = team1.espn_name, team2 = team2.espn_name, game_location = gloc, db_session = session, num_games = NUM_GAMES, season = 2025, over_under = vegas_over_under)
    else:

        print('\n********** Analyzing the following game **********\n')
        print(f"{team1.espn_name} vs {team2.espn_name} at {game_location.location}\n")
    
        game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts = predict_game(team1 = team1.espn_name, team2 = team2.espn_name, game_location=game_location.location, db_session=session, num_games=NUM_GAMES, season=2025, over_under=vegas_over_under)

    if game_winner == team1.espn_name:
        gw = team1.id
        lw = team2.id
    else:
        gw = team2.id
        lw = team1.id
    # this assumes that team1 id is the winner
    model_prediction = Predictions(date = todays_date, team1_id = gw, team2_id = lw, team1_pts = win_pts, team2_pts = loser_pts, win_pct = win_pct, win_margin = win_margin, total_pts = total_pts,winner = gw)
    session.add(model_prediction)
    session.commit()

    game_dict['winning_team'].append(game_winner)
    game_dict['winning_team_pts'].append(win_pts)
    game_dict['win_pct'].append(win_pct)
    game_dict['losing_team'].append(game_loser)
    game_dict['losing_team_pts'].append(loser_pts)
    game_dict['over_perc'].append(over_under)
    game_dict['win_margin'].append(win_margin)
    game_dict['total_pts'].append(total_pts)

game_df = pd.DataFrame(game_dict)

print(game_df.sort_values(by='win_pct', ascending=False))

game_df.to_csv(f'{todays_date}_predictions.csv', index=False)

