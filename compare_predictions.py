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

nn_model = NN_Model()

nn_model.load_model(session)

season = 2025

season_id = session.query(Season).filter(Season.year == season).first()

# print(nn_model.params)

def all_game_data():
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
    search_cols = ['team_off_eff','team_def_eff','team_eff_margin','opp_off_eff','opp_def_eff','opp_eff_margin']
    all_df = combined[search_cols]
    return combined, all_df

def nearest_game_search(all_games = None, season_df = None, team_of_interest = None, opponent = None, n_neighbors=500):
    scaler = StandardScaler()
    X = scaler.fit_transform(all_games)
    # #fit the model
    nn = NearestNeighbors(n_neighbors=n_neighbors,algorithm='kd_tree')
    nn.fit(X)

    team1_avg = season_df.loc[team_of_interest]
    team2_avg = season_df.loc[opponent]

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

    return distance, indicies

def get_season_stats():
    games = session.query(Games).filter(Games.season_id == season_id.id).all()
    season_data = session.query(Games, AdjustedMetrics, Teams).join(AdjustedMetrics, (Games.game_id == AdjustedMetrics.game_id) & (Games.team_id == AdjustedMetrics.team_id)).join(Teams, Games.team_id==Teams.id).filter(Games.season_id == season_id.id).all()

    game_df = pd.DataFrame([game.__dict__ for game in games])

    # Count the number of times each game_id occurs in the DataFrame
    game_id_counts = game_df['game_id'].value_counts()

    dup_games = game_id_counts[game_id_counts > 2]
    # if len(dup_games) > 0:
    #     print(f"duplicate games in games table: {dup_games}")
    # else:
    #     print("No duplicate games in games table")

    off_by = len(season_data) - len(games)

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
    return sdf
    # print(sdf.sort_values(by='adj_efficiency_margin',ascending=False))

#first step is to get the game id and make the predictions. 
#want to see how close the season varied stats come to the nearest neighbor searches 

game_id_db = session.query(Games.game_id).filter(Games.season_id == season_id.id).filter(Games.date > '2025-02-01').all()

# game_ids = np.unique([game.game_id for game in game_id_db])
# print(game_ids)

NUM_GAMES = 10000

# game_id = 401708421 #auburn kentucky
game_id = 401724905 #duke fsu 
# game_id = 401708407 #florida LSU

# for game_id in game_ids:
# game_id = int(game_id)
game = session.query(Games).filter(Games.game_id == game_id).first()


team1 = session.query(Teams).filter(Teams.id == game.team_id).first()
team2 = session.query(Teams).filter(Teams.id == game.opponent_id).first()

game_location = game.game_state

team1_stats = ModelStats(team_name = team1.espn_name, home_team = False, season = 2025, db_session = session)
team2_stats = ModelStats(team_name = team2.espn_name, home_team = True, season = 2025, db_session = session)

results = pd.DataFrame(columns = [team1_stats.team_name, team2_stats.team_name], index = ['season_stats','nearest_neighbor','actual'])

methods = ['season_stats','nearest_neighbor']
print(f"Matchup between {team1.espn_name} and {team2.espn_name} at {game_location}")
for method in methods:
    print(f"Method: {method}")
    # if method == 'nearest_neighbor':
    #     continue

    matchup = ModelMatchup(team1=team1_stats, team2=team2_stats, num_games=NUM_GAMES, season=2025, game_location=game_location, db_session=session, ml_model=nn_model, prediction_method=method)
    matchup.get_season_data()
    matchup.calculate_metrics(team1_stats, team2_stats)
    matchup.check_distances(team1_stats, team2_stats)
    matchup.adjust_metrics(team_to_adjust=team1_stats, opponent=team2_stats)
    matchup.adjust_metrics(team_to_adjust=team2_stats, opponent=team1_stats)
    # plt.hist(team1_stats.varied_df['three_point_field_goal_percentage'], bins=20, alpha=0.5, label=f'{method}')

    matchup.simulate_game(team1_stats, team2_stats, over_under=None)
    # print(f"matchup winner: {matchup.winner} with {matchup.winner_pts:0.2f} points")
    # print(f"matchup loser: {matchup.loser} with {matchup.loser_pts:0.2f} points")
    winner_id = session.query(Teams).filter(Teams.espn_name == matchup.winner).first().id
    loser_id = session.query(Teams).filter(Teams.espn_name == matchup.loser).first().id
    actual_outcome = session.query(Games).filter(Games.game_id == game_id).filter(Games.team_id == winner_id).first()
    actual_loser_outcome = session.query(Games).filter(Games.game_id == game_id).filter(Games.team_id == loser_id).first()

    # print(f"predicted points using {method} for {matchup.winner} {matchup.winner_pts :0.2f} vs actual outcome {actual_outcome.points:0.2f}")
    # print(f"predicted points using {method} for {matchup.loser} {matchup.loser_pts :0.2f} vs actual outcome {actual_loser_outcome.points:0.2f}")
    results.loc[method,matchup.winner] = matchup.winner_pts
    results.loc[method,matchup.loser] = matchup.loser_pts
    results.loc['actual',matchup.winner] = actual_outcome.points
    results.loc['actual',matchup.loser] = actual_loser_outcome.points

print(results)
# plt.legend()
# plt.show()
    # plt.hist(team1_stats.varied_df['pace'], bins=20, alpha=0.5, label='Model', color='blue')
#     sns.histplot(team1_stats.varied_df['three_point_field_goal_percentage'], kde = True, stat = 'density',bins=20, alpha=0.5, label='Model', color='blue')
#     plt.axvline(matchup.season_stats.loc[team1.espn_name]['three_point_field_goal_percentage'], color='black', linestyle = '--', label='Season Stats')
#     # sns.kdeplot(team1_stats.varied_df['pace'], label='Model', color='blue')
#     # sns.ecdfplot(team1_stats.varied_df['pace'], label='Model', color='blue')
#     #get the stat from the game to see where it lies on the distribution 
#     game_stats = session.query(Games).filter(Games.game_id == game_id).filter(Games.team_id == team1.id).first()
#     plt.axvline(game_stats.three_point_field_goal_percentage, color='red', label='Actual Game')
#     plt.legend()
# plt.show()
# all_games, search_data = all_game_data()

# distance, indicies = nearest_game_search(search_data, season_stats, team1.espn_name, team2.espn_name, n_neighbors=1000)

# ag = session.query(Games).all()

# df = pd.DataFrame([{**game.__dict__} for game in ag])

# common_stats = df.iloc[indicies[0]]
# common_stats = common_stats[matchup.team1.boxscore_params]

# means = common_stats.mean()
# stds = common_stats.std()

# common_samples = np.empty((NUM_GAMES, len(means)))
# for i in range(NUM_GAMES):
#     common_samples[i] = np.random.normal(means, stds)

# plt.hist(common_stats['two_point_field_goal_percentage'], bins=20, alpha=0.5, label='Nearest Neighbors', color='blue')
# plt.hist(common_samples[:,6], bins=20, alpha=0.5, label='Random Samples', color='green')
# plt.hist(matchup.team1.varied_df['two_point_field_goal_percentage'], bins=20, alpha=0.5, label='Team season stats', color='red')
# plt.legend()

# print(matchup.team1.varied_df.mean()['offensive_efficiency'])
# print(all_games.iloc[indicies[0]]['team_off_eff'].mean())

# fig, ax = plt.subplots(1,2,figsize=(10,5))
# sns.histplot(matchup.team1.varied_df['offensive_efficiency'], bins=20, alpha=0.5, label='Model', ax=ax[0])
# sns.histplot(all_games.iloc[indicies[0]]['team_off_eff'], bins=20, alpha=0.5, label='Nearest Neighbors', ax=ax[0])
# sns.histplot(matchup.team1.varied_df['defensive_efficiency'], bins=20, alpha=0.5, label='Model', ax=ax[1])
# sns.histplot(all_games.iloc[indicies[0]]['team_def_eff'], bins=20, alpha=0.5, label='Nearest Neighbors', ax=ax[1])

# plt.legend()
plt.show()