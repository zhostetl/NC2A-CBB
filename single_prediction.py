import os
import glob
import os
import glob
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import copy
import matplotlib.pyplot as plt
import pickle

import torch 
import torch.nn as nn

from geopy.geocoders import Nominatim
from geopy.distance import geodesic

from database.database import *
from models.NN_model import *


class ModelStats():

    def __init__(self, team_name=None, home_team = False, season = None, db_session = None):
        
        self.team_name = team_name
        self.home = home_team
        self.season = season
        self.db_session = db_session
        self.team_id = self.db_session.query(Teams).filter(Teams.espn_name == self.team_name).first().id
        self.season_year = self.db_session.query(Season).filter(Season.year == self.season).first()
        self.season_id = self.season_year.id
        self.team_conf_id = self.db_session.query(TeamSeasonConference).filter(TeamSeasonConference.team_id == self.team_id).filter(TeamSeasonConference.season_id == self.season_year.id).first().conference_id
        self.conference = self.db_session.query(Conferences).filter(Conferences.id == self.team_conf_id).first().name
        self.team_stats = self.db_session.query(Games).filter(Games.season_id == self.season_year.id).filter(Games.team_id == self.team_id).all()

        self.df = pd.DataFrame([{**game.__dict__} for game in self.team_stats])
        self.boxscore_params = ['total_turnovers', 'fouls', 'steals', 'blocks', 'rebounds', 'assists', 'two_point_field_goal_percentage',   'three_point_field_goal_percentage', 'free_throw_percentage','two_point_field_goals_made', 'three_point_field_goals_made',
        'two_point_field_goals_attempted','three_point_field_goals_attempted','field_goals_made', 'field_goals_attemped', 'free_throws_made',  'free_throws_attempted', 'offensive_rebounds', 'defensive_rebounds','points']
        # self.df = self.df.drop(columns=['_sa_instance_state'])

        self.sdf = self.df[self.boxscore_params]
        self.means = self.sdf.mean()
        self.stds = self.sdf.std()


class ModelMatchup():

    def __init__(self, team1 = None, team2 = None, num_games = 1000, season = None, game_location = None, db_session = None, ml_model = None):

        self.team1 = team1
        self.team2 = team2
        self.num_games = num_games
        self.season = season
        self.game_location = game_location
        self.db_session = db_session
        self.season_id = self.db_session.query(Season).filter(Season.year == self.season).first().id
        self.ml_model = ml_model

        #set up the array to randomly sample the baseline stats and then calculate additional metrics
        # self.team1.samples = np.empty((self.num_games, len(self.team1.means)))
        # self.team2.samples = np.empty((self.num_games, len(self.team2.means)))
        self.generate_samples(self.team1)
        self.generate_samples(self.team2)

    
    def generate_samples(self, team):
        team.samples = np.empty((self.num_games, len(team.means)))
        for i in range(self.num_games):
            team.samples[i] = np.random.normal(team.means, team.stds)
        team.varied_df = pd.DataFrame(team.samples, columns=team.boxscore_params)
        
    def calculate_metrics(self, team1, team2):
        
        team1.varied_df['possessions'] = 0.96 * (team1.varied_df['field_goals_attemped'] + team1.varied_df['total_turnovers'] + 0.44 * team1.varied_df['free_throws_attempted']- team1.varied_df['offensive_rebounds'])
        
        team2.varied_df['possessions'] = 0.96 * (team2.varied_df['field_goals_attemped'] + team2.varied_df['total_turnovers'] + 0.44 * team2.varied_df['free_throws_attempted']- team2.varied_df['offensive_rebounds'])

        team1.varied_df['pace'] = (40 * (team1.varied_df['possessions'] + team2.varied_df['possessions']) / 80)
        team2.varied_df['pace'] = (40 * (team1.varied_df['possessions'] + team2.varied_df['possessions']) / 80)

        ### calculate the offensive and defensive efficiency ###
        
        team1.varied_df['offensive_efficiency'] = team1.varied_df['points'] / team1.varied_df['possessions']
        team2.varied_df['offensive_efficiency'] = team2.varied_df['points'] / team2.varied_df['possessions']

        team1.varied_df['defensive_efficiency'] = team2.varied_df['points'] / team1.varied_df['possessions']
        team2.varied_df['defensive_efficiency'] = team1.varied_df['points'] / team2.varied_df['possessions']

        team1.varied_df['eff_margin'] = team1.varied_df['offensive_efficiency'] - team2.varied_df['defensive_efficiency']
        team2.varied_df['eff_margin'] = team2.varied_df['offensive_efficiency'] - team1.varied_df['defensive_efficiency']
        
        ### calculate the four factors ###

        team1.varied_df['eFG'] = (team1.varied_df['field_goals_made'] + 0.5 * team1.varied_df['three_point_field_goals_made']) / team1.varied_df['field_goals_attemped']
        team2.varied_df['eFG'] = (team2.varied_df['field_goals_made'] + 0.5 * team2.varied_df['three_point_field_goals_made']) / team2.varied_df['field_goals_attemped']

        team1.varied_df['opp_eFG'] = team2.varied_df['eFG']
        team2.varied_df['opp_eFG'] = team1.varied_df['eFG']


        team1.varied_df['TO_rate'] = team1.varied_df['total_turnovers'] / (team1.varied_df['field_goals_attemped'] + 0.44 * team1.varied_df['free_throws_attempted'] + team1.varied_df['total_turnovers'])
        team2.varied_df['TO_rate'] = team2.varied_df['total_turnovers'] / (team2.varied_df['field_goals_attemped'] + 0.44 * team2.varied_df['free_throws_attempted'] + team2.varied_df['total_turnovers'])

        team1.varied_df['opp_TO_rate'] = team2.varied_df['TO_rate']
        team2.varied_df['opp_TO_rate'] = team1.varied_df['TO_rate']

        team1.varied_df['FT_rate'] = team1.varied_df['free_throws_made'] / team1.varied_df['field_goals_attemped']
        team2.varied_df['FT_rate'] = team2.varied_df['free_throws_made'] / team2.varied_df['field_goals_attemped']

        team1.varied_df['opp_FT_rate'] = team2.varied_df['FT_rate']
        team2.varied_df['opp_FT_rate'] = team1.varied_df['FT_rate']
        
        team1.varied_df['OREB_per'] = team1.varied_df['offensive_rebounds'] / (team1.varied_df['offensive_rebounds'] + team2.varied_df['defensive_rebounds'])
        team2.varied_df['OREB_per'] = team2.varied_df['offensive_rebounds'] / (team2.varied_df['offensive_rebounds'] + team1.varied_df['defensive_rebounds'])
        
        team1.varied_df['DREB_per'] = team1.varied_df['defensive_rebounds'] / (team1.varied_df['defensive_rebounds'] + team2.varied_df['offensive_rebounds'])
        team2.varied_df['DREB_per'] = team2.varied_df['defensive_rebounds'] / (team2.varied_df['defensive_rebounds'] + team1.varied_df['offensive_rebounds'])

    def adjust_metrics(self, team_to_adjust = None, opponent = None):
        
        """
        adjust for opponent value, home court advantage, and conference strength, opponent conference strength
        """

        metric_mapper = {
            'eFG':'adj_efg_percentage',
            'TO_rate':'adj_turnover_percentage',
            'OREB_per':'adj_offensive_rebound_percentage',
            'FT_rate':'adj_free_throw_rate',
            'opp_eFG':'opp_adj_efg_percentage',
            'opp_TO_rate':'opp_adj_turnover_percentage',
            'DREB_per':'adj_def_rebound_percentage',
            'opp_FT_rate':'opp_adj_free_throw_rate',
            'offensive_efficiency':'adj_offensive_efficiency',
            'defensive_efficiency':'adj_defensive_efficiency',
        }

        ridge_metrics = self.db_session.query(RidgeMetrics).all()
        for metric in ridge_metrics:

            mname = metric_mapper[metric.metric]

            adjusted_db_value = self.db_session.query(RidgeResults).filter(RidgeResults.opponent_id == opponent.team_id).filter(RidgeResults.season_id== self.season_id).filter(RidgeResults.metric_id == metric.id).first()

            opponent_value = adjusted_db_value.opponent_value

            home_value = adjusted_db_value.home_value
            opponent_conference_value = adjusted_db_value.conference_value

            team_conference_value = self.db_session.query(RidgeResults).filter(RidgeResults.opponent_id == team_to_adjust.team_id).filter(RidgeResults.season_id == self.season_id).filter(RidgeResults.metric_id == metric.id).first().conference_value

            team_to_adjust.varied_df[mname] = team_to_adjust.varied_df[metric.metric] - home_value - opponent_value - team_conference_value -opponent_conference_value 
        
        team_to_adjust.varied_df['adj_efficiency_margin'] = team_to_adjust.varied_df['adj_offensive_efficiency'] - team_to_adjust.varied_df['adj_defensive_efficiency']

    def check_distances(self, team1, team2):
        
        def get_coordinates(city_name):
            geolocator = Nominatim(user_agent="city_distance_calculator")
            location = geolocator.geocode(city_name, timeout=5)
            if location:
                return (location.latitude, location.longitude)
            else:
                return None
        
        def check_db_distance(game_loc, team_loc):
            distance = self.db_session.query(GameLocations).filter(GameLocations.location == game_loc).filter(GameLocations.team_location == team_loc).first()
            if distance is None:
                print(f"Distance not found in database. Need to calculate")
                coordinates1 = get_coordinates(game_loc)
                coordinates2 = get_coordinates(team_loc)

                if coordinates1 and coordinates2:
                    distance = round(geodesic(coordinates1, coordinates2).miles)
                    # print(f"Distance traveled: {distance}")
                    self.db_session.add(GameLocations(location = game_loc, team_location = team_loc, distance = distance))
                    self.db_session.commit()
                    return distance
                else:
                    return None

            else:
                # print(f"\n\nDistance: {distance.distance}\n\n")
                return distance.distance
        
        #get the distance traveled for the game

        #first get the team's location from the database

        team1_location = self.db_session.query(Teams).filter(Teams.espn_name == team1.team_name).first().location
        team2_location = self.db_session.query(Teams).filter(Teams.espn_name == team2.team_name).first().location

        #check if the entry is in the database:
        distance1 = check_db_distance(self.game_location, team1_location)
        distance2 = check_db_distance(self.game_location, team2_location)

        team1.varied_df['distance_traveled'] = distance1
        team2.varied_df['distance_traveled'] = distance2

        if distance1 ==0:
            print(f"{team1.team_name} are playing at home")
            team1.varied_df['home'] = 1
            team1.varied_df['away'] = 0
            team2.varied_df['away'] = 1
            team2.varied_df['home'] = 0
        elif distance2 == 0:
            print(f"{team2.team_name} are playing at home")
            team1.varied_df['away'] = 1
            team1.varied_df['home'] = 0
            team2.varied_df['home'] = 1
            team2.varied_df['away'] = 0
        else:
            print(f"Game is played at a neutral site")
            team1.varied_df['home'] = 0
            team2.varied_df['home'] = 0
            team1.varied_df['away'] = 0
            team2.varied_df['away'] = 0
    
    def simulate_game(self, team1, team2, over_under):

        # print(team1.varied_df[nn_model.params])
        # print(team2.varied_df[nn_model.params])
        #load in the scalars 
        
        
        #scale the data
        team1_scaled = self.ml_model.scaler.transform(team1.varied_df[self.ml_model.params].values)
        team2_scaled = self.ml_model.scaler.transform(team2.varied_df[self.ml_model.params].values)
        #convert to tensor
        team1_scaled = torch.from_numpy(team1_scaled.astype(np.float32))
        team2_scaled = torch.from_numpy(team2_scaled.astype(np.float32))
        #predict the score
        predicted_score1 = self.ml_model.predict(team1_scaled)
        predicted_score2 = self.ml_model.predict(team2_scaled)
        
        self.predicted_over_under = np.zeros_like(predicted_score1.detach().numpy())
        # print(f"Mean Predicted score for {team1.team_name}: {predicted_score1.mean().detach().numpy():0.1f}")
        # print(f"Median Predicted score for {team1.team_name}: {predicted_score1.median().detach().numpy():0.1f}")
        # print(f"Mean Predicted score for {team2.team_name}: {predicted_score2.mean().detach().numpy():0.1f}")
        # print(f"Median Predicted score for {team2.team_name}: {predicted_score2.median().detach().numpy():0.1f}")
        self.total_score = predicted_score1.detach().numpy() + predicted_score2.detach().numpy()
        self.pred_total_score = self.total_score.mean()
        if over_under:
            #count all values in the total score that are greater than the over/under
            self.predicted_over_under = np.zeros_like(self.total_score)
            self.predicted_over_under[self.total_score > over_under] = 1
        
        self.ou_pct = self.predicted_over_under.sum().item() / self.num_games

        team1_wins = predicted_score1 > predicted_score2
        #count the number of wins
        team1_wins = team1_wins.sum().item()
        team1_win_percentage = team1_wins / self.num_games
        if team1_win_percentage > 0.5:
            self.winner = team1.team_name
            self.win_percentage = team1_win_percentage
            self.winner_pts = np.round(predicted_score1.mean().detach().numpy(),2)
            self.loser = team2.team_name
            self.loser_pts = np.round(predicted_score2.mean().detach().numpy(),2)
            self.win_margin = self.winner_pts - self.loser_pts

        else:
            self.winner = team2.team_name
            self.win_percentage = 1 - team1_win_percentage
            self.winner_pts = np.round(predicted_score2.mean().detach().numpy(),2)
            self.loser = team1.team_name
            self.loser_pts = np.round(predicted_score1.mean().detach().numpy(),2)
            self.win_margin = self.winner_pts - self.loser_pts
        
        # return self.winner, self.win_percentage, self.winner_pts, self.loser, self.loser_pts, self.ou_pct, self.win_margin


def predict_game(team1 = None, team2 = None, num_games = 1000, season = None, game_location = None, db_session = None, over_under = None):

    
    nn_model = NN_Model()

    nn_model.load_model(db_session)

    team1 = ModelStats(team_name=team1, home_team=False, season=season, db_session=db_session)
    team2 = ModelStats(team_name=team2, home_team=True, season=season, db_session=db_session)
    
    matchup = ModelMatchup(team1=team1, team2=team2, num_games=num_games, season=season, game_location=game_location, db_session=db_session, ml_model=nn_model)

    matchup.calculate_metrics(team1, team2)
    matchup.adjust_metrics(team_to_adjust=team1, opponent=team2)
    matchup.adjust_metrics(team_to_adjust=team2, opponent=team1)
    matchup.check_distances(team1, team2)

    matchup.simulate_game(team1, team2, over_under)

    return matchup.winner, matchup.win_percentage, matchup.winner_pts, matchup.loser, matchup.loser_pts, matchup.ou_pct, matchup.win_margin, matchup.pred_total_score


if __name__ == '__main__':


    engine = create_engine('sqlite:///database/ncaa_basketball.db')
    Base = declarative_base()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    #retrieve the scalers and model parameters from the database

    nn_model = NN_Model()

    nn_model.load_model(session)
    
    
    # print(f"Here are the current model parameters: {nn_model.params}")


    season = 2025

    season_year = session.query(Season).filter(Season.year == season).first()

    team1 = ModelStats(team_name='Baylor Bears', home_team=True, season=season, db_session=session)
    team2 = ModelStats(team_name='Kansas Jayhawks', home_team=False, season=season, db_session=session)


    matchup = ModelMatchup(team1=team1, team2=team2, num_games=10000, game_location = 'Waco, TX ', season=season, db_session=session)

    matchup.calculate_metrics(team1, team2)
    matchup.adjust_metrics(team_to_adjust=team1, opponent=team2)
    matchup.adjust_metrics(team_to_adjust=team2, opponent=team1)
    matchup.check_distances(team1, team2)

    matchup.simulate_game(team1, team2)


