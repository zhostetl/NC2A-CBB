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
from sklearn.neighbors import NearestNeighbors
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
        'two_point_field_goals_attempted','three_point_field_goals_attempted','field_goals_made', 'field_goals_attemped', 'free_throws_made',  'free_throws_attempted', 'offensive_rebounds', 'defensive_rebounds','points',
        'offensive_efficiency','defensive_efficiency','eFG','opp_eFG','TO_rate','opp_TO_rate','FT_rate','opp_FT_rate','OREB_per','DREB_per',
        'pace']
        
        """
        This probably needs to be moved down below and referenced depending on which method is used to predict the scores
        """
       


class ModelMatchup():

    def __init__(self, team1 = None, team2 = None, num_games = 1000, season = None, game_location = None, db_session = None, ml_model = None, prediction_method = 'nearest_neighbor'):

        self.team1 = team1
        self.team2 = team2
        self.num_games = num_games
        self.season = season
        self.game_location = game_location
        self.db_session = db_session
        self.season_id = self.db_session.query(Season).filter(Season.year == self.season).first().id
        self.ml_model = ml_model
        self.prediction_method = prediction_method
        #set up the array to randomly sample the baseline stats and then calculate additional metrics
    
        self.generate_samples(team_of_interest = self.team1, opponent = self.team2)
        self.generate_samples(team_of_interest = self.team2, opponent = self.team1)
    
    def get_season_data(self):
        #get season stats for the nearest neighbor search method
        games = self.db_session.query(Games).filter(Games.season_id == self.season_id).all()
        season_data = self.db_session.query(Games, AdjustedMetrics, Teams).join(AdjustedMetrics, (Games.game_id == AdjustedMetrics.game_id) & (Games.team_id == AdjustedMetrics.team_id)).join(Teams, Games.team_id==Teams.id).filter(Games.season_id == self.season_id).all()

        game_df = pd.DataFrame([game.__dict__ for game in games])

        # Count the number of times each game_id occurs in the DataFrame
        game_id_counts = game_df['game_id'].value_counts()

        dup_games = game_id_counts[game_id_counts > 2]
    
        off_by = len(season_data) - len(games)

        df = pd.DataFrame([{**game.__dict__, **metrics.__dict__, **teams.__dict__} for game, metrics, teams, in season_data])

        df = df.drop(columns=['_sa_instance_state','game_location','game_state','over_under','betting_line','date','referee1','referee2','referee3','espn_name','name','location'])

        sdf = df.groupby('team_id').mean()
        # print(f"standard deviation of season stats: {df.groupby('team_id').std()}")

        #now map the team id back to the team name for display purposes 
        teams = self.db_session.query(Teams).all()
        team_dict = {team.id: team.espn_name for team in teams}

        sdf['team_name'] = sdf.index.map(team_dict)
        sdf = sdf.set_index('team_name')

        keep_cols = ['adj_offensive_efficiency','adj_defensive_efficiency','adj_efficiency_margin','pace']

        self.season_stats = sdf

    def query_all_data(self):
        
        TeamAdjMetrics = aliased(AdjustedMetrics)
        OpponentAdjMetrics = aliased(AdjustedMetrics)
        Opponent = aliased(Teams)
        TeamMetrics = aliased(Games)
        OpponentMetrics = aliased(Games)
        all_data = self.db_session.query(Games,
                                        TeamAdjMetrics.adj_offensive_efficiency.label('team_off_eff'),
                                        TeamAdjMetrics.adj_defensive_efficiency.label('team_def_eff'),
                                        TeamAdjMetrics.adj_efficiency_margin.label('team_eff_margin'),
                                        TeamMetrics.pace.label('team_pace'),
                                        TeamMetrics.two_point_field_goal_percentage.label('team_two_point_field_goal_percentage'),
                                        TeamMetrics.three_point_field_goal_percentage.label('team_three_point_field_goal_percentage'),
                                        TeamMetrics.free_throw_percentage.label('team_free_throw_percentage'),
                                        OpponentAdjMetrics.adj_offensive_efficiency.label('opp_off_eff'),
                                        OpponentAdjMetrics.adj_defensive_efficiency.label('opp_def_eff'),
                                        OpponentAdjMetrics.adj_efficiency_margin.label('opp_eff_margin'),
                                        OpponentMetrics.pace.label('opp_pace'),
                                        OpponentMetrics.two_point_field_goal_percentage.label('opp_two_point_field_goal_percentage'),
                                        OpponentMetrics.three_point_field_goal_percentage.label('opp_three_point_field_goal_percentage'),
                                        OpponentMetrics.free_throw_percentage.label('opp_free_throw_percentage'),
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
                                        ).join(TeamAdjMetrics, (Games.game_id == TeamAdjMetrics.game_id) & (Games.team_id == TeamAdjMetrics.team_id)
                                        ).join(OpponentAdjMetrics, (Games.game_id == OpponentAdjMetrics.game_id) & (Games.opponent_id == OpponentAdjMetrics.team_id)).all()
                                            

        # combined = pd.DataFrame([{**game.__dict__, **metrics.__dict__, **teams.__dict__} for game, metrics, teams, in all_data])
        combined = pd.DataFrame(all_data)
        search_cols = ['team_off_eff','team_def_eff','team_eff_margin','team_pace','team_two_point_field_goal_percentage','team_three_point_field_goal_percentage','team_free_throw_percentage',
        'opp_off_eff','opp_def_eff','opp_eff_margin','opp_pace','opp_two_point_field_goal_percentage','opp_three_point_field_goal_percentage','opp_free_throw_percentage']
        
        all_df = combined[search_cols]
        return all_df
    
    def season_trends(self, team_name):

        team_id = self.db_session.query(Teams).filter(Teams.espn_name == team_name).first().id

        OpponentAdjMetrics = aliased(AdjustedMetrics)
        OpponentMetrics = aliased(Games)
        
        season_games = self.db_session.query(Games, Teams, OpponentAdjMetrics,OpponentMetrics.pace.label('opp_pace'),OpponentMetrics.possessions.label('opp_possessions')).join(Teams, Games.team_id == Teams.id).join(OpponentAdjMetrics, (Games.game_id == OpponentAdjMetrics.game_id) & (Games.opponent_id == OpponentAdjMetrics.team_id)).join(OpponentMetrics, (Games.game_id==OpponentMetrics.game_id) &(Games.opponent_id == OpponentMetrics.team_id)).filter(Games.season_id == self.season_id, Games.team_id == team_id).all()
    
        #unpack each tuple to dataframe
        self.season_df = pd.DataFrame([{**game.__dict__, **teams.__dict__, **opponent.__dict__, 'opp_pace':opp_pace,'opp_possessions':opp_possessions} for game, teams, opponent,opp_pace,opp_possessions in season_games])
   
    def nearest_game_search(self, team_of_interest = None, opponent = None, n_neighbors=5):
        # print(f"looking up season data for {team_of_interest.team_name}")
        #first look up all games that the team of interest has played and find the adjusted values for the opponents they faced
        self.season_trends(team_name = team_of_interest.team_name)
        search_cols = ['adj_defensive_efficiency','adj_efficiency_margin','opp_pace'] 
        search_cols2 = ['adj_defensive_efficiency','adj_efficiency_margin','pace']
        X = self.season_df[search_cols].values
        # #fit the model
        nn = NearestNeighbors(n_neighbors=n_neighbors,algorithm='ball_tree')
        nn.fit(X)

        opponent_value = self.season_stats.loc[opponent.team_name]
        # print(f"searching most similar games against {opponent.team_name}")
        neighbor_search = opponent_value[search_cols2].values.reshape(1, -1)
        distance, indicies = nn.kneighbors(neighbor_search)
        # print(f"distances for nearest neighbors: {distance}")


        common_stats = team_of_interest.df.iloc[indicies[0]]
        
        common_stats = common_stats[self.team1.boxscore_params]
        
        team_of_interest.means = common_stats.mean()
        team_of_interest.stds = common_stats.std()

    def generate_samples(self, team_of_interest = None, opponent = None):
        if self.prediction_method == 'season_stats':
            
            df = team_of_interest.df[team_of_interest.boxscore_params]
            team_of_interest.means = df.mean()
            team_of_interest.stds = df.std()
            team_of_interest.samples = np.empty((self.num_games, len(team_of_interest.means)))
            for i in range(self.num_games):
                team_of_interest.samples[i] = np.random.normal(team_of_interest.means, team_of_interest.stds)
            team_of_interest.varied_df = pd.DataFrame(team_of_interest.samples, columns=team_of_interest.boxscore_params)
            # print(f"varied dataframe for {team_of_interest.team_name} using {self.prediction_method}: {team_of_interest.varied_df}")
            
        elif self.prediction_method == 'nearest_neighbor':
            self.get_season_data()
            # all_df = self.query_all_data()
            self.nearest_game_search(team_of_interest = team_of_interest, opponent = opponent)

            team_of_interest.samples = np.empty((self.num_games, len(team_of_interest.means)))
            for i in range(self.num_games):
                team_of_interest.samples[i] = np.random.normal(team_of_interest.means, team_of_interest.stds)
            team_of_interest.varied_df = pd.DataFrame(team_of_interest.samples, columns=team_of_interest.boxscore_params)
            
            # print(f"varied dataframe for {team_of_interest.team_name}: {team_of_interest.varied_df.head()}")
        
    def calculate_metrics(self, team1, team2):
        
        team1.varied_df['possessions'] = 0.96 * (team1.varied_df['field_goals_attemped'] + team1.varied_df['total_turnovers'] + 0.44 * team1.varied_df['free_throws_attempted']- team1.varied_df['offensive_rebounds'])
        
        team2.varied_df['possessions'] = 0.96 * (team2.varied_df['field_goals_attemped'] + team2.varied_df['total_turnovers'] + 0.44 * team2.varied_df['free_throws_attempted']- team2.varied_df['offensive_rebounds'])

        team1.varied_df['opp_pace'] = team2.varied_df['pace']
        team2.varied_df['opp_pace'] = team1.varied_df['pace']

        # team1.varied_df['pace'] = (40 * (team1.varied_df['possessions'] + team2.varied_df['possessions']) / 80)
        # team2.varied_df['pace'] = (40 * (team1.varied_df['possessions'] + team2.varied_df['possessions']) / 80)

        ### calculate the offensive and defensive efficiency ###
        
        # team1.varied_df['offensive_efficiency'] = team1.varied_df['points'] / team1.varied_df['possessions']
        # team2.varied_df['offensive_efficiency'] = team2.varied_df['points'] / team2.varied_df['possessions']

        # team1.varied_df['defensive_efficiency'] = team2.varied_df['points'] / team1.varied_df['possessions']
        # team2.varied_df['defensive_efficiency'] = team1.varied_df['points'] / team2.varied_df['possessions']

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
            away_value = adjusted_db_value.away_value
            opponent_conference_value = adjusted_db_value.conference_value

            if team_to_adjust.home_team:
                ha = home_value
            elif team_to_adjust.away:
                ha = away_value
            elif team_to_adjust.neutral: 
                ha = 0 

            # team_conference_value = self.db_session.query(RidgeResults).filter(RidgeResults.opponent_id == team_to_adjust.team_id).filter(RidgeResults.season_id == self.season_id).filter(RidgeResults.metric_id == metric.id).first().conference_value

            team_to_adjust.varied_df[mname] = team_to_adjust.varied_df[metric.metric] - ha - opponent_value -opponent_conference_value 
        
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
            team1.home_team = True
            team1.away = False
            team1.neutral = False
            team2.varied_df['away'] = 1
            team2.varied_df['home'] = 0
            team2.home_team = False
            team2.away = True
            team2.neutral = False
            
        elif distance2 == 0:
            print(f"{team2.team_name} are playing at home")
            team1.varied_df['away'] = 1
            team1.varied_df['home'] = 0
            team1.home_team = False
            team1.away = True
            team1.neutral = False
            team2.varied_df['home'] = 1
            team2.varied_df['away'] = 0
            team2.home_team = True
            team2.away = False
            team2.neutral = False
        else:
            print(f"Game is played at a neutral site")
            team1.varied_df['home'] = 0
            team2.varied_df['home'] = 0
            team1.varied_df['away'] = 0
            team2.varied_df['away'] = 0
            team1.neutral = True
            team2.neutral = True
            team1.home_team = False
            team2.home_team = False
            team1.away = False
            team2.away = False
    
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
    matchup.check_distances(team1, team2)
    matchup.adjust_metrics(team_to_adjust=team1, opponent=team2)
    matchup.adjust_metrics(team_to_adjust=team2, opponent=team1)
    

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


