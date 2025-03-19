import os
import glob
import random
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from webscraping.web_scrapper import Scraper
from database.database import *
from models.NN_model import *
from single_prediction import ModelMatchup, ModelStats, predict_game

import time


data_file = r'.\04_tournaments\2025_ncaa_tournament_seeds.xlsx'


df = pd.read_excel(data_file)

all_teams = df['Team'].to_list()

regions = ['East', 'Midwest', 'South', 'West']
first_round = [1,2,3,4,5,6,7,8]

NUM_GAMES = 10000

# t1 = 'Iowa State Cyclones'
# t2 = 'South Dakota State Jackrabbits'

# sanity_check = test_df.groupby('Team')[['adj_Raw_Off_Eff','adj_Raw_Def_Eff','adj_EM']].mean().loc[[t1,t2]]
# print(sanity_check)

class team:
    def __init__(self, team_name=None, team_seed = None, region = None):
        self.team_name = team_name
        self.team_seed = team_seed
        self.region = region

class matchup:
    def __init__(self, team1, team2, game_location, num_games=50, game_id= None, game_round=None, next_round = None, current_round = None, region=None,
                  vary_params = None, model_params=None, season_data=None, ridge_df=None, standardizer=None,summary_stats=None,
                  tournament_sim = None,probability_dict=None,line=None):
        self.team1 = team1
        self.team2 = team2
        self.game_location = game_location
        self.num_games = num_games
        self.game_id = game_id
        self.game_round = game_round
        self.region = region
        self.vary_params = vary_params
        self.model_params = model_params
        self.season_data = season_data
        self.ridge_df = ridge_df
        self.input_standardizer = standardizer
        self.team1_wins = 0
        self.team1_winprob = 0 
        self.team2_wins = 0
        self.team2_winprob = 0
        self.game_winner = None
        self.upset = False
        self.game_winner_prob = 0
        self.game_mapper = {'Game_1':[1,2],
                            'Game_2':[3,4],
                            'Game_3':[5,6],
                            'Game_4':[7,8],}
        self.next_round = next_round
        self.current_round = current_round
        self.summary_stats = summary_stats
        self.tournament_sim = tournament_sim
        self.line = line
        self.over = 0 

        self.game_string = f"{self.team1.team_name} vs {self.team2.team_name}"
        
        if self.game_string not in probability_dict:
            probability_dict[self.game_string] = {self.team1.team_name:0, f"{self.team1.team_name}_score":0,
                                            self.team2.team_name:0, f"{self.team2.team_name}_score":0}
        
    
    
    
    def simulate_game(self):
        team1_scores = np.array([])
        team2_scores = np.array([])
        total_score = np.array([])
        for i in range(self.num_games):
            x1 = self.team_data[self.team1.team_name].iloc[i][self.model_params].values.reshape(1,-1)
            x2 = self.team_data[self.team2.team_name].iloc[i][self.model_params].values.reshape(1,-1)

            x1 = self.input_standardizer.transform(x1)
            x2 = self.input_standardizer.transform(x2)
            
            x1 = torch.from_numpy(x1.astype(np.float32))
            x2 = torch.from_numpy(x2.astype(np.float32))
            
            team1_score = ANNreg(x1)
            team2_score = ANNreg(x2)
            # print(f"game: {i+1}, {self.team1.team_name} score: {team1_score.detach().numpy()}, {self.team2.team_name} score: {team2_score.detach().numpy()}")
            team1_scores = np.append(team1_scores, team1_score.detach().numpy())
            team2_scores = np.append(team2_scores, team2_score.detach().numpy())
            total_score = np.append(total_score, team1_score.detach().numpy()+team2_score.detach().numpy())

            if team1_score > team2_score:
                self.team1_wins += 1
            else:
                self.team2_wins += 1
            
            if self.line:
                if team1_score.detach().numpy() + team2_score.detach().numpy() > self.line:
                    self.over += 1
        
        self.team1_winprob = self.team1_wins/self.num_games
        self.team2_winprob = self.team2_wins/self.num_games

        self.over_prob = self.over/self.num_games

        self.team1_score = team1_scores.mean()
        self.team1_score_sd = team1_scores.std()
        self.team2_score = team2_scores.mean()
        self.team2_score_sd = team2_scores.std()
        
                    

        prob_mapper[self.game_string][self.team1.team_name] = self.team1_winprob
        prob_mapper[self.game_string][f"{self.team1.team_name}_score"] = self.team1_score
        prob_mapper[self.game_string][self.team2.team_name] = self.team2_winprob
        prob_mapper[self.game_string][f"{self.team2.team_name}_score"] = self.team2_score
        prob_mapper[self.game_string]['flag'] = True
        
        # print(f"{self.team1.team_name} win probability: {self.team1_winprob:0.2f}, {self.team2.team_name} win probability: {self.team2_winprob:0.2f}")

        random_number = random.random()
        # print(f"random number: {random_number}")
        if random_number < self.team1_winprob:
            self.game_winner = self.team1.team_name
            self.game_winner_prob = self.team1_winprob
            self.winning_team = self.team1
            if self.team1.team_seed > self.team2.team_seed:
                self.upset = True
            else:
                self.upset = False
            # print(f"{self.team1.team_name} wins!")
        else:
            # print(f"{self.team2.team_name} wins!")
            self.game_winner = self.team2.team_name
            self.game_winner_prob = self.team2_winprob
            self.winning_team = self.team2
            if self.team2.team_seed > self.team1.team_seed:
                self.upset = True
            else:
                self.upset = False

    def game_summary(self):
        print(f"{self.team1.team_name} win probability: {self.team1_winprob:0.2f}, {self.team2.team_name} win probability: {self.team2_winprob:0.2f}")
        print(f"{self.game_winner} wins!")
        print(f"{self.team1.team_name} ({self.team1.team_seed}) vs {self.team2.team_name} ({self.team2.team_seed})\n"
              f"----------------------------------------------\n"
            #   f"score: {self.team1_score:0.2f} ± {self.team1_score_sd:0.2f} vs {self.team2_score:0.2f} ± {self.team2_score_sd:0.2f}\n"
              f"score: {self.team1_score:0.2f} vs {self.team2_score:0.2f}\n"
              f"total score: {self.team1_score+self.team2_score:0.2f}\n"
              f"over prob: {self.over_prob:0.2f}\n"
              f"{self.team1.team_name} win prob: {self.team1_winprob:0.2f} vs {self.team2.team_name} win prob: {self.team2_winprob:0.2f}\n"
        )
        if self.upset:
            print(f"upset!")
        if self.current_round == 'championship':
            self.summary_stats.loc[self.game_winner, self.current_round]+=1
            return

        for game, teams in self.game_mapper.items():
            if self.game_id in teams:
                self.next_game = game
                # print(f"{self.game_winner} advances to {game} of {self.next_round} in the {self.region}!\n")
                if self.next_round == 'final four':
                    self.game_round[self.winning_team.region] = self.winning_team
                elif self.next_round == 'championship':
                    self.game_round['Game_1'].append(self.winning_team)
                else:
                    self.game_round[self.region][game].append(self.winning_team)
                self.summary_stats.loc[self.game_winner, self.current_round]+=1
                
                # print(self.summary_stats)
            # print(f"{self.game_winner} win probability: {self.game_winner_prob:0.2f}") 

    def short_game(self):
        random_number = random.random()
        team1_win = prob_mapper[self.game_string][self.team1.team_name]
        team2_win = prob_mapper[self.game_string][self.team2.team_name]
        if random_number < team1_win:
            self.game_winner = self.team1.team_name
            self.game_winner_prob = self.team1_winprob
            self.winning_team = self.team1
            if self.team1.team_seed > self.team2.team_seed:
                self.upset = True
            else:
                self.upset = False
        else:
            self.game_winner = self.team2.team_name
            self.game_winner_prob = self.team2_winprob
            self.winning_team = self.team2
            if self.team2.team_seed > self.team1.team_seed:
                self.upset = True
            else:
                self.upset = False

class Tournament():
    def __init__(self, seed_df, teams, summary_stats, num_games, num_tournaments):
        self.seed_df = seed_df
        self.teams = teams
        self.summary_stats = summary_stats
        self.num_games = num_games
        self.num_tournaments = num_tournaments
        self.probability_dict = {}
        self.regions = ['East','West','Midwest','South']
        self.first_round = [1,2,3,4,5,6,7,8]
        self.game_mapper = {'Game_1':[1,2],
                            'Game_2':[3,4],
                            'Game_3':[5,6],
                            'Game_4':[7,8],}
        self.game_rounds = ['round of 64','round of 32','Sweet 16','Elite 8','Final Four','Championship']
        self.tourney_locs = {'round of 32':{'East':{'Game_1':'Raleigh, NC',
                                       'Game_2':'Seattle, WA',
                                       'Game_3':'Denver, CO',
                                        'Game_4':'Cleveland, OH'
                                        },
                               'West':{'Game_1':'Raleigh, NC',
                                        'Game_2':'Seattle, WA',
                                        'Game_3':'Wichita, KS',
                                        'Game_4':'Providence, RI'
                                        },
                                'Midwest':{'Game_1':'Wichita, KS',
                                        'Game_2':'Providence, RI',
                                        'Game_3':'Milwaukee, WI',
                                        'Game_4':'Lexington, KY'
                                        },
                                'South':{'Game_1':'Lexington, KY',
                                        'Game_2':'Denver, CO',
                                        'Game_3':'Milwaukee, WI',
                                        'Game_4':'Cleveland, OH'
                                        }
                                 },
                'Sweet 16':{'East':{'Game_1':'Newark, NJ',
                                    'Game_2':'Newark, NJ',
                                        },
                            'West':{'Game_1':'San Francisco, CA',
                                    'Game_2':'San Francisco, CA',
                                    },
                            'Midwest':{'Game_1':'Indianapolis, IN',
                                       'Game_2':'Indianapolis, IN',
                                    },
                            'South':{'Game_1':'Atlanta, GA',
                                     'Game_2':'Atlanta, GA',
                                    }
                            },
                'Elite 8':{'East':{'Game_1':'Newark, NJ'},
                            'West':{'Game_1':'San Francisco, CA'},
                            'Midwest':{'Game_1':'Indianapolis, IN'},
                            'South':{'Game_1':'Atlanta, GA'}
                            },
                'Final Four':{'Game_1':'San Antonio, TX',
                              'Game_2':'San Antonio, TX'
                            },
                'Championship':{'Game_1':'San Antonio, TX'}
                               }
        self.mapping = {'round of 32': {'East':{'Game_1':[],
                                             'Game_2':[],
                                             'Game_3':[],
                                             'Game_4':[]},
                                     'West':{'Game_1':[],
                                             'Game_2':[],
                                             'Game_3':[],
                                             'Game_4':[]},
                                     'Midwest':{'Game_1':[],
                                             'Game_2':[],
                                             'Game_3':[],
                                             'Game_4':[]},
                                     'South':{'Game_1':[],
                                             'Game_2':[],
                                             'Game_3':[],
                                             'Game_4':[]
                                             }
                                             },
                        'Sweet 16':{'East':{'Game_1':[],
                                            'Game_2':[]},
                                    'West':{'Game_1':[],
                                            'Game_2':[]},
                                    'Midwest':{'Game_1':[],
                                            'Game_2':[]},
                                    'South':{'Game_1':[],
                                            'Game_2':[]
                                            }
                                    },
                        'Elite 8':{'East':{'Game_1':[]},
                                    'West':{'Game_1':[]},
                                    'Midwest':{'Game_1':[]},
                                    'South':{'Game_1':[]}
                                    },
                        'Final Four':{'East':None,
                                      'West':None,
                                      'Midwest':None,
                                       'South':None
                                        },
                        'Championship':{'Game_1':[]}
                }
    
    def game_prediction(self, game_loc, prediction_method, team1, team2):

        game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts = predict_game(team1 = team1.team_name, team2 = team2.team_name, game_location = game_loc, db_session = session, num_games = NUM_GAMES, season = 2025, over_under = None,method = prediction_method)

        return game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts
    
    def simulate_tournament(self, tourney_num):
        for rounds in self.game_rounds:
            print(f"\nStarting {rounds} of tournament {tourney_num}\n")
            if rounds == 'round of 64':
                next_round = 'round of 32'
                for region in self.regions:
                    for game in self.first_round:
                        sdf = self.seed_df[(df['Game'] == game) & (df['Region']==region)]
                        game_loc = sdf['Location'].values[0].lstrip()
                        game_id = sdf['Game'].values[0]
                        team1 = team(team_name=sdf['Team'].values[0], team_seed=sdf['Seed'].values[0], region=sdf['Region'].values[0])
                        team2 = team(team_name=sdf['Team'].values[1], team_seed=sdf['Seed'].values[1], region=sdf['Region'].values[1])
                        game_string = f"{team1.team_name} vs {team2.team_name}"
                        #check if the win probability has already been calculated
                        if game_string in self.probability_dict:
                            print(self.probability_dict[game_string])
                            # print(f"already simulated {game_string}. let me check the probability")
                            # print(f"{team1.team_name} has a {self.probability_dict[game_string][team1.team_name]} win chance")
                            random_number = random.random()
                            if random_number < self.probability_dict[game_string][team1.team_name]:
                                game_winner = team1
                            else:
                                game_winner = team2
                        else:
                            self.probability_dict[game_string] = {team1.team_name:0, team2.team_name:0}
                            game_location = game_loc.lstrip()
                            game_location+=' '
                            gloc = game_location
                            game_location = session.query(GameLocations).filter(GameLocations.location.ilike(f'%{game_location}%')).first()
                            if game_location is None:
                                game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts = self.game_prediction(gloc,'season_stats', team1, team2)
                            else:    
                                game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts= self.game_prediction(game_location.location,'season_stats', team1, team2)
                            
                            print(f"{game_winner} beats {game_loser} {win_pts:0.2f} to {loser_pts:0.2f} with a win probability of {win_pct:0.2f}")
                            #store the probabilities for later
                            if game_winner == team1.team_name:
                                gw = team1 
                                gl = team2 
                            else:
                                gw = team2
                                gl = team1

                            self.probability_dict[game_string][gw.team_name] = win_pct
                            self.probability_dict[game_string][gl.team_name] = 1 - win_pct
                            print(self.probability_dict[game_string])
                            random_number = random.random()
                            if game_winner == team1.team_name:
                                if random_number < win_pct:
                                    game_winner = team1
                                else:
                                    game_winner = team2
                            else:
                                if random_number < win_pct:
                                    game_winner = team2
                                else:
                                    game_winner = team1

                        for game, teams in self.game_mapper.items():
                            if game_id in teams:
                                
                                self.mapping[next_round][region][game].append(game_winner)
                                self.summary_stats.loc[game_winner.team_name, rounds]+=1
                                break
            
            elif rounds == 'Final Four':
                ff_matchup = {'Game_1':[self.mapping['Final Four']['East'],self.mapping['Final Four']['West']],
                              'Game_2':[self.mapping['Final Four']['Midwest'],self.mapping['Final Four']['South']]}
                for game in ff_matchup:
                    team1 = ff_matchup[game][0]
                    team2 = ff_matchup[game][1]
                    game_location = self.tourney_locs[rounds][game].lstrip()
                    game_location+=' '
                    gloc = game_location
                    game_string = f"{team1.team_name} vs {team2.team_name}"
                    if game_string in self.probability_dict:
                        random_number = random.random()
                        if random_number < self.probability_dict[game_string][team1.team_name]:
                            game_winner = team1
                        else:
                            game_winner = team2
                    else:
                        self.probability_dict[game_string] = {team1.team_name:0, team2.team_name:0}
                        game_location = session.query(GameLocations).filter(GameLocations.location.ilike(f'%{game_location}%')).first()
                    
                        if game_location is None:
                            game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts = self.game_prediction(gloc,'season_stats', team1, team2)
                        else:    
                            game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts= self.game_prediction(game_location.location,'season_stats', team1, team2)
                        
                        # print(f"{game_winner} beats {game_loser} {win_pts:0.2f} to {loser_pts:0.2f} with a win probability of {win_pct:0.2f}")
                        if game_winner == team1.team_name:
                            gw = team1
                            gl = team2
                        else:
                            gw = team2
                            gl = team1
                        self.probability_dict[game_string][gw.team_name] = win_pct
                        self.probability_dict[game_string][gl.team_name] = 1 - win_pct

                        random_number = random.random()
                        if random_number < win_pct:
                            game_winner = team1
                        else:
                            game_winner = team2

                    for game, teams in self.game_mapper.items():
                        if game_id in teams:
                            self.mapping['Championship'][game].append(game_winner)
                            self.summary_stats.loc[game_winner.team_name, rounds]+=1
                            break
            elif rounds == 'Championship':
                team1 = self.mapping[rounds]['Game_1'][0]
                team2 = self.mapping[rounds]['Game_1'][1]
                game_location = self.tourney_locs[rounds][game].lstrip()
                game_location+=' '
                gloc = game_location
                game_string = f"{team1.team_name} vs {team2.team_name}"
                if game_string in self.probability_dict:
                    random_number = random.random()
                    if random_number < self.probability_dict[game_string][team1.team_name]:
                        game_winner = team1
                        print(f"{game_winner.team_name} beats {team2.team_name} to win the championship!")
                    else:
                        game_winner = team2
                        print(f"{game_winner.team_name} beats {team1.team_name} to win the championship!")
                    self.summary_stats.loc[game_winner.team_name, rounds]+=1
                    break
                else:
                    self.probability_dict[game_string] = {team1.team_name:0, team2.team_name:0}
                game_location = session.query(GameLocations).filter(GameLocations.location.ilike(f'%{game_location}%')).first()
                
                if game_location is None:
                    game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts = self.game_prediction(gloc,'season_stats', team1, team2)
                else:    
                    game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts= self.game_prediction(game_location.location,'season_stats', team1, team2)
              
                    
                    self.probability_dict[game_string][team1.team_name] = win_pct
                    self.probability_dict[game_string][team2.team_name] = 1 - win_pct
                    random_number = random.random()
                    if random_number < win_pct:
                        game_winner = team1
                        print(f"{game_winner.team_name} beats {team2.team_name} to win the championship!")
                    else:
                        game_winner = team2
                        print(f"{game_winner.team_name} beats {team1.team_name} to win the championship!")
                    self.summary_stats.loc[game_winner.team_name, rounds]+=1
                    break

            else:
                next_round = self.game_rounds[self.game_rounds.index(rounds)+1]
                for region in self.regions:
                    for game in self.mapping[rounds][region]:
                        # print(f"\n{game}\n")
                        team1 = self.mapping[rounds][region][game][0]
                        team2 = self.mapping[rounds][region][game][1]
                        game_id=int(game.split('_')[-1])
                        game_location = self.tourney_locs[rounds][region][game].lstrip()
                        game_location+=' '
                        gloc = game_location
                        game_string = f"{team1.team_name} vs {team2.team_name}"
                        if game_string in self.probability_dict:
                            random_number = random.random()
                            if random_number < self.probability_dict[game_string][team1.team_name]:
                                game_winner = team1
                            else:
                                game_winner = team2
                        else:
                            self.probability_dict[game_string] = {team1.team_name:0, team2.team_name:0}
                            game_location = session.query(GameLocations).filter(GameLocations.location.ilike(f'%{game_location}%')).first()

                            team1 = team(team_name=team1.team_name, team_seed=team1.team_seed, region=team1.region)
                            team2 = team(team_name=team2.team_name, team_seed=team2.team_seed, region=team2.region)
                            
                            if game_location is None:
                                game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts = self.game_prediction(gloc,'season_stats', team1, team2)
                            else:    
                                game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts= self.game_prediction(game_location.location,'season_stats', team1, team2)
                            
                            # print(f"{game_winner} beats {game_loser} {win_pts:0.2f} to {loser_pts:0.2f} with a win probability of {win_pct:0.2f}")

                            if game_winner == team1.team_name:
                                gw = team1
                                gl = team2
                            else:
                                gw = team2
                                gl = team1

                            self.probability_dict[game_string][gw.team_name] = win_pct
                            self.probability_dict[game_string][gl.team_name] = 1 - win_pct
                            
                            random_number = random.random()
                            if random_number < win_pct:
                                game_winner = team1
                            else:
                                game_winner = team2
                        for game, teams in self.game_mapper.items():
                            if game_id in teams:
                                if next_round == 'Final Four':
                                    self.mapping[next_round][game_winner.region] = game_winner
                                elif next_round == 'Championship':
                                    self.mapping[next_round]['Game_1'].append(game_winner)
                                else:
                                    self.mapping[next_round][region][game].append(game_winner)
                                self.summary_stats.loc[game_winner.team_name, rounds]+=1
                                break


###################################################################################################### 
##### --------------------------- STARTING THE TOURNAMENT ---------------------------------------- ###
######################################################################################################
engine = create_engine('sqlite:///database/ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

t1 = time.time()
prob_mapper = {}

tourney_locs = {'round of 32':{'East':{'Game_1':'Raleigh, NC',
                                       'Game_2':'Seattle, WA',
                                       'Game_3':'Denver, CO',
                                        'Game_4':'Cleveland, OH'
                                        },
                               'West':{'Game_1':'Raleigh, NC',
                                        'Game_2':'Seattle, WA',
                                        'Game_3':'Wichita, KS',
                                        'Game_4':'Providence, RI'
                                        },
                                'Midwest':{'Game_1':'Wichita, KS',
                                        'Game_2':'Providence, RI',
                                        'Game_3':'Milwaukee, WI',
                                        'Game_4':'Lexington, KY'
                                        },
                                'South':{'Game_1':'Lexington, KY',
                                        'Game_2':'Denver, CO',
                                        'Game_3':'Milwaukee, WI',
                                        'Game_4':'Cleveland, OH'
                                        }
                                 },
                'sweet 16':{'East':{'Game_1':'Newark, NJ',
                                    'Game_2':'Newark, NJ',
                                        },
                            'West':{'Game_1':'San Francisco, CA',
                                    'Game_2':'San Francisco, CA',
                                    },
                            'Midwest':{'Game_1':'Indianapolis, IN',
                                       'Game_2':'Indianapolis, IN',
                                    },
                            'South':{'Game_1':'Atlanta, GA',
                                     'Game_2':'Atlanta, GA',
                                    }
                            },
                'Elite 8':{'East':{'Game_1':'Newark, NJ'},
                            'West':{'Game_1':'San Francisco, CA'},
                            'Midwest':{'Game_1':'Indianapolis, IN'},
                            'South':{'Game_1':'Atlanta, GA'}
                            },
                'final four':{'Game_1':'San Antonio, TX',
                              'Game_2':'San Antonio, TX'
                            },
                'championship':{'Game_1':'San Antonio, TX'}
                               }

game_rounds = ['round of 64','round of 32','Sweet 16','Elite 8','Final Four','Championship']
summary_tournament = pd.DataFrame(columns = game_rounds,index=all_teams,data=0)

ncaa_tournament = Tournament(seed_df = df, teams=all_teams, summary_stats=summary_tournament, num_games=NUM_GAMES, num_tournaments=1)


good_regions = ['East','West','Midwest','South']
NUM_TOURNAMENTS = 100

for i in range(NUM_TOURNAMENTS):
    counter = i + 1
    print(f"\nSimulating tournament {i+1} of {NUM_TOURNAMENTS}\n")
    ncaa_tournament.simulate_tournament(counter)

ncaa_tournament.summary_stats = ncaa_tournament.summary_stats/NUM_TOURNAMENTS

t2 = time.time()

print(f"Time taken to simulate tournament: {t2-t1:0.2f} seconds")
print(f"Summary of tournament: \n{ncaa_tournament.summary_stats}")

print(ncaa_tournament.summary_stats['Championship'].sort_values(ascending=False).head(10))
# for tidx in range(1,NUM_TOURNAMENTS+1):
#     t1 = time.time()
