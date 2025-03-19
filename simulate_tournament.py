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

###################################################################################################### 
##### --------------------------- STARTING THE TOURNAMENT ---------------------------------------- ###
######################################################################################################
engine = create_engine('sqlite:///database/ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

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

game_rounds = ['round of 64','round of 32','sweet 16','Elite 8','final four','championship']
summary_tournament = pd.DataFrame(columns = game_rounds,index=all_teams,data=0)
good_regions = ['East','West','Midwest','South']
NUM_TOURNAMENTS = 1
for tidx in range(1,NUM_TOURNAMENTS+1):
    t1 = time.time()
    round_32 = {'East':{'Game_1':[],
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
                }
    good_regions = ['East','West','Midwest','South']
    for region in regions:
        if region not in good_regions:
            continue
        # if region !='East':
        #     continue
        current_round = 'round of 64'
        print(f"\n\n starting {current_round}\n\n")
        for game in first_round:
            sdf = df[(df['Game'] == game) & (df['Region']==region)]
            game_loc = sdf['Location'].values[0].lstrip()
            game_id = sdf['Game'].values[0]
            team1 = team(team_name=sdf['Team'].values[0], team_seed=sdf['Seed'].values[0], region=region)
            team2 = team(team_name=sdf['Team'].values[1], team_seed=sdf['Seed'].values[1], region=region)

            game_location = game_location.lstrip()
            game_location = session.query(GameLocations).filter(GameLocations.location.ilike(f'%{game_location}%')).first()
            gloc = game_location
            if game_location is None:
                print(f"Game location {gloc} not found in database")
                game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts = predict_game(team1 = team1.team_name, team2 = team2.team_name, game_location = gloc, db_session = session, num_games = NUM_GAMES, season = 2025, over_under = None)
            else:

                print('\n********** Analyzing the following game **********\n')
                print(f"{team1.team_name} vs {team2.team_name} at {game_location.location}\n")
            
                game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts = predict_game(team1 = team1.team_name, team2 = team2.team_name, game_location=game_location.location, db_session=session, num_games=NUM_GAMES, season=2025, over_under=None)

            
            
#             if prob_mapper[game.game_string][team1.team_name] ==0:
#                 game.vary_stats()
#             # ax.hist(game.team_data[team1.team_name]['adj_EM'], bins = 50, alpha = 0.5, label = team1.team_name)
#             # ax.hist(game.team_data[team2.team_name]['adj_EM'], bins = 50, alpha = 0.5, label = team2.team_name)
#             # ax.legend()
#             # ax.set_title(f"adjusted offensive efficiency for {team1.team_name} and {team2.team_name}")
#                 game.simulate_game()
#             else: 
#                 game.short_game()
#             game.game_summary()
#             # round_32[game.next_game].append(game.game_winner)
#     # plt.show()
#     # print(summary_tournament)
            
#     round_16 = {'East':{'Game_1':[],
#                         'Game_2':[]},
#                 'West':{'Game_1':[],
#                         'Game_2':[]},
#                 'Midwest':{'Game_1':[],
#                         'Game_2':[]},
#                 'South':{'Game_1':[],
#                         'Game_2':[]}
#     }

#     # print(round_32)
  
#     current_round = 'round of 32'
#     print(f"\n\n starting {current_round}\n\n")
#     for region in good_regions:

#         for game, teams in round_32[region].items():
#             game_loc = tourney_locs[current_round][region][game]
#             # print(game_loc)
#             team1 = team(team_name=teams[0].team_name, team_seed=teams[0].team_seed, region=teams[0].region)
#             team2 = team(team_name=teams[1].team_name, team_seed=teams[1].team_seed, region=teams[1].region)
#             game = matchup(team1, team2, game_location=game_loc, num_games=NUM_GAMES, game_id=int(game.split('_')[-1]), game_round=round_16, next_round= 'sweet 16',current_round = 'round of 32', 
#                         region=region, vary_params=VARY_PARAMS, model_params=NN_PARAMS, season_data=MODEL_DF, ridge_df=RIDGE_DF, standardizer=SCALER,
#                         summary_stats=summary_tournament,tournament_sim=tidx,probability_dict=prob_mapper)
#             if prob_mapper[game.game_string][team1.team_name] ==0:
#                 game.vary_stats()
#                 game.simulate_game()
#             else:
#                 game.short_game()
#             game.game_summary()
#             # game.vary_stats()
#             # game.simulate_game()
#             # game.game_summary()


#     round_8 = {'East':{'Game_1':[]},
#             'West':{'Game_1':[]},
#                 'Midwest':{'Game_1':[]},
#                 'South':{'Game_1':[]}
#     }
#     # print(f"\n\n starting next round\n\n")
#     current_round = 'sweet 16'
#     print(f"\n\n starting {current_round}\n\n")
#     for region in good_regions:

#         for game, teams in round_16[region].items():
#             game_loc = tourney_locs[current_round][region][game]
#             team1 = team(team_name=teams[0].team_name, team_seed=teams[0].team_seed, region=teams[0].region)
#             team2 = team(team_name=teams[1].team_name, team_seed=teams[1].team_seed, region=teams[1].region)
#             game = matchup(team1, team2, game_location=game_loc, num_games=NUM_GAMES, game_id=int(game.split('_')[-1]), game_round=round_8, next_round = 'Elite 8', current_round= current_round,
#                         region=region, vary_params=VARY_PARAMS, model_params=NN_PARAMS, season_data=MODEL_DF, ridge_df=RIDGE_DF, standardizer=SCALER,
#                         summary_stats=summary_tournament,tournament_sim=tidx,probability_dict=prob_mapper)
#             if prob_mapper[game.game_string][team1.team_name] ==0:
#                 game.vary_stats()
#                 game.simulate_game()
#             else:
#                 game.short_game()
#             # game.vary_stats()
#             # game.simulate_game()
#             game.game_summary()

#     final_four = {'East':None,
#                 'West':None,
#                 'Midwest':None,
#                 'South':None}

#     current_round = 'Elite 8'
#     print(f"\n\n starting {current_round}\n\n")
#     for region in good_regions:

#         for game, teams in round_8[region].items():
#             game_loc = tourney_locs[current_round][region][game]
#             team1 = team(team_name=teams[0].team_name, team_seed=teams[0].team_seed, region=teams[0].region)
#             team2 = team(team_name=teams[1].team_name, team_seed=teams[1].team_seed, region=teams[1].region)
#             game = matchup(team1, team2, game_location=game_loc, num_games=NUM_GAMES, game_id=int(game.split('_')[-1]), game_round=final_four, next_round = 'final four',current_round=current_round,
#                         region=None, vary_params=VARY_PARAMS, model_params=NN_PARAMS, season_data=MODEL_DF, ridge_df=RIDGE_DF, standardizer=SCALER,
#                         summary_stats=summary_tournament,tournament_sim=tidx,probability_dict=prob_mapper)
#             if prob_mapper[game.game_string][team1.team_name] ==0:
#                 game.vary_stats()
#                 game.simulate_game()
#             else:
#                 game.short_game()
#             # game.vary_stats()
#             # game.simulate_game()
#             game.game_summary()

#     # print(final_four)
#     current_round = 'final four'
#     print(f"\n\n starting {current_round}\n\n")
#     ff_matchup = {'Game_1':[final_four['East'],final_four['West']],
#                 'Game_2':[final_four['Midwest'],final_four['South']]}

#     championship = {'Game_1':[]}

#     for game, teams in ff_matchup.items():
#         game_loc = 'Phoenix, AZ'
#         team1 = team(team_name=teams[0].team_name, team_seed=teams[0].team_seed, region=teams[0].region)
#         team2 = team(team_name=teams[1].team_name, team_seed=teams[1].team_seed, region=teams[1].region)
#         game = matchup(team1, team2, game_location=game_loc, num_games=NUM_GAMES, game_id=int(game.split('_')[-1]), game_round=championship, next_round = 'championship', current_round=current_round,
#                     vary_params=VARY_PARAMS, model_params=NN_PARAMS, season_data=MODEL_DF, ridge_df=RIDGE_DF, standardizer=SCALER,
#                     summary_stats=summary_tournament,tournament_sim=tidx,probability_dict=prob_mapper)
#         if prob_mapper[game.game_string][team1.team_name] ==0:
#             game.vary_stats()
#             game.simulate_game()
#         else:
#             game.short_game()
#         # game.vary_stats()
#         # game.simulate_game()
#         game.game_summary()
#     current_round = 'championship'
#     print(f"\n\n starting {current_round}\n\n")
#     for game,teams in championship.items():
#         game_loc = 'Phoenix, AZ'
#         team1 = team(team_name=teams[0].team_name, team_seed=teams[0].team_seed, region=teams[0].region)
#         team2 = team(team_name=teams[1].team_name, team_seed=teams[1].team_seed, region=teams[1].region)
#         game = matchup(team1, team2, game_location=game_loc, num_games=NUM_GAMES, game_id=int(game.split('_')[-1]), game_round=None, next_round = None, current_round=current_round,
#                     vary_params=VARY_PARAMS, model_params=NN_PARAMS, season_data=MODEL_DF, ridge_df=RIDGE_DF, standardizer=SCALER,
#                     summary_stats=summary_tournament,tournament_sim=tidx,probability_dict=prob_mapper)
#         game.vary_stats()
#         game.simulate_game()
#         game.game_summary()

#         print(f"\n\n national champion is: {game.game_winner}!\n\n")
    
#     print(f"starting new tournament\n\n")
#     print(f"%complete: {(tidx/NUM_TOURNAMENTS)*100:0.2f}%")
#     print(summary_tournament)
#     t2 = time.time()
#     print(f"tournament {tidx} complete in {t2-t1:0.2f} seconds")

# summary_tournament = summary_tournament/NUM_TOURNAMENTS
# print(summary_tournament)

# summary_tournament.to_excel('2024_tournament_summary.xlsx')
# print('complete!')

# round_32 = {'East':{'Game_1':[],
#                     'Game_2':[],
#                     'Game_3':[],
#                     'Game_4':[]},
#             'West':{'Game_1':[],
#                     'Game_2':[],
#                     'Game_3':[],
#                     'Game_4':[]},
#             'Midwest':{'Game_1':[],
#                     'Game_2':[],
#                     'Game_3':[],
#                     'Game_4':[]},
#             'South':{'Game_1':[],
#                     'Game_2':[],
#                     'Game_3':[],
#                     'Game_4':[]
#                     }
#             }
# final_four = {'East':None,
#             'West':None,
#             'Midwest':None,
#             'South':None}

# # for region in regions:
# #         if region not in good_regions:
# #             continue
# #         # if region !='East':
# #         #     continue
# #         current_round = 'round of 64'
# #         # print(f"\n\n starting {current_round}\n\n")
# #         for game in first_round:
# #             # fig, ax = plt.subplots()
# #             sdf = df[(df['Game'] == game) & (df['Region']==region)]
# #             game_loc = sdf['Location'].values[0]
# #             game_id = sdf['Game'].values[0]
# #             team1 = team(team_name=sdf['Team'].values[0], team_seed=sdf['Seed'].values[0], region=region)
# #             team2 = team(team_name=sdf['Team'].values[1], team_seed=sdf['Seed'].values[1], region=region)
# #             game = matchup(team1, team2, game_location=game_loc, num_games=NUM_GAMES, game_id=game_id, game_round=round_32, next_round = 'round of 32',current_round = current_round,
# #                         region=region, vary_params=VARY_PARAMS, model_params=NN_PARAMS, season_data=MODEL_DF, ridge_df=RIDGE_DF, standardizer=SCALER,
# #                         summary_stats=summary_tournament,tournament_sim=1,probability_dict=prob_mapper)
# #             teams = [team1, team2]
# #             if 'TCU Horned Frogs' in [team.team_name for team in teams]:
# #                 game.vary_stats()
# #                 game.simulate_game()
# #                 game.game_summary()

# current_round = 'final four'
# # game_loc = 'Charlotte, NC'
# line = 146.5
# region = None
# team1 = team(team_name='UConn Huskies', team_seed = 1, region=region)
# team2 = team(team_name='Alabama Crimson Tide', team_seed = 4, region=region)
# game_loc = tourney_locs[current_round]['Game_1']

# game = matchup(team1, team2, game_location=game_loc, num_games=NUM_GAMES, game_id=1, game_round=final_four, next_round = 'championship',current_round = current_round,
#             region=region, vary_params=VARY_PARAMS, model_params=NN_PARAMS, season_data=MODEL_DF, ridge_df=RIDGE_DF, standardizer=SCALER,
#             summary_stats=summary_tournament,tournament_sim=1,probability_dict=prob_mapper, line = line)
# game.vary_stats()
# game.simulate_game()
# game.game_summary()