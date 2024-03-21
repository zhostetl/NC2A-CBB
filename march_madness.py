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
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time

def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="city_distance_calculator")
    location = geolocator.geocode(city_name, timeout=15)
    if location:
        return (location.latitude, location.longitude)
    
def calculate_distance(city1, city2):
    coordinates1 = get_coordinates(city1)
    coordinates2 = get_coordinates(city2)

    if coordinates1 and coordinates2:
        distance = geodesic(coordinates1, coordinates2).miles
        return distance
    else:
        return None

ncaa_team_info =  pd.read_csv(r'ncaa_info.csv')

data_file = r'2024_ncaa_tournament_seeds.xlsx'

model_data = r'.\03_modelfitting\2023_2024_adjusted_data_EM.xlsx'

MODEL_DF = pd.read_excel(model_data)

# off_check = model_df.groupby('Team')['adj_Raw_Off_Eff'].mean().sort_values(ascending = False).head(20)
# def_check = model_df.groupby('Team')['adj_Raw_Def_Eff'].mean().sort_values(ascending = True).head(20)
# print(f"adjusted off eff: {off_check}")
# print(f"adjusted def eff: {def_check}")

df = pd.read_excel(data_file)

all_teams = df['Team'].to_list()

regions = ['East', 'Midwest', 'South', 'West']
first_round = [1,2,3,4,5,6,7,8]


# Load the scaler
SCALER = joblib.load(r'.\03_modelfitting\scaler_EM.pkl')

#this should be updated to just be the adjusted values since the raw values lead to inaccurate predictions
# VARY_PARAMS = ['Raw_Off_Eff','off_eFG','off_TOV','off_ORB','off_FTR',
#             #    'Raw_Def_Eff','def_eFG','def_TOV','def_ORB','def_FTR',
#                'Pace','Total Turnovers','Fouls','FG_attempted','3PT_attempted','Team_Possessions']
VARY_PARAMS = ['adj_Raw_Off_Eff','adj_off_eFG','adj_off_TOV','adj_off_ORB','adj_off_FTR',
            #    'adj_Raw_Def_Eff','adj_def_eFG','adj_def_TOV','adj_def_ORB','adj_def_FTR',
               'Pace','Total Turnovers','Fouls','FG_attempted','3PT_attempted','Team_Possessions','adj_EM']

NN_PARAMS = ['Distance_Traveled',
          'adj_Raw_Off_Eff','adj_off_eFG','adj_off_TOV','adj_off_ORB','adj_off_FTR',
          'adj_Raw_Def_Eff','adj_def_eFG','adj_def_TOV','adj_def_ORB','adj_def_FTR',
          'Pace','Total Turnovers','Fouls','FG_attempted','3PT_attempted',
          'Team_Possessions','Home','Away','adj_EM']

NUM_GAMES = 500

test_df = pd.read_excel(r'.\03_modelfitting\2023_2024_adjusted_data_EM.xlsx',index_col=0)

RIDGE_DF = pd.read_excel(r'.\03_modelfitting\2023_2024_adjusted_team_data.xlsx')

X = test_df[NN_PARAMS].values
X = SCALER.transform(X)

#reload the model 
# add_layers = 40 #normal train 
add_layers = 60 #deep train
ANNreg = nn.Sequential(
    nn.Linear(X.shape[1], add_layers),
    # nn.ReLU(),
    nn.Sigmoid(),
    nn.Linear(add_layers, add_layers),
    # nn.ReLU(),
    nn.Sigmoid(),
    nn.Linear(add_layers, add_layers),
    nn.Sigmoid(),
    nn.Linear(add_layers, 1)
)

ANNreg.load_state_dict(torch.load(r'.\03_modelfitting\ANNreg_EM_train.pth'))


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
                  vary_params = None, model_params=None, season_data=None, ridge_df=None, standardizer=None,summary_stats=None,tournament_sim = None,probability_dict=None):
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

        self.game_string = f"{self.team1.team_name} vs {self.team2.team_name}"
        
        if self.game_string not in probability_dict:
            probability_dict[self.game_string] = {self.team1.team_name:0, f"{self.team1.team_name}_score":0,
                                            self.team2.team_name:0, f"{self.team2.team_name}_score":0}
        
    
    def vary_stats(self):
        teams = [self.team1, self.team2]
        self.team_data = {}
        for team in teams:
            if self.game_location not in distance_mapper[team.team_name]:
                team_dist = calculate_distance(self.game_location, ncaa_team_info[ncaa_team_info['ESPN_Name'] == team.team_name]['address'].values[0])
                distance_mapper[team.team_name][self.game_location] = team_dist
                # print('used distance method')
            else:
                team_dist = distance_mapper[team.team_name][self.game_location]
                # print('used dictionary method')
            # team_dist = calculate_distance(self.game_location, ncaa_team_info[ncaa_team_info['ESPN_Name'] == team.team_name]['address'].values[0])
            
            team_df = self.season_data[self.season_data['Team'] == team.team_name][self.vary_params]
            means = team_df.mean()
            stds = team_df.std()
            random_samples = np.empty((self.num_games, len(self.vary_params)))
            for i in range(self.num_games):
                random_samples[i] = np.random.normal(means, stds)
            random_df = pd.DataFrame(random_samples, columns = self.vary_params)
            random_df['Distance_Traveled'] = team_dist
            random_df['Home'] = 0
            random_df['Away'] = 0
            self.team_data[team.team_name] = random_df
        for idx, row in self.team_data[self.team1.team_name].iterrows():
           self.team_data[self.team1.team_name].loc[idx, 'adj_Raw_Def_Eff'] = self.team_data[self.team2.team_name].loc[idx, 'adj_Raw_Off_Eff']
           self.team_data[self.team1.team_name].loc[idx, 'adj_def_eFG'] = self.team_data[self.team2.team_name].loc[idx, 'adj_off_eFG']
           self.team_data[self.team1.team_name].loc[idx, 'adj_def_TOV'] = self.team_data[self.team2.team_name].loc[idx, 'adj_off_TOV']
           self.team_data[self.team1.team_name].loc[idx, 'adj_def_ORB'] = self.team_data[self.team2.team_name].loc[idx, 'adj_off_ORB']
           self.team_data[self.team1.team_name].loc[idx, 'adj_def_FTR'] = self.team_data[self.team2.team_name].loc[idx, 'adj_off_FTR']

           self.team_data[self.team2.team_name].loc[idx, 'adj_Raw_Def_Eff'] = self.team_data[self.team1.team_name].loc[idx, 'adj_Raw_Off_Eff']
           self.team_data[self.team2.team_name].loc[idx, 'adj_def_eFG'] = self.team_data[self.team1.team_name].loc[idx, 'adj_off_eFG']
           self.team_data[self.team2.team_name].loc[idx, 'adj_def_TOV'] = self.team_data[self.team1.team_name].loc[idx, 'adj_off_TOV']
           self.team_data[self.team2.team_name].loc[idx, 'adj_def_ORB'] = self.team_data[self.team1.team_name].loc[idx, 'adj_off_ORB']
           self.team_data[self.team2.team_name].loc[idx, 'adj_def_FTR'] = self.team_data[self.team1.team_name].loc[idx, 'adj_off_FTR']
        
        # print(self.team_data[self.team1.team_name].head())
        # print(self.team_data[self.team2.team_name].head())

        # print(f"\n\ngetting adjusted values \n\n")
        #get the adjusted values 
        # for idx, row in self.team_data[self.team1.team_name].iterrows():
        #     self.team_data[self.team1.team_name].loc[idx, 'adj_Raw_Off_Eff'] = row['Raw_Off_Eff']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team2.team_name}"]['Raw_Off_Eff'].values[0]
        #     self.team_data[self.team1.team_name].loc[idx, 'adj_off_eFG'] = row['off_eFG']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team2.team_name}"]['off_eFG'].values[0]
        #     self.team_data[self.team1.team_name].loc[idx, 'adj_off_TOV'] = row['off_TOV']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team2.team_name}"]['off_TOV'].values[0]
        #     self.team_data[self.team1.team_name].loc[idx, 'adj_off_ORB'] = row['off_ORB']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team2.team_name}"]['off_ORB'].values[0]
        #     self.team_data[self.team1.team_name].loc[idx, 'adj_off_FTR'] = row['off_FTR']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team2.team_name}"]['off_FTR'].values[0]
        #     self.team_data[self.team1.team_name].loc[idx, 'adj_Raw_Def_Eff'] = row['Raw_Def_Eff']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team2.team_name}"]['Raw_Def_Eff'].values[0]
        #     self.team_data[self.team1.team_name].loc[idx, 'adj_def_eFG'] = row['def_eFG']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team2.team_name}"]['def_eFG'].values[0]
        #     self.team_data[self.team1.team_name].loc[idx, 'adj_def_TOV'] = row['def_TOV']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team2.team_name}"]['def_TOV'].values[0]
        #     self.team_data[self.team1.team_name].loc[idx, 'adj_def_ORB'] = row['def_ORB']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team2.team_name}"]['def_ORB'].values[0]
        #     self.team_data[self.team1.team_name].loc[idx, 'adj_def_FTR'] = row['def_FTR']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team2.team_name}"]['def_FTR'].values[0]

        #     self.team_data[self.team2.team_name].loc[idx, 'adj_Raw_Off_Eff'] = row['Raw_Off_Eff']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team1.team_name}"]['Raw_Off_Eff'].values[0]
        #     self.team_data[self.team2.team_name].loc[idx, 'adj_off_eFG'] = row['off_eFG']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team1.team_name}"]['off_eFG'].values[0]
        #     self.team_data[self.team2.team_name].loc[idx, 'adj_off_TOV'] = row['off_TOV']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team1.team_name}"]['off_TOV'].values[0]
        #     self.team_data[self.team2.team_name].loc[idx, 'adj_off_ORB'] = row['off_ORB']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team1.team_name}"]['off_ORB'].values[0]
        #     self.team_data[self.team2.team_name].loc[idx, 'adj_off_FTR'] = row['off_FTR']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team1.team_name}"]['off_FTR'].values[0]
        #     self.team_data[self.team2.team_name].loc[idx, 'adj_Raw_Def_Eff'] = row['Raw_Def_Eff']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team1.team_name}"]['Raw_Def_Eff'].values[0]
        #     self.team_data[self.team2.team_name].loc[idx, 'adj_def_eFG'] = row['def_eFG']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team1.team_name}"]['def_eFG'].values[0]
        #     self.team_data[self.team2.team_name].loc[idx, 'adj_def_TOV'] = row['def_TOV']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team1.team_name}"]['def_TOV'].values[0]
        #     self.team_data[self.team2.team_name].loc[idx, 'adj_def_ORB'] = row['def_ORB']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team1.team_name}"]['def_ORB'].values[0]
        #     self.team_data[self.team2.team_name].loc[idx, 'adj_def_FTR'] = row['def_FTR']-self.ridge_df[self.ridge_df['coef_name'] == f"Opponent_{self.team1.team_name}"]['def_FTR'].values[0]

        # print(self.team_data[self.team1.team_name][self.model_params].head())
        # print(self.team_data[self.team2.team_name][self.model_params].head())
    
    def simulate_game(self):
        team1_scores = np.array([])
        team2_scores = np.array([])
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

            if team1_score > team2_score:
                self.team1_wins += 1
            else:
                self.team2_wins += 1
        
        self.team1_winprob = self.team1_wins/self.num_games
        self.team2_winprob = self.team2_wins/self.num_games

        self.team1_score = team1_scores.mean()
        self.team2_score = team2_scores.mean()

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
              f"score: {self.team1_score:0.2f} vs {self.team2_score:0.2f}\n"
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

distance_mapper = {team:{} for team in all_teams}

prob_mapper = {}

tourney_locs = {'round of 32':{'East':{'Game_1':'Brooklyn, NY',
                                       'Game_2':'Spokane, WA',
                                       'Game_3':'Omaha, NE',
                                        'Game_4':'Omaha, NE'
                                        },
                               'West':{'Game_1':'Charlotte, NC',
                                        'Game_2':'Spokane, WA',
                                        'Game_3':'Memphis, TN',
                                        'Game_4':'Salt Lake City, UT'
                                        },
                                'Midwest':{'Game_1':'Indianapolis, IN',
                                        'Game_2':'Salt Lake City, UT',
                                        'Game_3':'Pittsburgh, PA',
                                        'Game_4':'Charlotte, NC'
                                        },
                                'South':{'Game_1':'Charlotte, NC',
                                        'Game_2':'Brooklyn, NY',
                                        'Game_3':'Tampa, FL',
                                        'Game_4':'Tampa, FL'
                                        }
                                 },
                'sweet 16':{'East':{'Game_1':'Boston, MA',
                                    'Game_2':'Boston, MA',
                                        },
                            'West':{'Game_1':'Los Angeles, CA',
                                    'Game_2':'Los Angeles, CA',
                                    },
                            'Midwest':{'Game_1':'Detroit, MI',
                                       'Game_2':'Detroit, MI',
                                    },
                            'South':{'Game_1':'Dallas, TX',
                                     'Game_2':'Dallas, TX',
                                    }
                            },
                'Elite 8':{'East':{'Game_1':'Boston, MA'},
                            'West':{'Game_1':'Los Angeles, CA'},
                            'Midwest':{'Game_1':'Detroit, MI'},
                            'South':{'Game_1':'Dallas, TX'}
                            },
                'final four':{'Game_1':'Phoenix, AZ',
                              'Game_2':'Phoenix, AZ'
                            },
                'championship':{'Game_1':'Phoenix, AZ'}
                               }

game_rounds = ['round of 64','round of 32','sweet 16','Elite 8','final four','championship']
summary_tournament = pd.DataFrame(columns = game_rounds,index=all_teams,data=0)
good_regions = ['East','West','Midwest','South']
NUM_TOURNAMENTS = 50
# for tidx in range(1,NUM_TOURNAMENTS+1):
#     t1 = time.time()
#     round_32 = {'East':{'Game_1':[],
#                         'Game_2':[],
#                         'Game_3':[],
#                         'Game_4':[]},
#                 'West':{'Game_1':[],
#                         'Game_2':[],
#                         'Game_3':[],
#                         'Game_4':[]},
#                 'Midwest':{'Game_1':[],
#                         'Game_2':[],
#                         'Game_3':[],
#                         'Game_4':[]},
#                 'South':{'Game_1':[],
#                         'Game_2':[],
#                         'Game_3':[],
#                         'Game_4':[]
#                         }
#                 }
#     good_regions = ['East','West','Midwest','South']
#     for region in regions:
#         if region not in good_regions:
#             continue
#         # if region !='East':
#         #     continue
#         current_round = 'round of 64'
#         print(f"\n\n starting {current_round}\n\n")
#         for game in first_round:
#             # fig, ax = plt.subplots()
#             sdf = df[(df['Game'] == game) & (df['Region']==region)]
#             game_loc = sdf['Location'].values[0]
#             game_id = sdf['Game'].values[0]
#             team1 = team(team_name=sdf['Team'].values[0], team_seed=sdf['Seed'].values[0], region=region)
#             team2 = team(team_name=sdf['Team'].values[1], team_seed=sdf['Seed'].values[1], region=region)
#             game = matchup(team1, team2, game_location=game_loc, num_games=NUM_GAMES, game_id=game_id, game_round=round_32, next_round = 'round of 32',current_round = current_round,
#                         region=region, vary_params=VARY_PARAMS, model_params=NN_PARAMS, season_data=MODEL_DF, ridge_df=RIDGE_DF, standardizer=SCALER,
#                         summary_stats=summary_tournament,tournament_sim=tidx,probability_dict=prob_mapper)
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

for region in regions:
        if region not in good_regions:
            continue
        # if region !='East':
        #     continue
        current_round = 'round of 64'
        print(f"\n\n starting {current_round}\n\n")
        for game in first_round:
            # fig, ax = plt.subplots()
            sdf = df[(df['Game'] == game) & (df['Region']==region)]
            game_loc = sdf['Location'].values[0]
            game_id = sdf['Game'].values[0]
            team1 = team(team_name=sdf['Team'].values[0], team_seed=sdf['Seed'].values[0], region=region)
            team2 = team(team_name=sdf['Team'].values[1], team_seed=sdf['Seed'].values[1], region=region)
            game = matchup(team1, team2, game_location=game_loc, num_games=NUM_GAMES, game_id=game_id, game_round=round_32, next_round = 'round of 32',current_round = current_round,
                        region=region, vary_params=VARY_PARAMS, model_params=NN_PARAMS, season_data=MODEL_DF, ridge_df=RIDGE_DF, standardizer=SCALER,
                        summary_stats=summary_tournament,tournament_sim=1,probability_dict=prob_mapper)
            
            game.vary_stats()
            game.simulate_game()
            game.game_summary()

            # if prob_mapper[game.game_string][team1.team_name] ==0:
            #     game.vary_stats()
            # # ax.hist(game.team_data[team1.team_name]['adj_EM'], bins = 50, alpha = 0.5, label = team1.team_name)
            # # ax.hist(game.team_data[team2.team_name]['adj_EM'], bins = 50, alpha = 0.5, label = team2.team_name)
            # # ax.legend()
            # # ax.set_title(f"adjusted offensive efficiency for {team1.team_name} and {team2.team_name}")
            #     game.simulate_game()
            # else: 
            #     game.short_game()
            # game.game_summary()