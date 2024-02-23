###########################################
####### --------- IMPORTS --------- #######
###########################################

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time

# https://www.ncsasports.org/division-1-colleges 

###########################################
####### ------- FUNCTIONS --------- #######
###########################################

class season_data:

    def __init__(self, season, data):
        self.season = season
        self.data = data
        
    
    def split_columns(self):
        self.data[['FG_made', 'FG_attempted']] = self.data['FG'].str.split('-', expand=True).astype(int)
        self.data[['3PT_made', '3PT_attempted']] = self.data['3PT'].str.split('-', expand=True).astype(int)
        self.data['2PT_made'] = self.data['FG_made'] - self.data['3PT_made'].astype(int)
        self.data['2PT_attempted'] = self.data['FG_attempted'] - self.data['3PT_attempted'].astype(int)
        self.data[['FT_made', 'FT_attempted']] = self.data['FT'].str.split('-', expand=True).astype(int)
        self.data['Team_score'] = self.data.apply(lambda row: row['2PT_made']*2 + row['3PT_made']*3 + row['FT_made'], axis=1)

    def win_game(self): 
        self.data['Win'] = 0
        self.data['Loss'] = 0
        game_ids = self.data['GameID'].unique()
        for idx, game_id in enumerate(game_ids):
            # if idx < 4978:
            #     continue
            game = self.data[self.data['GameID'] == game_id]
            # total_score = game['Team_score'].sum()
            # over_under = float(game['Over Under'].iloc[0].split(':')[-1].strip())
            # line = game['Betting Line'].iloc[0].split(' ')[-1]
            
            # score_diff = np.abs(game['Team_score'].iloc[0] - game['Team_score'].iloc[1])
            # print(f"betting line:{game['Betting Line'].iloc[0]}\nactual diff: {score_diff}")

            if game['Team_score'].iloc[0] > game['Team_score'].iloc[1]:
                self.data.iloc[game.index[0], self.data.columns.get_loc('Win')] = 1
                self.data.iloc[game.index[0], self.data.columns.get_loc('Loss')] = 0
                self.data.iloc[game.index[1], self.data.columns.get_loc('Loss')] = 1
                self.data.iloc[game.index[1], self.data.columns.get_loc('Win')] = 0

            else:
                self.data.iloc[game.index[1], self.data.columns.get_loc('Win')] = 1
                self.data.iloc[game.index[1], self.data.columns.get_loc('Loss')] = 0
                self.data.iloc[game.index[0], self.data.columns.get_loc('Loss')] = 1
                self.data.iloc[game.index[0], self.data.columns.get_loc('Win')] = 0



    def possesions(self):
        self.data['Possessions'] = 0
        self.data['Team_Possessions'] = 0
        self.data['Raw_Off_Eff'] = 0
        self.data['Raw_Def_Eff'] = 0
        self.data['Pace'] = 0

        self.data['Possessions'] = self.data['Possessions'].astype(float)
        self.data['Team_Possessions'] = self.data['Team_Possessions'].astype(float)
        self.data['Raw_Off_Eff'] = self.data['Raw_Off_Eff'].astype(float)
        self.data['Raw_Def_Eff'] = self.data['Raw_Def_Eff'].astype(float)
        self.data['Pace'] = self.data['Pace'].astype(float)

        for game_id in self.data['GameID'].unique():
            game = self.data[self.data['GameID'] == game_id]
            
            #get the number of possesions for each team and then average them for the game 
            avg_possessions=game.apply(lambda row: row['FG_attempted'] + row['FT_attempted']*0.44 + row['Total Turnovers'] - row['Offensive Rebounds'], axis=1).mean()
            game.loc[:,'Team_Possessions'] = game.apply(lambda row: row['FG_attempted'] + row['FT_attempted']*0.44 + row['Total Turnovers'] - row['Offensive Rebounds'], axis=1)
            
            game.loc[:,'Raw_Off_Eff'] = game.apply(lambda row: row['Team_score']/avg_possessions, axis=1)
            self.data.iloc[game.index, self.data.columns.get_loc('Possessions')] = avg_possessions
            self.data.iloc[game.index, self.data.columns.get_loc('Raw_Off_Eff')] = game['Raw_Off_Eff']
            #add in the defensive efficiency for each team; since it is the raw value, raw defensive efficiency is the same as raw offensive efficiency for the opposing team
            self.data.iloc[game.index[0], self.data.columns.get_loc('Raw_Def_Eff')] = self.data.iloc[game.index[1], self.data.columns.get_loc('Raw_Off_Eff')]
            self.data.iloc[game.index[1], self.data.columns.get_loc('Raw_Def_Eff')] = self.data.iloc[game.index[0], self.data.columns.get_loc('Raw_Off_Eff')]
            #calculate the pace of the game
            
            self.data.iloc[game.index[0], self.data.columns.get_loc('Team_Possessions')] = game['Team_Possessions'].iloc[0]
            self.data.iloc[game.index[1], self.data.columns.get_loc('Team_Possessions')] = game['Team_Possessions'].iloc[1]
            self.data.iloc[game.index[0], self.data.columns.get_loc('Pace')] = game['Team_Possessions'].iloc[0]/40
            self.data.iloc[game.index[1], self.data.columns.get_loc('Pace')] = game['Team_Possessions'].iloc[1]/40

        
    def four_factors(self):
        #calculate the four factors for each team
        #1. Effective Field Goal Percentage
        #2. Turnover Percentage
        #3. Offensive Rebound Percentage
        #4. Free Throw Rate
        self.data['off_eFG'] = 0
        self.data['off_TOV'] = 0
        self.data['off_ORB'] = 0
        self.data['off_FTR'] = 0

        self.data['def_eFG'] = 0
        self.data['def_TOV'] = 0
        self.data['def_ORB'] = 0
        self.data['def_FTR'] = 0

        self.data['off_eFG'] = self.data['off_eFG'].astype(float)
        self.data['off_TOV'] = self.data['off_TOV'].astype(float)
        self.data['off_ORB'] = self.data['off_ORB'].astype(float)
        self.data['off_FTR'] = self.data['off_FTR'].astype(float)

        self.data['def_eFG'] = self.data['def_eFG'].astype(float)
        self.data['def_TOV'] = self.data['def_TOV'].astype(float)
        self.data['def_ORB'] = self.data['def_ORB'].astype(float)
        self.data['def_FTR'] = self.data['def_FTR'].astype(float)


        for idx, game_id in enumerate(self.data['GameID'].unique()):
            
            game = self.data[self.data['GameID'] == game_id]
            game = game.copy()
            #calculate the effective field goal percentage for each team
            game.loc[:,'eFG'] = game.apply(lambda row: (row['FG_made'] + 0.5*row['3PT_made'])/row['FG_attempted'], axis=1)
            #calculate the turnover percentage for each team
            game.loc[:,'TOV'] = game.apply(lambda row: row['Total Turnovers']/(row['FG_attempted'] + 0.44*row['FT_attempted'] + row['Total Turnovers']), axis=1)
            
            self.data.iloc[game.index[0], self.data.columns.get_loc('off_eFG')] = game['eFG'].iloc[0]
            self.data.iloc[game.index[1], self.data.columns.get_loc('off_eFG')] = game['eFG'].iloc[1]
            self.data.iloc[game.index[0], self.data.columns.get_loc('def_eFG')] = game['eFG'].iloc[1]
            self.data.iloc[game.index[1], self.data.columns.get_loc('def_eFG')] = game['eFG'].iloc[0]

            self.data.iloc[game.index[0], self.data.columns.get_loc('off_TOV')] = game['TOV'].iloc[0]
            self.data.iloc[game.index[1], self.data.columns.get_loc('off_TOV')] = game['TOV'].iloc[1]
            self.data.iloc[game.index[0], self.data.columns.get_loc('def_TOV')] = game['TOV'].iloc[1]
            self.data.iloc[game.index[1], self.data.columns.get_loc('def_TOV')] = game['TOV'].iloc[0]

            self.data.iloc[game.index[0], self.data.columns.get_loc('off_ORB')] = game['Offensive Rebounds'].iloc[0]/(game['Offensive Rebounds'].iloc[0] + game['Defensive Rebounds'].iloc[1])
            self.data.iloc[game.index[1], self.data.columns.get_loc('off_ORB')] = game['Offensive Rebounds'].iloc[1]/(game['Offensive Rebounds'].iloc[1] + game['Defensive Rebounds'].iloc[0])
            
            self.data.iloc[game.index[0], self.data.columns.get_loc('def_ORB')] = game['Defensive Rebounds'].iloc[0]/(game['Offensive Rebounds'].iloc[1] + game['Defensive Rebounds'].iloc[0])
            self.data.iloc[game.index[1], self.data.columns.get_loc('def_ORB')] = game['Defensive Rebounds'].iloc[1]/(game['Offensive Rebounds'].iloc[0] + game['Defensive Rebounds'].iloc[1])

            #calculate free throw rate for each team
            game.loc[:,'FTR'] = game.apply(lambda row: row['FT_attempted']/row['FG_attempted'], axis=1)
            self.data.iloc[game.index[0], self.data.columns.get_loc('off_FTR')] = game['FTR'].iloc[0]
            self.data.iloc[game.index[1], self.data.columns.get_loc('off_FTR')] = game['FTR'].iloc[1]
            self.data.iloc[game.index[0], self.data.columns.get_loc('def_FTR')] = game['FTR'].iloc[1]
            self.data.iloc[game.index[1], self.data.columns.get_loc('def_FTR')] = game['FTR'].iloc[0]


    
    def home_away(self):

        def get_coordinates(city_name):
            geolocator = Nominatim(user_agent="city_distance_calculator")
            location = geolocator.geocode(city_name, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            else:
                return None
        
        def calculate_distance(city1, city2):
            coordinates1 = get_coordinates(city1)
            coordinates2 = get_coordinates(city2)
        
            if coordinates1 and coordinates2:
                distance = geodesic(coordinates1, coordinates2).miles
                return distance
            else:
                return None
            
        self.data['Home'] = 0
        self.data['Away'] = 0
        self.data['Neutral'] = 0
        self.data['Distance_Traveled'] = 0
        game_ids = self.data['GameID'].unique()
        t1 = time.time()
        for idx, game_id in enumerate(game_ids):
            game = self.data[self.data['GameID'] == game_id]
            team1 = game['Team'].iloc[0]
            team2 = game['Team'].iloc[1]
            game_location = game['State'].iloc[0]
            college_location1 = ncaa_team_info[ncaa_team_info['ESPN_Name'] == team1]['address'].values[0]
            college_location2 = ncaa_team_info[ncaa_team_info['ESPN_Name'] == team2]['address'].values[0]
            distance1 = calculate_distance(game_location, college_location1)
            distance2 = calculate_distance(game_location, college_location2)
            
            if distance1 == 0:
                self.data.iloc[game.index[0], self.data.columns.get_loc('Home')] = 1
                self.data.iloc[game.index[1], self.data.columns.get_loc('Away')] = 1
                self.data.iloc[game.index[0], self.data.columns.get_loc('Distance_Traveled')] = 0
                self.data.iloc[game.index[1], self.data.columns.get_loc('Distance_Traveled')] = distance2
            elif distance2 == 0:
                self.data.iloc[game.index[1], self.data.columns.get_loc('Home')] = 1
                self.data.iloc[game.index[0], self.data.columns.get_loc('Away')] = 1
                self.data.iloc[game.index[1], self.data.columns.get_loc('Distance_Traveled')] = 0
                self.data.iloc[game.index[0], self.data.columns.get_loc('Distance_Traveled')] = distance1
            else:
                self.data.iloc[game.index[0], self.data.columns.get_loc('Neutral')] = 1
                self.data.iloc[game.index[1], self.data.columns.get_loc('Neutral')] = 1
                self.data.iloc[game.index[0], self.data.columns.get_loc('Distance_Traveled')] = distance1
                self.data.iloc[game.index[1], self.data.columns.get_loc('Distance_Traveled')] = distance2
        t2 = time.time()
        if t2-t1 > 60 and t2-t1 < 3600:
            print(f'time to calculate distances: {(t2-t1)/60} minutes')
        elif t2-t1 > 3600:
            print(f'time to calculate distances: {(t2-t1)/3600} hours')
            
        

###########################################
####### ------- MAIN CODE ------- #########
###########################################


bp = r'C:\Users\zhostetl\Documents\11_CBB\01_rawdata'

seasons = ['2022_2023', '2023_2024']
data_dict = {}
for season in seasons:
    season_file = f"{season}_team_stats.xlsx"
    season_path = os.path.join(bp, season_file)
    data_dict[season] = pd.read_excel(season_path)

soi = '2022_2023'

ncaa_team_info =  pd.read_csv(os.path.join(bp, 'ncaa_info.csv'))


###############################################
### ---- MANUAL DATA CLEANING CHECKS ----- ####
###############################################

# print(data_dict[soi].columns)

# for column in data_dict[soi].columns:
#     print(f"NaNs in {column}: {data_dict[soi][column].isna().sum()} {data_dict[soi][data_dict[soi][column].isna()].index}")

# print(data_dict[soi].isna().sum())

###############################################
### ---- Calculate relevant metrics ------ ####
###############################################

season = season_data(soi, data_dict[soi])

excluded_teams = []
missing = 0 
row_drop = []
for game in season.data['GameID'].unique():
    game_data = season.data[season.data['GameID'] == game]
    for team in game_data['Team']:
        if team not in ncaa_team_info['ESPN_Name'].values:
            row_drop.append(game)
            if team not in excluded_teams:
                excluded_teams.append(team)
            missing += 1

# print(season.data.dtypes)
season.data = season.data[~season.data['GameID'].isin(row_drop)].reset_index(drop=True)

# print(season.data.dtypes)

# season.data.to_excel('cleaned_data.xlsx', index = False)
# for idx, row in season.data.iterrows():
#     print(row['3PT'].split('-'))
      
season.split_columns()

season.win_game()
season.possesions()
season.four_factors()
season.home_away()

print(season.data.head(15))
season.data.to_excel('cleaned_data.xlsx', index = False)
print('completed')

