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

#define the tournament matchups 
big_12_matchups = {'round_1':{'team1': ['Oklahoma State Cowboys','West Virginia Mountaineers'],
                            'team2':['UCF Knights','Cincinnati Bearcats'],
                            'round_winners':[]},
                'round_2':{'team1':['Oklahoma Sooners','Kansas State Wildcats',],
                           'team2':['BYU Cougars','TCU Horned Frogs','Texas Longhorns','Kansas Jayhawks'],
                           'round_winners':[]},
                'round_3':{'team1':[],
                           'team2':['Texas Tech Red Raiders','Houston Cougars','Iowa State Cyclones','Baylor Bears'],
                           'round_winners':[]},
                'round_4':{'team1':[],
                           'team2':[],
                           'round_winners':[]},
                'round_5':{'team1':[],
                           'team2':[],
                           'round_winners':[]},
                'round_6':{'champion':[]}
                }

acc_matchups = {'round_1':{'team1': ['Notre Dame Fighting Irish', 'NC State Wolfpack','Boston College Eagles'],
                            'team2':['Georgia Tech Yellow Jackets','Louisville Cardinals','Miami Hurricanes'],
                            'round_winners':[]},
                'round_2':{'team1':['Virginia Tech Hokies'],
                           'team2':['Florida State Seminoles','Wake Forest Demon Deacons','Syracuse Orange','Clemson Tigers'],
                           'round_winners':[]},
                'round_3':{'team1':[],
                           'team2':['North Carolina Tar Heels','Pittsburgh Panthers','Duke Blue Devils','Virginia Cavaliers'],
                           'round_winners':[]},
                'round_4':{'team1':[],
                           'team2':[],
                           'round_winners':[]},
                'round_5':{'team1':[],
                           'team2':[],
                           'round_winners':[]},
                'round_6':{'champion':[]}
                }

# Load the scaler
scaler = joblib.load(r'03_modelfitting\scaler.pkl')

# Load the data
model_params = ['FG_attempted','FT_attempted','Total Turnovers','Offensive Rebounds','Team_score',
                'FG_made','3PT_made','Defensive Rebounds','Pace','Fouls','3PT_attempted']
nn_params = ['Distance_Traveled',
          'adj_Raw_Off_Eff','adj_off_eFG','adj_off_TOV','adj_off_ORB','adj_off_FTR',
          'adj_Raw_Def_Eff','adj_def_eFG','adj_def_TOV','adj_def_ORB','adj_def_FTR',
          'Pace','Total Turnovers','Fouls','FG_attempted','3PT_attempted',
          'Team_Possessions','Home','Away']

test_df = pd.read_excel(r'C:\Users\zhostetl\Documents\11_CBB\99_git\NC2A-CBB\03_modelfitting\2023_2024_adjusted_data.xlsx',index_col=0)

ridge_df = pd.read_excel(r'C:\Users\zhostetl\Documents\11_CBB\99_git\NC2A-CBB\03_modelfitting\2023_2024_adjusted_team_data.xlsx')

X = test_df[nn_params].values
X = scaler.transform(X)

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

ANNreg.load_state_dict(torch.load(r'03_modelfitting\ANNreg_deep_train.pth'))

X_input = torch.from_numpy(X.astype(np.float32))

class matchup():
    def __init__(self, dataframe, team1, team2, home_team, varied_params, model_params, number_of_games, input_standardizer):
        self.dataframe = dataframe
        self.team1 = team1
        self.team2 = team2
        self.home_team = home_team
        self.varied_params = varied_params
        self.model_params = model_params
        self.number_of_games = number_of_games
        self.input_standardizer = input_standardizer
        self.team1_wins = 0 
        self.team2_wins = 0

    def vary_stats(self):
        self.stats = {}
        teams = [self.team1, self.team2]
        for team in teams: 
            sdf = self.dataframe[self.dataframe['Team'] == team][self.varied_params]
            means = sdf.mean()
            stds = sdf.std()
            random_samples = np.empty((self.number_of_games, len(self.varied_params)))
            for i in range(self.number_of_games):
                random_samples[i] = np.random.normal(means, stds)
            self.stats[team] = pd.DataFrame(random_samples, columns = self.varied_params)

            self.stats[team].loc[:,'Team_Possessions'] = self.stats[team].apply(lambda row: row['FG_attempted'] + row['FT_attempted']*0.44 + row['Total Turnovers'] - row['Offensive Rebounds'], axis=1)
            self.stats[team].loc[:,'Raw_Off_Eff'] = self.stats[team].apply(lambda row: row['Team_score']/row['Team_Possessions'], axis=1)
            self.stats[team].loc[:,'off_eFG'] = self.stats[team].apply(lambda row: (row['FG_made'] + 0.5*row['3PT_made'])/row['FG_attempted'], axis=1)
            self.stats[team].loc[:,'off_TOV'] = self.stats[team].apply(lambda row: row['Total Turnovers']/(row['Team_Possessions'] + 0.44*row['FT_attempted'] + row['Total Turnovers']), axis=1)
            self.stats[team].loc[:,'off_FTR'] = self.stats[team].apply(lambda row: row['FT_attempted']/row['FG_attempted'], axis=1)
        
        for idx, row in self.stats[self.team1].iterrows():
            self.stats[self.team1].loc[idx,'Raw_Def_Eff'] = self.stats[self.team2].loc[idx,'Raw_Off_Eff']
            self.stats[self.team2].loc[idx,'Raw_Def_Eff'] = self.stats[self.team1].loc[idx,'Raw_Off_Eff']

            self.stats[self.team1].loc[idx,'def_eFG'] = self.stats[self.team2].loc[idx,'off_eFG']
            self.stats[self.team2].loc[idx,'def_eFG'] = self.stats[self.team1].loc[idx,'off_eFG']

            self.stats[self.team1].loc[idx,'def_TOV'] = self.stats[self.team2].loc[idx,'off_TOV']
            self.stats[self.team2].loc[idx,'def_TOV'] = self.stats[self.team1].loc[idx,'off_TOV']

            self.stats[self.team1].loc[idx,'off_ORB'] = row['Offensive Rebounds']/(row['Offensive Rebounds'] + self.stats[self.team2].loc[idx,'Defensive Rebounds'])
            self.stats[self.team2].loc[idx,'off_ORB'] = self.stats[self.team2].loc[idx,'Offensive Rebounds']/(self.stats[self.team2].loc[idx,'Offensive Rebounds'] + row['Defensive Rebounds'])
            
            self.stats[self.team1].loc[idx,'def_ORB'] = row['Defensive Rebounds']/(row['Defensive Rebounds'] + self.stats[self.team2].loc[idx,'Offensive Rebounds'])
            self.stats[self.team2].loc[idx,'def_ORB'] = self.stats[self.team2].loc[idx,'Defensive Rebounds']/(self.stats[self.team2].loc[idx,'Defensive Rebounds'] + row['Offensive Rebounds'])
            
            self.stats[self.team1].loc[idx,'def_FTR'] = self.stats[self.team2].loc[idx,'off_FTR']
            self.stats[self.team2].loc[idx,'def_FTR'] = self.stats[self.team1].loc[idx,'off_FTR']
        
        #adjusted stats = off_eff, off_eFG, off_TOV, off_ORB, off_FTR,
        # def_eff, def_eFG, def_TOV, def_ORB, def_FTR
        
        for team in teams:
            self.stats[team].loc[:,'adj_Raw_Off_Eff'] = self.stats[team].apply(lambda row: row['Raw_Off_Eff'] - ridge_df[ridge_df['coef_name'] ==f'Opponent_{team}']['Raw_Off_Eff'].values[0], axis=1)
            self.stats[team].loc[:,'adj_off_eFG'] = self.stats[team].apply(lambda row: row['off_eFG'] - ridge_df[ridge_df['coef_name'] ==f'Opponent_{team}']['off_eFG'].values[0], axis=1)
            self.stats[team].loc[:,'adj_off_TOV'] = self.stats[team].apply(lambda row: row['off_TOV'] - ridge_df[ridge_df['coef_name'] ==f'Opponent_{team}']['off_TOV'].values[0], axis=1)
            self.stats[team].loc[:,'adj_off_ORB'] = self.stats[team].apply(lambda row: row['off_ORB'] - ridge_df[ridge_df['coef_name'] ==f'Opponent_{team}']['off_ORB'].values[0], axis=1)
            self.stats[team].loc[:,'adj_off_FTR'] = self.stats[team].apply(lambda row: row['off_FTR'] - ridge_df[ridge_df['coef_name'] ==f'Opponent_{team}']['off_FTR'].values[0], axis=1)
            self.stats[team].loc[:,'adj_Raw_Def_Eff'] = self.stats[team].apply(lambda row: row['Raw_Def_Eff'] - ridge_df[ridge_df['coef_name'] ==f'Opponent_{team}']['Raw_Def_Eff'].values[0], axis=1)
            self.stats[team].loc[:,'adj_def_eFG'] = self.stats[team].apply(lambda row: row['def_eFG'] - ridge_df[ridge_df['coef_name'] ==f'Opponent_{team}']['def_eFG'].values[0], axis=1)
            self.stats[team].loc[:,'adj_def_TOV'] = self.stats[team].apply(lambda row: row['def_TOV'] - ridge_df[ridge_df['coef_name'] ==f'Opponent_{team}']['def_TOV'].values[0], axis=1)
            self.stats[team].loc[:,'adj_def_ORB'] = self.stats[team].apply(lambda row: row['def_ORB'] - ridge_df[ridge_df['coef_name'] ==f'Opponent_{team}']['def_ORB'].values[0], axis=1)
            self.stats[team].loc[:,'adj_def_FTR'] = self.stats[team].apply(lambda row: row['def_FTR'] - ridge_df[ridge_df['coef_name'] ==f'Opponent_{team}']['def_FTR'].values[0], axis=1)
            self.stats[team].loc[:,'Distance_Traveled'] = 100 
            self.stats[team].loc[:,'Home'] = 0
            self.stats[team].loc[:,'Away'] = 0

    def simulate_game(self):
        team1_scores = np.array([])
        team2_scores = np.array([])
        fig, ax = plt.subplots()
        print(self.stats[self.team1])
        for i in range(self.number_of_games):
            x1 = self.stats[self.team1].iloc[i][self.model_params].values.reshape(1,-1)
            x2 = self.stats[self.team2].iloc[i][self.model_params].values.reshape(1,-1)
            x1 = self.input_standardizer.transform(x1)
            x2 = self.input_standardizer.transform(x2)
            x1 = torch.from_numpy(x1.astype(np.float32))
            x2 = torch.from_numpy(x2.astype(np.float32))
            team1_score = ANNreg(x1)
            team2_score = ANNreg(x2)
            team1_scores = np.append(team1_scores, team1_score.detach().numpy())
            team2_scores = np.append(team2_scores, team2_score.detach().numpy())
            # print(f"mean team 1: {team1_scores.mean()}, mean team 2: {team2_scores.mean()}")
            # print(f"team 1 score: {team1_score.detach().numpy()}, team 2 score: {team2_score.detach().numpy()}")
            # ax.plot(i, team1_score.detach(),'bo')
            # ax.plot(i, team2_score.detach(),'ro')

            if team1_score.detach().numpy() > team2_score.detach().numpy():
                self.team1_wins += 1
            else:
                self.team2_wins += 1
        ax.set_title(f'{self.team1} vs {self.team2} Scores')
        ax.set_xlabel('Game Number')
        ax.set_ylabel('Score')
        
        self.team1_winprob = self.team1_wins/self.number_of_games
        self.team2_winprob = self.team2_wins/self.number_of_games
        self.team1_predscore = team1_scores.mean()
        self.team2_predscore = team2_scores.mean()
        if self.team1_winprob > self.team2_winprob:
            self.game_winner = self.team1
            self.win_prob = self.team1_winprob
        else:
            self.game_winner = self.team2
            self.win_prob = self.team2_winprob


                
num_games = 500
tournament_df = pd.DataFrame()
def tournament(matchup_dict, num_games,tournament_idx, conference):
    game_counter = 1 
    for round_idx, tourney_round in enumerate(matchup_dict):
        # print(f"{tourney_round} matchups: {matchup_dict[tourney_round]}\n")
        game_num = 1
        for team1, team2 in zip(matchup_dict[tourney_round]['team1'], matchup_dict[tourney_round]['team2']):
            # print(f"{team1} vs {team2}")
            game = matchup(dataframe=test_df, team1=team1, team2=team2, home_team=None,
                        varied_params=model_params, model_params=nn_params, number_of_games=num_games, input_standardizer=scaler)
            
            game.vary_stats()
            game.simulate_game()
            matchup_dict[tourney_round]['round_winners'].append(game.game_winner)
            tournament_df.at[tournament_idx,f"game_{game_counter}"] = None  # Initialize the column with None
            tournament_df.at[tournament_idx, f"game_{game_counter}"] = game.game_winner  # Assign the value
            game_counter += 1

            # print(f"{game.team1} win probability: {game.team1_winprob:0.3f}, {game.team2} win probability: {game.team2_winprob:0.3f}")
            # print(f"{game.team1} predicted score: {game.team1_predscore:0.1f}, {game.team2} predicted score: {game.team2_predscore:0.1f}")
            print(f"{team1} vs {team2} Winner: {game.game_winner}({game.win_prob:0.3f})")
            # print(f"{tourney_round.replace('_',' ')} winners: {matchup_dict[tourney_round]['round_winners']}\n\n")
            #add the winners from the current round into the next round
            next_round = f'round_{round_idx+2}'
            if round_idx + 1 ==1 and conference =='Big 12':
                if game_num == 1:
                    matchup_dict[next_round]['team1'].insert(0,game.game_winner)
                    game_num += 1
                else:
                    matchup_dict[next_round]['team1'].append(game.game_winner)
                    game_num += 1
                # matchup_dict[next_round]['team1'].append(game.game_winner)
            
            if round_idx + 2 < 4:
                if round_idx + 1 == 1 and conference == 'Big 12':
                    continue
            # if next_round!='round_4':
                matchup_dict[next_round]['team1'].append(game.game_winner)
            elif next_round == 'round_4' or next_round == 'round_5':
                if game_num % 2 == 0:
                    matchup_dict[next_round]['team1'].append(game.game_winner)
                    game_num += 1
                else:
                    matchup_dict[next_round]['team2'].append(game.game_winner)
                    game_num += 1
            else:
                matchup_dict[next_round]['champion']= game.game_winner
                print('Champion:', matchup_dict[next_round]['champion'])
                return
            # print(f"Next round matchups: {matchup_dict[next_round]['team1']}\n\n")
conference = 'Big 12'
tournament_simulations = 20

def simulate_tournament(matchup_dict, num_games, tournament_simulations, conference):
    for i in range(tournament_simulations):
        tournament_dict = copy.deepcopy(matchup_dict)
        tournament(tournament_dict, num_games, i,conference)
        # print(tournament_df.describe(include='all').transpose())
    print(tournament_df)
    for col in tournament_df.columns:
        print(tournament_df[col].value_counts(normalize=True))
# for i in range(tournament_simulations):
#     matchup_dict = copy.deepcopy(big_12_matchups)
 
#     print(f"simulating tournament {i+1}")

#     tournament(matchup_dict, num_games, i,conference)
#     # print(tournament_df.describe(include='all').transpose())
# print(tournament_df)
# for col in tournament_df.columns:
#     print(tournament_df[col].value_counts(normalize=True))

# acc = tournament(matchup_dict, num_games,0)
# big_12 = tournament(big_12_matchups, num_games, 0,'Big 12')
        
indiv_game = matchup(dataframe=test_df, team1='Baylor Bears', team2='Cincinnati Bearcats', home_team=None,
                        varied_params=model_params, model_params=nn_params, number_of_games=num_games, input_standardizer=scaler)

indiv_game.vary_stats()
indiv_game.simulate_game()
print(f"{indiv_game.team1} score: {indiv_game.team1_predscore:0.2f}, {indiv_game.team2} score: {indiv_game.team2_predscore:0.2f}")
plt.show()

