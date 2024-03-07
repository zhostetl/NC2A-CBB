import os 
import glob
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#based on this article
#https://blog.collegefootballdata.com/opponent-adjusted-stats-ridge-regression/#:~:text=Averaging%20Method,average%20against%20all%20other%20opponents.
#https://medium.com/analyzing-ncaa-college-basketball-with-gcp/from-college-to-the-pros-with-google-cloud-platform-part-1-f1b151c7b61f
#https://medium.com/analyzing-ncaa-college-basketball-with-gcp/fitting-it-in-adjusting-team-metrics-for-schedule-strength-4e8239be0530
#https://colab.research.google.com/drive/13L4b36cTrnC55ahD6dVf4-r9pzkYV-j5#scrollTo=TQ1zmrOMYSBh 

file = r'C:\Users\zhostetl\Documents\11_CBB\99_git\NC2A-CBB\02_cleandata\cleaned_data.xlsx'

df = pd.read_excel(file)

# print(df.head(15))

def add_opp(df):
    games = df['GameID'].unique()
    
    for game in games: 
        game_indices = df[df['GameID'] == game]
        df.loc[game_indices.index[0], 'Opponent'] = df.loc[game_indices.index[1], 'Team']
        df.loc[game_indices.index[1], 'Opponent'] = df.loc[game_indices.index[0], 'Team']
    return df

## need to set it up as the team name in a column/variable; include the opponent in that same column and then home field advantage and the distance traveled 
## this is what gets used for the pd.get_dummies function
## then you can set the outcome variable to the desired stat
## this is the outcome variable that we want to adjust to 
        
df = add_opp(df)

# print(df[['Team','Opponent','GameID','Win','Loss']].head(15))

dummy_df = pd.get_dummies(df[['Team','Opponent','Home']])

#stats to adjust for: 
#Raw_Off_Eff, Raw_Def_Eff
# + four factors
# off_eFg, off_TOV, off_ORB, off_FTR
# def_eFg, def_TOV, def_ORB, def_FTR
stats_to_adjust = ['Raw_Off_Eff','Raw_Def_Eff','off_eFG','off_TOV','off_ORB','off_FTR','def_eFG','def_TOV','def_ORB','def_FTR']
# stats_to_adjust = ['Raw_Off_Eff','Raw_Def_Eff']
adjusted_dict = {}
for stat in stats_to_adjust:
    rdcv = linear_model.RidgeCV(alphas = [0.1,0.5,1,2,3,4,5,10,15], fit_intercept = True)
    rdcv.fit(dummy_df,df[stat])
    alf = rdcv.alpha_
    r_squared = rdcv.score(dummy_df,df[stat])
    print(f"\nridge regression r-squared: {r_squared:0.3f}, alpha: {alf} for {stat}\n")
    reg = linear_model.Ridge(alpha = alf, fit_intercept = True)
    reg.fit(X = dummy_df, y = df[stat])
    dfRegResults = pd.DataFrame({
        'coef_name': dummy_df.columns.values,
        'ridge_reg_coef': reg.coef_})
    # print(dfRegResults.tail(15))
    # fig, ax = plt.subplots()
    # ax.hist(dfRegResults['ridge_reg_coef'], bins = 50)
    # ax.set_title(f"ridge regression coefficients for {stat}")
    # dfRegResults['ridge_reg_value'] = (dfRegResults['ridge_reg_coef']+reg.intercept_)
    dfRegResults['ridge_reg_value'] = (dfRegResults['ridge_reg_coef'])
    # ax.hist(dfRegResults['ridge_reg_value'], bins = 50)
    # print(dfRegResults.tail(20))
    # sdf = dfRegResults[dfRegResults['coef_name'].str.contains('Baylor')].rename(columns = {"ridge_reg_value": stat}).reset_index(drop = True)
    # print(sdf)
    
    # adjusted_df = dfRegResults[dfRegResults['coef_name'].str.contains('Team_')].rename(columns = {"ridge_reg_value": stat}).reset_index(drop = True)
    adjusted_df = dfRegResults.rename(columns = {"ridge_reg_value": stat}).reset_index(drop = True)
    adjusted_df['coef_name'] = adjusted_df['coef_name'].str.replace('Team_','')
    adjusted_df = adjusted_df.drop(columns=['ridge_reg_coef'])
    adjusted_dict[stat] = adjusted_df
    # print(adjusted_df.tail(15))

# adjusted metric becomes: 
# for each row in the dataframe, find the team and the name of the opponent and also the home court advantage
# adjusted metric becomes: raw metric - home court advantage - opponent metric
    
for key in adjusted_dict:
    for idx, row in df.iterrows():
        raw_metric = row[key]
        opponent = row['Opponent']
        adf = adjusted_dict[key]
        if row['Home'] == 1:
            home_adv = adf[adf['coef_name']=='Home'][key].values[0]
        else:
            home_adv = 0 
        
        adj_team_metric = adf[adf['coef_name']==row['Team']][key].values[0] #dont think we actually need this
        adj_opp_metric = adf[adf['coef_name']==f"Opponent_{opponent}"][key].values[0] 
        df.loc[idx,'adj_'+key] = raw_metric - adj_opp_metric - home_adv
        # print(f"raw metric: {raw_metric:0.2f}, adjusted metric: {row['adj_'+key]:0.2f},\n {homeaway} opponent rating: {adj_opp_metric:0.2f} opponent: {opponent}, {winloss}")
        # print(adj_team_metric)
print(df[['Team','Opponent','GameID','Win','Loss','Raw_Off_Eff','adj_Raw_Off_Eff','Raw_Def_Eff','adj_Raw_Def_Eff']])

# df.to_excel(r'C:\Users\zhostetl\Documents\11_CBB\99_git\NC2A-CBB\03_modelfitting\adjusted_data.xlsx')

# team_of_interest = 'Baylor Bears'
# sdf = df[df['Team'] == team_of_interest]
# print(sdf.head())

# for idx, row in sdf.iterrows():
#     raw_metric = row['Raw_Off_Eff']
#     opponent = row['Opponent']
    
#     adf = adjusted_dict['Raw_Off_Eff']
#     if row['Home'] == 1:
#         home_adv = adf[adf['coef_name']=='Home']['Raw_Off_Eff'].values[0]
#         homeaway = 'Home'
#     else:
#         home_adv = 0 
#         homeaway = 'Away'
#     if row['Win'] == 1:
#         winloss = 'Win'
#     else:
#         winloss = 'Loss'
#     adj_team_metric = adf[adf['coef_name']==team_of_interest]['Raw_Off_Eff'].values[0]
#     adj_opp_metric = adf[adf['coef_name']==f"Opponent_{opponent}"]['Raw_Off_Eff'].values[0] 
#     sdf.loc[idx,'adj_Raw_Off_Eff'] = raw_metric - adj_opp_metric - home_adv
#     row['adj_Raw_Off_Eff'] = raw_metric - adj_opp_metric - home_adv
#     # print(f"raw metric: {raw_metric:0.2f}, adjusted metric: {row['adj_Raw_Off_Eff']:0.2f},\n {homeaway} opponent rating: {adj_opp_metric:0.2f} opponent: {opponent}, {winloss}")
#     # print(adj_team_metric)
# print(sdf[['Team','Opponent','GameID','Win','Loss','Raw_Off_Eff','adj_Raw_Off_Eff']])

# # for key in adjusted_dict:
# #     print(adjusted_df[key][key])
# #     df['adj_'+key] = df['Team'].map(adjusted_dict[key][key])
# #     print(df.head(15))

# plt.show()
