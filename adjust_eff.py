import os 
import glob
import pandas as pd
import numpy as np
from sklearn import linear_model

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
adjusted_dict = {}
for stat in stats_to_adjust:
    rdcv = linear_model.RidgeCV(alphas = [0.1,0.5,1,2,3,4,5,10,15], fit_intercept = True)
    rdcv.fit(dummy_df,df[stat])
    alf = rdcv.alpha_
    r_squared = rdcv.score(dummy_df,df[stat])
    print(f"\n\nridge regression r-squared: {r_squared:0.3f}, alpha: {alf} for {stat}\n\n")
    reg = linear_model.Ridge(alpha = alf, fit_intercept = True)
    reg.fit(X = dummy_df, y = df[stat])
    dfRegResults = pd.DataFrame({
        'coef_name': dummy_df.columns.values,
        'ridge_reg_coef': reg.coef_})
    
    dfRegResults['ridge_reg_value'] = (dfRegResults['ridge_reg_coef']+reg.intercept_)
    # print(dfRegResults.tail(20))
    # sdf = dfRegResults[dfRegResults['coef_name'].str.contains('Baylor')].rename(columns = {"ridge_reg_value": stat}).reset_index(drop = True)
    # print(sdf)
    
    adjusted_df = dfRegResults[dfRegResults['coef_name'].str.contains('Team_')].rename(columns = {"ridge_reg_value": stat}).reset_index(drop = True)
    adjusted_df['coef_name'] = adjusted_df['coef_name'].str.replace('Team_','')
    adjusted_df = adjusted_df.drop(columns=['ridge_reg_coef'])
    adjusted_dict[stat] = adjusted_df
    # print(adjusted_df.head(15))

for key in adjusted_dict:
    print(adjusted_df[key][key])
    df['adj_'+key] = df['Team'].map(adjusted_dict[key][key])
    print(df.head(15))


# # print(dummy_df.head(15))
# rdcv = linear_model.RidgeCV(alphas = [0.1,0.5,1,1.5,2,2.5], fit_intercept = True)
# rdcv.fit(dummy_df,df['Raw_Def_Eff'])
# alf = rdcv.alpha_
# r_squared = rdcv.score(dummy_df,df['Raw_Off_Eff'])
# print(f"\n\nridge regression r-squared: {r_squared}\n\n")
# print(f"\n\nridge regression alpha: {alf}\n\n")
# stat = 'Raw_Off_Eff'

# reg = linear_model.Ridge(alpha = alf, fit_intercept = True) 

# reg.fit(X = dummy_df, y = df[stat])

# # Extract regression coefficients
# dfRegResults = pd.DataFrame({
#     'coef_name': dummy_df.columns.values,
#     'ridge_reg_coef': reg.coef_})

# # Add intercept back in to reg coef to get 'adjusted' value
# dfRegResults['ridge_reg_value'] = (dfRegResults['ridge_reg_coef']+reg.intercept_)
