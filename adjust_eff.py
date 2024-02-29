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

print(df.head(15))

## need to set it up as the team name in a column/variable; include the opponent in that same column and then home field advantage and the distance traveled 
## this is what gets used for the pd.get_dummies function
## then you can set the outcome variable to the desired stat
## this is the outcome variable that we want to adjust to 

dummy_df = pd.get_dummies(df[['Team','Raw_Def_Eff']])

print(dummy_df.head(15))

alf = 175
stat = 'Raw_Off_Eff'

reg = linear_model.Ridge(alpha = alf, fit_intercept = True) 

reg.fit(X = dummy_df, y = df[stat])

# Extract regression coefficients
dfRegResults = pd.DataFrame({
    'coef_name': dummy_df.columns.values,
    'ridge_reg_coef': reg.coef_})

# Add intercept back in to reg coef to get 'adjusted' value
dfRegResults['ridge_reg_value'] = (dfRegResults['ridge_reg_coef']+reg.intercept_)

print(dfRegResults)