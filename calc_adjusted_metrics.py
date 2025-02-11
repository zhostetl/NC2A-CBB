from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
import pandas as pd
import time 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from webscraping.web_scrapper import Scraper
from database.database import *
import matplotlib.pyplot as plt

t1 = time.time()

engine = create_engine('sqlite:///database/ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

soi = '2025'

# season_year = session.query(Season).filter(Season.year == soi).first()

# season_data = session.query(Games).filter(Games.season_id == season_year.id).filter(Games.game_num==1).all()
# print(len(season_data))
#form dataframe for pd.get_dummies and ridge regression

from sqlalchemy.orm import aliased

def query_db(session, season):
    season_year = session.query(Season).filter(Season.year == season).first()

    game_ids = session.query(GameID).filter(GameID.season_id == season_year.year).all()
    print(f"Number of game IDs in {season}: {len(game_ids)} from GameID table")
    games = session.query(Games).filter(Games.season_id == season_year.id).all()
    print(f"Number of games in {season}: {len(games)} from Games table")
    team_alias = aliased(Teams)
    opponent_alias = aliased(Teams)
    team_conf_alias = aliased(Conferences)
    opponent_conf_alias = aliased(Conferences)

    season_data = session.query(Games, team_alias, opponent_alias, team_conf_alias, opponent_conf_alias).join(
        team_alias, Games.team_id == team_alias.id).join(
        opponent_alias, Games.opponent_id == opponent_alias.id).join(
        team_conf_alias, Games.conference_id == team_conf_alias.id).join(
        opponent_conf_alias, Games.opponent_conference_id == opponent_conf_alias.id).filter(
        Games.season_id == season_year.id).all()
    
    # print(f"Number of games in {season}: {len(season_data)} from Games table")

    """
    There are some game ids that do not get converted to the games table because one of the teams is not a D1 team and does not have the correct data in the database. That is why the games table will be shorter than the game ids table
    """

    return season_data

def return_df2(season_data):
    df1 = pd.DataFrame([game[0].__dict__ for game in season_data]) #game stat data
    df2 = pd.DataFrame([game[1].__dict__ for game in season_data]) #home team name
    df3 = pd.DataFrame([game[2].__dict__ for game in season_data]) #away team name
    df4 = pd.DataFrame([game[3].__dict__ for game in season_data]) #home team conference
    df5 = pd.DataFrame([game[4].__dict__ for game in season_data]) #away team conference

    df1_cols = ['home','away','neutral','game_num','offensive_efficiency','defensive_efficiency','effective_field_goal_percentage','turnover_percentage','offensive_rebound_percentage','free_throw_rate']
    df2_cols = ['espn_name']
    df3_cols = ['espn_name']
    df4_cols = ['name']
    df5_cols = ['name']

    df1 = df1.drop(columns=['_sa_instance_state','referee1','referee2','referee3','game_state','game_location','over_under','betting_line','date'])
    df2 = df2[df2_cols].rename(columns={'espn_name':'team_name'})
    df3 = df3[df3_cols].rename(columns={'espn_name':'opponent_name'})
    df4 = df4[df4_cols].rename(columns={'name':'conference'})
    df5 = df5[df5_cols].rename(columns={'name':'opponent_conference'})

    df = pd.concat([df1, df2, df3, df4, df5], axis=1)
    
    return df

def return_df(season_data):
    data = [{
        'team_name': session.query(Teams).filter(Teams.id == game.team_id).first().espn_name,
        'opponent_name': session.query(Teams).filter(Teams.id == game.opponent_id).first().espn_name,
        'conference': session.query(Conferences).filter(Conferences.id == game.conference_id).first().name,
        'home': game.home,
        'away': game.away,
        'neutral': game.neutral,
        'game_num': game.game_num,

        'offensive_efficiency': game.offensive_efficiency,
        'defensive_efficiency': game.defensive_efficiency,
        
        'offensive_effective_field_goal_percentage': game.effective_field_goal_percentage,
        'offensive_turnover_percentage': game.turnover_percentage,
        'offensive_rebound_percentage': game.offensive_rebound_percentage,
        'free_throw_rate': game.free_throw_rate,
        
        # 'defensive_effective_field_goal_percentage': game.defensive_effective_field_goal_percentage,
        # 'defensive_turnover_percentage': game.defensive_turnover_percentage,
        # 'defensive_rebound_percentage': game.defensive_rebound_percentage,
        # 'opponent_offensive_rebound_percentage': game.opponent_offensive_rebound_percentage,
    }
    for game in season_data]

    df = pd.DataFrame(data)
    return df

def ridge_regression(df, metric, alphas = [0.001,0.01,0.1,1,10,100,1000], cross_val = 5, normalize_data = True):
    
    # # dummy_df = pd.get_dummies(df, columns = ['team_name','opponent_name', 'conference'], drop_first = True)
    dummy_df = pd.get_dummies(df, columns = ['team_name','opponent_name', 'conference', 'opponent_conference','home'])
    
    # Step 2: Define features (X) and target (y)
    X = dummy_df.drop(metric, axis=1)
    y = dummy_df[metric]

    x_cols = X.columns

    if normalize_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = scaler.fit_transform(y.values.reshape(-1, 1))

    # Step 3: Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Apply RidgeCV for cross-validated alpha selection
      # Define range of alpha values
    ridge_cv_model = linear_model.RidgeCV(alphas=alphas, cv=cross_val)  # 5-fold cross-validation
    ridge_cv_model.fit(X_train, y_train)
    # Step 5: Make predictions and evaluate
    y_pred = ridge_cv_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Output results
    print(f"----------{metric}----------")
    print(f"Mean Squared Error: {mse:0.5f}")
    print(f"Optimal Alpha (Regularization Parameter): {ridge_cv_model.alpha_}")
    print(f"R^2: {ridge_cv_model.score(X_test, y_test):0.2f}")
    print(f"---------------------\n")
    # Step 6: Feature importance
    # print("Feature Coefficients:")
    coef_dict = dict(zip(x_cols, ridge_cv_model.coef_.flatten()))
    coef_df = pd.DataFrame({
        "Feature": x_cols,
        "Coefficient": ridge_cv_model.coef_.flatten()
    })
    # for key, val in coef_dict.items():
    #     print(f"{key}: {val:0.5f}")
    
    return coef_dict, coef_df

def adjust_metrics(df, adjusted_vals_dict):
    adjusted_dict = {f"adj_{key}": [] for key in adjusted_vals_dict}
    # adjusted_dict['efficiency_margin'] = []
    for idx, row in df.iterrows():
        # if idx > 10:
        #     break
        game_id = row['game_id']
        team_id = row['team_id']
        team_name = row['team_name']
        game_num = row['game_num']
        opponent = row['opponent_name']
        opponent_conf = row['opponent_conference']
        # adjusted_dict = {}
        for metric in adjusted_vals_dict:
            raw_metric = row[metric]
            #check if it was a home game
            if row['home'] == 1:
                home_adv = adjusted_vals_dict[metric]['home_1']
            else:
                home_adv = adjusted_vals_dict[metric]['home_0']
            
            #check the conference
            conference = row['conference']
            conference_adv = adjusted_vals_dict[metric][f"conference_{conference}"]
            
            #look up the opponent's adjusted metric
            opponent_value = adjusted_vals_dict[metric][f"opponent_name_{opponent}"]
            opponent_conf_value = adjusted_vals_dict[metric][f"opponent_conference_{opponent_conf}"]
            # adj = 'adjusted_' + metric
            adjusted_dict[f"adj_{metric}"].append(raw_metric - home_adv - opponent_value - conference_adv - opponent_conf_value)
        # adjusted_dict['efficiency_margin'].append(adjusted_dict['adj_offensive_efficiency'] - adjusted_dict['adj_defensive_efficiency'])
        
    adjusted_df = pd.DataFrame.from_dict(adjusted_dict)
    adjusted_df['efficiency_margin'] = adjusted_df['adj_offensive_efficiency'] - adjusted_df['adj_defensive_efficiency']
    print(adjusted_df.head())
    return adjusted_df
        


season_data = query_db(session, soi)

t2 = time.time()
print(f"Time to query: {t2-t1:0.2f} seconds")

t3 = time.time()
df = return_df2(season_data)

# # df = return_df(season_data)
t4 = time.time()
print(f"Time to return df: {t4-t3:0.2f} seconds\n")

adjusted_metrics = ['eFG','TO_rate','OREB_per','FT_rate',
                    'opp_eFG','opp_TO_rate','DREB_per','opp_FT_rate',
                    'offensive_efficiency','defensive_efficiency']

# adjusted_metrics = ['offensive_efficiency','defensive_efficiency']

# for adj_metric in adjusted_metrics:
#     rm = RidgeMetrics(metric = adj_metric)
#     session.add(rm)
# session.commit()

adjusted_dict = {}

for metric in adjusted_metrics:
       
    t5 = time.time()
    # sdf = df[df['game_num'] <= 3]
    
    adjusted_dict[metric], adjusted_vals = ridge_regression(df, f'{metric}', normalize_data=True)

    metric_id = session.query(RidgeMetrics).filter(RidgeMetrics.metric == metric).first().id
    season_id = session.query(Season).filter(Season.year == soi).first().id


    for key, val in adjusted_dict[metric].items():
        if 'opponent_name' in key: 
            opponent_name = key.split('_')[-1]
            opponent_id = session.query(Teams).filter(Teams.espn_name == opponent_name).first().id
            conference_id = session.query(Conferences).filter(Conferences.name == df[df['team_name'] == opponent_name]['conference'].values[0]).first().id
            opponent_value = val
            conference_value = adjusted_dict[metric][f"opponent_conference_{df[df['team_name'] == opponent_name]['conference'].values[0]}"]
            # print(f"Metric: {metric}, Opponent: {opponent_name}, Conference id: {conference_id}, opponent_id: {opponent_id}, value: {opponent_value}, conference_value: {conference_value}")
            home_value = adjusted_dict[metric]['home_1']
            away_value = adjusted_dict[metric]['home_0']
            #need to look up current value in the database and update it
            ridge_value = session.query(RidgeResults).filter(RidgeResults.metric_id == metric_id, RidgeResults.opponent_id == opponent_id, RidgeResults.conference_id == conference_id, season_id == season_id).first()
            ridge_value.opponent_value = opponent_value
            ridge_value.conference_value = conference_value
            ridge_value.home_value = home_value
            ridge_value.away_value = away_value
            session.commit()

            # ridge_value = RidgeResults(metric_id = metric_id, season_id = season_id, opponent_id = opponent_id, opponent_value = opponent_value, conference_id = conference_id, conference_value = conference_value, home_value = home_value, away_value = away_value)
    #         session.add(ridge_value)
    # session.commit()
        # print(f"{key}: {val:0.5f}")



ridge_adjusted = adjust_metrics(df, adjusted_dict)


#add adjusted metrics to database; but only for the latest games
adjusted_game_ids = session.query(AdjustedMetrics.game_id).all()

print(f"\n********************\nAdding adjusted metrics to database\n********************\n")

for idx, row in df.iterrows():
    game_id = row['game_id']
    if (game_id,) in adjusted_game_ids:
        continue
        # print(f"Game {game_id} already has adjusted metrics")
    team_id = row['team_id']
    # print(idx, game_id)
    adj_efg_percentage = ridge_adjusted.loc[idx, 'adj_eFG']
    adj_turnover_percentage = ridge_adjusted.loc[idx, 'adj_TO_rate']
    adj_offensive_rebound_percentage = ridge_adjusted.loc[idx, 'adj_OREB_per']
    adj_free_throw_rate = ridge_adjusted.loc[idx, 'adj_FT_rate']
    
    opp_adj_efg_percentage = ridge_adjusted.loc[idx, 'adj_opp_eFG']
    opp_adj_turnover_percentage = ridge_adjusted.loc[idx, 'adj_opp_TO_rate']
    adj_def_rebound_percentage = ridge_adjusted.loc[idx, 'adj_DREB_per']
    opp_adj_free_throw_rate = ridge_adjusted.loc[idx, 'adj_opp_FT_rate']

    adj_offensive_efficiency = ridge_adjusted.loc[idx, 'adj_offensive_efficiency']
    adj_defensive_efficiency = ridge_adjusted.loc[idx, 'adj_defensive_efficiency']
    adj_efficiency_margin = ridge_adjusted.loc[idx, 'efficiency_margin']

    adjusted_game = AdjustedMetrics(game_id = game_id, team_id = team_id,
                                     adj_efg_percentage = adj_efg_percentage, adj_turnover_percentage = adj_turnover_percentage,
                                     adj_offensive_rebound_percentage = adj_offensive_rebound_percentage, adj_free_throw_rate = adj_free_throw_rate,
                                     opp_adj_efg_percentage = opp_adj_efg_percentage, opp_adj_turnover_percentage = opp_adj_turnover_percentage,
                                     adj_def_rebound_percentage = adj_def_rebound_percentage, opp_adj_free_throw_rate = opp_adj_free_throw_rate,
                                     adj_offensive_efficiency = adj_offensive_efficiency, adj_defensive_efficiency = adj_defensive_efficiency, adj_efficiency_margin = adj_efficiency_margin)
    session.add(adjusted_game)

session.commit()

test_df = pd.concat([df, ridge_adjusted], axis=1)
print(test_df.groupby('team_name')['adj_offensive_efficiency'].mean().sort_values(ascending=False))
 


t6 = time.time()
print(f"Time to adjust metrics: {t6-t5:0.2f} seconds\n")


t2 = time.time()

print(f"Time to run: {t2-t1:0.2f} seconds")

# plt.show()