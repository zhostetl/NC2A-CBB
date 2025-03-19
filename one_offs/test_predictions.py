import os 
import pandas as pd 
import numpy as np 
from database.database import *
from models.NN_model import *
import matplotlib.pyplot as plt
import seaborn as sns

# Construct the relative path to the database file
db_folder = os.path.join(os.path.dirname(__file__), '..', 'database')
db_name = 'ncaa_basketball.db'
db_path = os.path.join(db_folder, db_name)

# Create the SQLAlchemy engine using the constructed database path
engine = create_engine(f'sqlite:///{db_path}')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

nn_model = NN_Model()
nn_model.load_model(session)

model_params = nn_model.params

print(f"model params: {model_params}")

season = 2025

season_id = session.query(Season).filter(Season.year == season).first()

game_id = 401706746

# game_ids = session.query(Games.game_id).filter(Games.season_id == season_id.id).filter(Games.date > '2025-02-01').all()

# game_ids = [set(game.game_id for game in game_ids)]
# error = []

# def check_predicted_score(game_id,error):


#     game_of_interest = session.query(Games).filter(Games.game_id == game_id).all()
    
#     for idx, team in enumerate(game_of_interest):
        

#         team1_id = game_of_interest[idx].team_id
#         team2_id = game_of_interest[idx].opponent_id

#         team1_name = session.query(Teams).filter(Teams.id == team1_id).first().espn_name
#         team2_name = session.query(Teams).filter(Teams.id == team2_id).first().espn_name

#         # game_location = game_of_interest.game_state

#         #get the input parameters for the model from the games table. need to join with the adjusted metrics table 
#         # game_data = session.query(Games, AdjustedMetrics, Teams).join(AdjustedMetrics, (Games.game_id == AdjustedMetrics.game_id) & (Games.team_id == AdjustedMetrics.team_id)).join(Teams, Games.team_id==Teams.id).filter(Games.season_id == game_id).all()

#         season_data = session.query(Games, AdjustedMetrics, Teams).join(AdjustedMetrics, (Games.game_id == AdjustedMetrics.game_id) & (Games.team_id == AdjustedMetrics.team_id)).join(Teams, Games.team_id==Teams.id).filter(Games.game_id == game_id).filter(Games.team_id==team1_id).all()


#         game_df = pd.DataFrame([{**game.__dict__, **metrics.__dict__, **teams.__dict__} for game, metrics, teams, in season_data])

#         # print(game_df[model_params])

#         #scale the data
#         team1_scaled = nn_model.scaler.transform(game_df[model_params].values)
#         #convert to tensor
#         team1_scaled = torch.from_numpy(team1_scaled.astype(np.float32))

#         #predict the score
#         predicted_score1 = nn_model.predict(team1_scaled)
#         predicted_score1 = predicted_score1.detach().numpy()
#         difference = predicted_score1[0][0] - game_of_interest[idx].points

#         error.append(difference)
#         # print(f"Predicted score for {team1_name}: {predicted_score1} vs actual score {game_of_interest[idx].points} (difference: {difference:0.2f})")
#         # print(f"Actual score for {team1_name}: {game_of_interest[idx].points}")

# for game_id in game_ids[0]:
#     check_predicted_score(game_id,error)

# plt.hist(error, bins=20)
# plt.show()