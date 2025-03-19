from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, aliased

from database.database import *
import pandas as pd


engine = create_engine('sqlite:///database/ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

class AppDB:
    def __init__(self):
        self.engine = create_engine('sqlite:///database/ncaa_basketball.db')
        self.Base = declarative_base()
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        

    
    def games_by_season(self, season, conference_value=None):
        
        keep_cols = ['date','team_name','opponent_name','game_state','points','win']

        
        # season_data = session.query(Games, AdjustedMetrics, Teams).join(AdjustedMetrics, (Games.game_id == AdjustedMetrics.game_id) & (Games.team_id == AdjustedMetrics.team_id)).join(Teams, Games.team_id==Teams.id).filter(Games.season_id == season_id.id).all()

        Opponent = aliased(Teams)

        season_id = self.session.query(Season).filter(Season.year == season).first()

        if conference_value:
            conference_id = self.session.query(Conferences).filter(Conferences.name == conference_value).first()
            games = self.session.query(Games, Teams, Opponent).join(Teams, Games.team_id == Teams.id).join(Opponent, Games.opponent_id == Opponent.id).filter(Games.season_id == season_id.id, Games.conference_id == conference_id.id).all()
        else:
            games = self.session.query(Games,Teams,Opponent).join(Teams, Games.team_id==Teams.id).join(Opponent, Games.opponent_id ==Opponent.id).filter(Games.season_id == season_id.id).all()
        
        df = pd.DataFrame([{**game.__dict__, **teams.__dict__, 'team_name': teams.espn_name, 'opponent_name':Opponent.espn_name} for game, teams,Opponent in games])
        
        df = df[keep_cols]
        return df
        
    def matchup_data(self, data_table):
        opponent = data_table['opponent_name']
        home_team = data_table['team_name']
        game_loc = data_table['game_state']
        
        team1 = self.session.query(Teams).filter(Teams.espn_name.ilike(f'%{opponent}%')).first()
        team2 = self.session.query(Teams).filter(Teams.espn_name.ilike(f'%{home_team}%')).first()
        game_location = session.query(GameLocations).filter(GameLocations.location.ilike(f'%{game_loc}%')).first()
        return team1,team2,game_location

