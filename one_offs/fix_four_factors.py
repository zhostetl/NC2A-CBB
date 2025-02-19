from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, aliased
from datetime import datetime
import pandas as pd
import time 

from web_scrapper import Scraper
from database import *
import matplotlib.pyplot as plt
import seaborn as sns

t1 = time.time()

engine = create_engine('sqlite:///ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

season_data = session.query(Games).all()

print(f"len(season_data): {len(season_data)}")


def four_factors(team_stats, opponent_stats):

    opp_efg = (opponent_stats.field_goals_made + 0.5 * opponent_stats.three_point_field_goals_made) / opponent_stats.field_goals_attemped
    opp_to = opponent_stats.total_turnovers / (opponent_stats.field_goals_attemped + 0.44 * opponent_stats.free_throws_attempted + opponent_stats.total_turnovers)
    dreb_per = team_stats.defensive_rebounds / (team_stats.defensive_rebounds + opponent_stats.defensive_rebounds)
    opp_ft_rate = opponent_stats.free_throws_made / team_stats.field_goals_attemped

    to_rate = team_stats.total_turnovers / (team_stats.field_goals_attemped + 0.44 * team_stats.free_throws_attempted + team_stats.total_turnovers)

    return opp_efg, opp_to, dreb_per, opp_ft_rate, to_rate

missing = []

for idx, game in enumerate(season_data):
    if idx > 5: 
        break
    game_id = game.game_id
    team_id = game.team_id
    opponent_id = game.opponent_id
    season_id = game.season_id
    # print(f"game_id: {game_id}, team_id: {team_id}, opponent_id: {opponent_id}")

    opp_stats = session.query(Games).filter(Games.game_id == game_id, Games.team_id == opponent_id).first()
    if opp_stats is None:
        print(f"opp_stats is None for game_id: {game_id}")
        missing.append(game_id)
        continue
    opp_eFG, opp_TO_per, dreb_per, opp_FT_rate, TO_rate= four_factors(game, opp_stats)
    # print(f"opp_eFG: {opp_eFG}, opp_TO_per: {opp_TO_per}, dreb_per: {dreb_per}, opp_FT_rate: {opp_FT_rate}\nCorrected TO_rate: {TO_rate}")

    game.opp_eFG = round(opp_eFG,5)
    game.opp_TO_rate = round(opp_TO_per,5)
    game.DREB_per = round(dreb_per,5)
    game.opp_FT_rate = round(opp_FT_rate,5)
    game.TO_rate = round(TO_rate,5)
    # session.commit()
    
    # print(f"updated game_id: {game_id}")
# session.commit()