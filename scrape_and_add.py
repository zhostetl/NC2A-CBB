from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
import pandas as pd
import numpy as np 
from concurrent.futures import ThreadPoolExecutor
import traceback
import concurrent.futures
import threading
import queue
import os
import time 
from datetime import date, timedelta

from geopy.geocoders import Nominatim
from geopy.distance import geodesic

from webscraping.web_scrapper import Scraper
from webscraping.scrape_gamescores import Matchup
from database.database import *

engine = create_engine('sqlite:///database/ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

season = 2025

db_season = session.query(Season).filter(Season.year == season).first()

db_games = session.query(Games).filter(Games.season_id == db_season.id).all()
game_ids = [game.game_id for game in db_games]

db_gameids = session.query(GameID).filter(GameID.season_id == season).all()


db_game_identifiers = [game.game_id for game in db_gameids]


scraped_games = []

def threaded_scrape_v2(game_queue, scraped_games):
    while not game_queue.empty():
        try:
            lock = threading.Lock()
            game = game_queue.get_nowait()
            webscraper = Scraper()
            with lock:
                game_stat = webscraper.scrape_teamstats(game_id=game)
                scraped_games.append(game_stat)
                game_queue.task_done()
        except queue.Empty:
            break
        except Exception as e:
            print(f"Error scraping game {game}: {e}")
            traceback.print_exc()
        # game_data = Matchup(game_stat, session)
        # game_data.add_game_data(team_of_interest = game_data.team1, opponent = game_data.team2)
        # game_data.add_game_data(team_of_interest = game_data.team2, opponent = game_data.team1)
        # session.commit()
            game_queue.task_done()

MAX_WORKERS = 7

game_queue = queue.Queue()

webscraper = Scraper()

start_date = date(2025, 2, 9)
end_date = date(2025, 2, 9)
                
delta = end_date - start_date

for i in range(delta.days + 1):
    test_date = start_date + timedelta(days=i)
    print(test_date)
    # if test_date != date(2025, 1, 17):
    #     continue
    webscraper.get_list_of_games(test_date)
    if not hasattr(webscraper, 'game_ids'):
        continue
    

    for idx, game in enumerate(webscraper.game_ids):
   
        if int(game) not in db_game_identifiers:
            game_id = GameID(game_id = game, season_id = db_season.id)
            session.add(game_id)
            session.commit()
            print(f"Added game ID: {game} to database")
        
        if int(game) in game_ids: 
            print(f"game ID: {game} already in database")
        
        else:
            game_queue.put(game)
            # game_stat = webscraper.scrape_teamstats(game_id=game)
            # game_data = Matchup(game_stat, session)
            # # game_data.summarize_game()
                
            # game_data.add_game_data(team_of_interest = game_data.team1, opponent = game_data.team2)
            # game_data.add_game_data(team_of_interest = game_data.team2, opponent = game_data.team1)
            # session.commit()

print(f"Total games to scrape: {game_queue.qsize()}")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(threaded_scrape_v2, game_queue, scraped_games) for _ in range(MAX_WORKERS)]

print(f"Scraped {len(scraped_games)} games")

for game in scraped_games:
    try:
        game_data = Matchup(game, session)
        game_data.add_game_data(team_of_interest = game_data.team1, opponent = game_data.team2)
        game_data.add_game_data(team_of_interest = game_data.team2, opponent = game_data.team1)
        session.commit()
    except Exception as e:
        print(f"Error adding {game}: {e}")
        traceback.print_exc()
