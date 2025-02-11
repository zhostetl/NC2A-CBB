import os
import glob
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np 
import random
import time
from datetime import date, timedelta

from webscraping.web_scrapper import Scraper
from database.database import *

"""
This is code that will get all of the game ids for a given season and input them into the database. 
This can be run for the following scenarios:
    1. Getting all of the game ids for a given season (e.g. 2020-2021 season)
    2. Getting the current game ids for a given date range in the current season

"""
# Construct the relative path to the database file
db_folder = os.path.join(os.path.dirname(__file__), '..', 'database')
db_name = 'ncaa_basketball.db'
db_path = os.path.join(db_folder, db_name)

# Create the SQLAlchemy engine using the constructed database path
engine = create_engine(f'sqlite:///{db_path}')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()


class scrape_gameids():
    def __init__(self, game_date, retries = 3):
        self.game_date = game_date
        self.retries = retries
        
    
    def get_list_of_games(self):

        headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36' }

        url_date = self.game_date.strftime("%Y%m%d")
        self.url = f'https://www.espn.com/mens-college-basketball/schedule/_/date/{url_date}'
        for i in range(1, self.retries + 1):
            time_interval = random.uniform(2, 4)

            try:
                
                html = requests.get(self.url, headers=headers)
                time.sleep(time_interval)
                
                self.game_schedule = BeautifulSoup(html.content, 'html.parser')
            
            except requests.exceptions.ConnectionError:
                print(f'error on {self.game_date} attempt {i} of {self.retries}')
                continue
            else:
                break
    
    def get_gameids(self):
        self.game_ids = []
        table = self.game_schedule.find_all('div', class_ = "Table__Scroller")
        
        rows = table[0].find_all('tr')
        
        for row in rows:
            cells = row.find_all("td", class_ = "teams__col Table__TD")
            if len(cells)>0:
                game_link = cells[0].find_all('a', class_ = "AnchorLink")
                full_url = game_link[0].get('href')
                game_id = full_url.split('/')[-2]
                self.game_ids.append(game_id)
        

webscraper = Scraper()

seasons = session.query(Season).all()


for season in seasons:
    if season.year != 2025:
        continue
    print(f"Getting game ids for {season.year}")
    start_year = season.year
    end_year = season.year
    start_date = date(start_year, 1, 18)
    end_date = date(end_year, 1, 25)
    delta = end_date - start_date

    
    for i in range(delta.days + 1):
        test_date = start_date + timedelta(days=i)
        webscraper.get_list_of_games(test_date)
        if not hasattr(webscraper, 'game_ids'):
            continue
        # webscraper.get_gameids()

        for game in webscraper.game_ids:
            game_id = GameID(game_id = game, season_id = season.year)
            session.add(game_id)
        session.commit()
        print(f"Added {len(webscraper.game_ids)} games for {test_date}")
