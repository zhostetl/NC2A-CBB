import os
import glob
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np 
import random
import time
from datetime import date, timedelta


class scrape_gameids():
    def __init__(self, game_date, retries = 3):
        self.game_date = game_date
        self.retries = retries
        
    
    def get_list_of_games(self):

        headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36' }

        url_date = self.game_date.strftime("%Y%m%d")
        self.url = f'https://www.espn.com/mens-college-basketball/schedule/_/date/{url_date}'
        for i in range(1, self.retries + 1):
            time_interval = random.uniform(3, 7)

            try:
                
                html = requests.get(self.url, time.sleep(time_interval), headers=headers)
                
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
        
                


test_date = date(2022, 11, 7)
# test_date = date(2023, 1, 26)


end_date = date(2023, 3, 12)

completed_game_ids = []
no_games = [date(2022, 12, 24), date(2022,12,25), date(2022, 12, 26)]

while test_date < end_date:
    print(test_date)
    time_interval = random.uniform(2,4)
    
    time.sleep(time_interval)

    if test_date in no_games:
        test_date = test_date + timedelta(days=1)
        continue
    
    game_schedule = scrape_gameids(test_date)
    game_schedule.get_list_of_games()
    game_schedule.get_gameids()
    completed_game_ids.extend(game_schedule.game_ids)
    test_date = test_date + timedelta(days=1)

print(f'Number of completed games: {len(completed_game_ids)}')

completed_game_ids = pd.DataFrame(completed_game_ids, columns = ['game_id'])
completed_game_ids.to_csv('2022_2023_completed_game_ids.csv', index = False)