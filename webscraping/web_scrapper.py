import os
import glob
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np 
import random
import time
from datetime import date, timedelta

from sqlalchemy import create_engine, Column, Integer, String, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

class Scraper:
    def __init__(self, retries = 3):
        self.retries = retries
        self.headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36' }

    
    def get_list_of_games(self, game_date):
        url_date = game_date.strftime("%Y%m%d")
        self.url = f'https://www.espn.com/mens-college-basketball/schedule/_/date/{url_date}'
        for i in range(1, self.retries + 1):
            time_interval = random.uniform(3, 7)

            try:
                
                html = requests.get(self.url, time.sleep(time_interval), headers=self.headers)
                
                self.game_schedule = BeautifulSoup(html.content, 'html.parser')
                if self.game_schedule.find_all('section', class_ = "EmptyTable"):
                    print(f'No games on {game_date}')
                    return
                else:
                    print('Games found')
                    self.get_gameids()
                # section = self.game_schedule.find_all('section', class_ = "EmptyTable")

                # print(section)
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
    
    def scrape_teamstats(self, game_id):
        self.boxscore_url = f'https://www.espn.com/mens-college-basketball/boxscore/_/gameId/{game_id}'
        self.teamstats_url = f'https://www.espn.com/mens-college-basketball/matchup/_/gameId/{game_id}'
        self.game_status = True
        #get the html content from beautiful soup
        web_content = {'boxscore':self.boxscore_url, 'teamstats':self.teamstats_url}
        
        for method, url in web_content.items():
            for i in range (1,self.retries+1):
            
                time_interval = random.uniform(2,4)

                try:
                    html = requests.get(url, headers = self.headers)
                    time.sleep(time_interval)
                    if html.status_code == 404:
                        print(f'game {self.game_id} does not exist')
                        self.game_status = False
                        return False
                    
                    if method == 'boxscore':
                        self.boxscore_content = BeautifulSoup(html.content, 'html.parser')
                    elif method == 'teamstats':
                        self.teamstats_content = BeautifulSoup(html.content, 'html.parser')
                    
                except requests.exceptions.ConnectionError:
                    print(f'error on {game_id} %s; retrying')
                    continue
                else:
                    # print(f'successfully scraped {game_id} content' )
                    break

        #check if the game was canceled or not
        big_table = self.boxscore_content.find_all('div', class_ = 'Wrapper Card__Content')
        for table in big_table:
            if 'No Box Score Available' in table.text:
                print(f'game {self.game_id} does not have a boxscore')
                self.game_status = False
                return False
            
        if self.game_status:
            #get the name of the two teams that played 
            teams = self.boxscore_content.find_all("div", class_ = 'BoxscoreItem__TeamName h5')
            self.teams = []
            #main html id for the webpage
            results = self.boxscore_content.find(id='themeProvider')

            #initialize dictionary to store the team stats in 
            self.compiled_stats = {}

            #get the player names
            player_names = results.find_all('table',class_="Table Table--align-right Table--fixed Table--fixed-left")

            for table, team in zip(player_names, teams):
                players = []
                
                for row in table.find_all("tr"):
                    columns = row.find_all("td")
                    
                    for col in columns:
                        players.append(col.get_text())
                
                player_df = pd.DataFrame(players)
                player_df.columns = player_df.iloc[0]
                player_df = player_df[1:]
                self.compiled_stats[team.text] = player_df
                self.teams.append(team.text)

            #get the stats for each player
            self.player_stats = {}

            player_scores = results.find_all('table', class_ = 'Table Table--align-right')

            good_tables = [1,2] #other data is store in this class but we only want the boxscore for our game
            for idx, player_score in enumerate(player_scores):
                
                if idx not in good_tables:
                    continue
                
                data = []
                for row in player_score.find_all("tr"):
                    row_data = []
                    columns = row.find_all(["th", "td"])
                    for column in columns:
                        row_data.append(column.get_text())
                    data.append(row_data)
                
                df = pd.DataFrame(data)
                df.columns = df.iloc[0]
                df = df[1:]
                #this could probably be done better
                if idx ==1:
                    self.player_stats[teams[0].text]=df
                else:
                    self.player_stats[teams[1].text]=df

            for ps in self.player_stats:
                self.compiled_stats[ps] = pd.concat([self.compiled_stats[ps],self.player_stats[ps]],axis=1)

            #get the game information 
            game_information = results.find_all('section', class_ = 'Card GameInfo')

            for game in game_information:
                # print(game.text)
                game_locations = game.find_all('div', class_ = 'GameInfo__Location')
                game_state = game.find_all('div', class_ = 'Weather')
                game_date = game.find_all('div', class_ = 'n8 GameInfo__Meta')
                betting_line = game.find_all('div', class_ = 'n8 GameInfo__BettingItem flex-expand line')
                betting_over_under = game.find_all('div', class_ = 'n8 GameInfo__BettingItem flex-expand ou')
                attendance = game.find_all('div', class_ = 'Attendance h8')
                referees = game.find_all('li', class_ = 'GameInfo__List__Item')

                self.game_location = game_locations[0].text
                
                self.game_state = game_state[0].text
                
                date_time_str = game_date[0].text.split('Coverage:')[0].strip()
                # print(f"game date: {date_time_str}\n")
                self.game_date = pd.to_datetime(date_time_str)
                    
                if len(betting_line) >0:
                    self.betting_line = betting_line[0].text
                else:
                    self.betting_line = 'NA'

                if len(betting_over_under) >0:
                    self.over_under = betting_over_under[0].text
                else:
                    self.over_under = 'NA'

                if len(attendance)>0:
                    attendance_str = attendance[0].text.split(':')[-1].strip()
                    self.attendance = int(attendance_str.replace(',',''))
                else:
                    self.attendance = 'NA'
                
                self.referees = "" 
                for ref in referees:
                    self.referees += ref.text + ', '
                self.referees = self.referees.strip(', ')
        #get the team stats
        team_stats = self.teamstats_content.find_all('div', class_ = 'Table__Scroller')
        # print(len(team_stats))
       # team_stats[0]  gives the half time stats table at the top of the page
        tables = team_stats[1].find_all("table", class_ = 'Table Table--align-right')
        # print(len(tables))
        

        # Initialize an empty list to store the rows
        data = []

        for table in tables:
            for row in table.find_all('tr'):
                columns = row.find_all('td')
                # Create a list to store the columns in the current row
                row_data = [column.text for column in columns]
                # Append the row data to the data list
                data.append(row_data)
        # Convert the list of rows into a DataFrame
        df = pd.DataFrame(data)
        df = df.T
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        df = df.reset_index(drop=True)
        for idx, team in enumerate(self.teams):
            df.loc[idx,'Team'] = team
            df.loc[idx,'GameID'] = game_id
            df.loc[idx,'Date'] = self.game_date
            df.loc[idx,'Location'] = self.game_location
            df.loc[idx,'State'] = self.game_state
            df.loc[idx,'Betting Line'] = self.betting_line
            df.loc[idx,'Over Under'] = self.over_under
            df.loc[idx,'Attendance'] = self.attendance
            df.loc[idx,'Referees'] = self.referees

        # Now you can print the DataFrame instead of printing each column
        self.team_stats = df

        return self.team_stats
                
    def scrape_conferences(self, season, conference):

        self.conferences_url = f"https://www.espn.com/mens-college-basketball/standings/_/season/{season}/group/{conference}"
        conference_teams = []
        for i in range(1, self.retries + 1):
            time_interval = random.uniform(2, 4)

            try:
                
                html = requests.get(self.conferences_url, headers=self.headers)
                time.sleep(time_interval)
                
                content = BeautifulSoup(html.content, 'html.parser')
                tbody = content.find_all('tbody', class_ = 'Table__TBODY')
                #ESPN doesn't contain the conference info for some reason
                if len(tbody) == 0:
                    return []
                rows = tbody[0].find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    for cell in cells:
                        span = cell.find('span', class_= 'hide-mobile')
                        if span:
                            # Do something with the span
                            conference_teams.append(span.text)
                       
                return conference_teams
            
            except requests.exceptions.ConnectionError:
                print(f'error on {conference} {season} attempt {i} of {self.retries}')
                continue
            else:
                break

    def team_page(self, team_url):

        for i in range(1, self.retries + 1):
            time_interval = random.uniform(1, 3)

            try:
        
                html = requests.get(team_url, time.sleep(time_interval), headers=self.headers)
                
                content = BeautifulSoup(html.content, 'html.parser')
                
                h1_element = content.find('h1', class_='ClubhouseHeader__Name ttu flex items-start n2')

                # Find all span elements within this h1 element
                span_elements = h1_element.find_all('span')

                # Extract text from each span element
                texts = [span.get_text() for span in span_elements]

                name_1 = texts[-2]
                name_2 = texts[-1]
                
                team_name = name_1 + ' ' + name_2
                
            
            except: 
                print(f'error on {team_url} attempt {i} of {self.retries}')
                continue
            else:
                break
        return team_name

    
    def scrape_future_games(self, date):
        # future_date.strftime('%m')
        month = date.strftime('%m')
        day = date.strftime('%d')
        
        self.future_games_url = f'https://www.espn.com/mens-college-basketball/schedule/_/date/{date.year}{month}{day}'
        print(self.future_games_url)
        for i in range(1, self.retries + 1):
            time_interval = random.uniform(1, 3)

            try:
                home_team = []
                away_team = []
                game_location = []
                over_under = []
                
                html = requests.get(self.future_games_url, headers=self.headers)
                
                content = BeautifulSoup(html.content, 'html.parser')
                
                tables = content.find_all('tbody', class_='Table__TBODY')
                
                rows = tables[0].find_all('tr')
                for idx, row in enumerate(rows):
                    # if idx >1:
                    #     continue
                    print(f"scraping game {idx+1}")
                    cells = row.find_all('td', class_='events__col Table__TD')
                    if cells[0].find('a', class_='AnchorLink') == None:
                        continue
                    else:
                        away_team_href = cells[0].find('a', class_='AnchorLink').get('href')
                    # print(away_team_href)
                    away_team_url = f'https://www.espn.com{away_team_href}'
                    
                    away_team.append(self.team_page(away_team_url))
                    # probably going to have to use this url to lookup the team name

                    at_cell = row.find_all('td', class_='colspan__col Table__TD')
                    if at_cell[0].find('a', class_='AnchorLink') == None:
                        continue
                    else:
                        home_team_href = at_cell[0].find('a', class_='AnchorLink').get('href')
                    home_team_url = f'https://www.espn.com{home_team_href}'
                    
                    home_team.append(self.team_page(home_team_url))

                    game_loc_cell = row.find_all('td', class_='venue__col Table__TD')
                    game_loc = game_loc_cell[0].text.split(',')[-2:]
                    game_loc_str = ','.join(game_loc)
                    game_location.append(game_loc_str)
                
                    odds_cell = row.find('div', class_='db')
                    if odds_cell:
                        espn_over_under = float(odds_cell.text.split(":")[-1].strip())
                        over_under.append(espn_over_under)
                    else:
                        over_under.append('None')

                self.future_df = pd.DataFrame({'away_team': away_team, 'home_team': home_team, 'location': game_location, 'over_under': over_under})

            except requests.exceptions.ConnectionError:
                print(f'error getting games for {date.year}-{date.month}-{date.day} attempt {i} of {self.retries}')
                continue
            else:
                break

if __name__ == '__main__':
    scraper = Scraper()
    test_date = date(2024, 1, 6)
    scraper.get_list_of_games(test_date)
    # print(scraper.game_schedule)
    
    # print(scraper.game_ids)

    game_id = scraper.game_ids[0]

    scraper.scrape_teamstats(game_id)

    print(scraper.game_status)

    print(f"teams playing: {scraper.teams}")

    # # Setup SQLAlchemy
    # engine = create_engine('sqlite:///game_ids.db')
    # Base = declarative_base()

    # # Define the GameID model
    # class GameID(Base):
    #     __tablename__ = 'game_ids'
    #     id = Column(Integer, Sequence('game_id_seq'), primary_key=True)
    #     game_id = Column(String(50))

    
    # # Create the table
    # Base.metadata.create_all(engine)

    # # Create a session
    # Session = sessionmaker(bind=engine)
    # session = Session()

    # # Assuming scraper.game_ids is a list of game IDs
    # game_ids = scraper.game_ids

    # # Store values in the database
    # for gid in game_ids:
    #     game_id_entry = GameID(game_id=gid)
    #     session.add(game_id_entry)

    # # Commit the session to save the entries
    # session.commit()


    # # Print confirmation
    # print(f"Stored {len(game_ids)} game IDs in the database.")

    # stored_game_ids = session.query(GameID).all()

    

    # for game_id_entry in stored_game_ids:
    #     print(game_id_entry.game_id)