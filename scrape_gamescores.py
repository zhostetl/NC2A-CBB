import os
import glob
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np 
import random
import time



class scrape_games():

    def __init__(self, game_id):
        
        self.game_id = game_id
        self.boxscore_url = f'https://www.espn.com/mens-college-basketball/boxscore/_/gameId/{game_id}'
        self.teamstats_url = f'https://www.espn.com/mens-college-basketball/matchup/_/gameId/{game_id}'
        self.game_status = True

    def scrape_boxscore(self, retries = 3):

        headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36' }
        #get the html content from beautiful soup
        web_content = {'boxscore':self.boxscore_url, 'teamstats':self.teamstats_url}
        for method, url in web_content.items():
            for i in range (1,retries+1):
            
                time_interval = random.uniform(3,7)

                try:
                    html = requests.get(url, time.sleep(time_interval), headers = headers)
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

    def boxscore_data(self):
    
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
            print(f"game date: {date_time_str}\n")
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
    
    def compile_boxscore(self, write_results = False, write_path = None):
        self.game_summary = {}

        for team in self.compiled_stats:

            int_columns = ['MIN','OREB','DREB','REB','AST','STL','BLK','TO','PF','PTS']
            string_columns = ['players','FG','3PT','FT']

            index_of_bench = (self.compiled_stats[team].iloc[:, 0] == 'bench').idxmax()
            index_of_team = (self.compiled_stats[team].iloc[:, 0] == 'team').idxmax()

            self.compiled_stats[team]['Started'] = 0 
            self.compiled_stats[team].loc[:index_of_bench,'Started']=1
            # game_results[team].loc[index_of_bench:index_of_team,'Started']=0
            self.compiled_stats[team].drop(index_of_bench,inplace = True)
            self.compiled_stats[team].drop(index_of_team,inplace = True)
            self.compiled_stats[team].drop(self.compiled_stats[team].index[-1],inplace = True)
            self.compiled_stats[team] = self.compiled_stats[team].rename(columns = {'starters':'players'})
            
            for col in int_columns:
                self.compiled_stats[team][col] = self.compiled_stats[team][col].astype(int)
                
            for col in string_columns:
                self.compiled_stats[team][col] = self.compiled_stats[team][col].astype(str)
            
            self.game_summary[team]=self.compiled_stats[team]
        
        if write_results:
            bp = r'C:\Users\zhostetl\Documents\11_CBB'
            op = os.path.join(bp,'raw_playerstats',f'{self.game_id}.xlsx')
            with pd.ExcelWriter(op) as writer:
                for team in self.game_summary:
                    self.game_summary[team].to_excel(writer,sheet_name = team)

    def get_teamstats(self):
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
            df.loc[idx,'GameID'] = self.game_id
            df.loc[idx,'Date'] = self.game_date
            df.loc[idx,'Location'] = self.game_location
            df.loc[idx,'State'] = self.game_state
            df.loc[idx,'Betting Line'] = self.betting_line
            df.loc[idx,'Over Under'] = self.over_under
            df.loc[idx,'Attendance'] = self.attendance
            df.loc[idx,'Referees'] = self.referees

        # Now you can print the DataFrame instead of printing each column
        self.team_stats = df

        # df.to_excel('team_stats.xlsx')
       
    def check_game_status(self):
        #check if the game was canceled or not
        big_table = self.boxscore_content.find_all('div', class_ = 'Wrapper Card__Content')
        for table in big_table:
            if 'No Box Score Available' in table.text:
                print(f'game {self.game_id} does not have a boxscore')
                self.game_status = False
                return False
        
        


bp = r'C:\Users\zhostetl\Documents\11_CBB'

team_of_interest = 'Baylor Bears'

retries = 3

######------------------------######
###### 401577535 game canceled ######
###### 401577670 game hasn't happened yet ######

game_id_file = r'C:\Users\zhostetl\Documents\11_CBB\2022_2023_completed_game_ids.csv'
game_df = pd.read_csv(game_id_file)
game_ids = game_df['game_id'].tolist()
# game_ids = [401583794,401583795, 401577535]
# game_ids = [401576155, 401571155]
# game_id = 401575451 #start game
# game_id = 401577535 #start game when debugging
# stop_game = 401583794 #stop game
# ideas to pull from : https://github.com/lbenz730/ncaahoopR 
storage = {}
df = pd.DataFrame()

t1 = time.time()
for idx, game_id in enumerate(game_ids):
# while game_id < 401583794:
    print(f'-------\nanalyzing game {game_id}\n')
    percent_complete = (idx/len(game_ids))*100
    print(f'percent complete: {percent_complete:0.2f}\n')

    bball_game = scrape_games(game_id = game_id )
    bball_game.scrape_boxscore(retries = retries)

    if bball_game.game_status == False:
        game_id +=1
        continue
    
    bball_game.check_game_status()
    if bball_game.game_status == False:
        game_id +=1
        continue

    bball_game.boxscore_data()
    # bball_game.compile_boxscore()
    month_name = bball_game.game_date.month_name()
    # print(month_name)
    # if bball_game.game_date.year ==2023:
    #     print(f'year: {bball_game.game_date.year}')
    print(f"teams playing: {bball_game.teams}\n")

    if len(bball_game.teams) !=2:
        print(f'game {game_id} does not have two teams')
        game_id +=1
        continue

    bball_game.get_teamstats()
    # print(bball_game.team_stats)
    
    df = pd.concat([df,bball_game.team_stats],axis = 0).reset_index(drop=True)
    # storage[game_id] = bball_game.team_stats
    game_id +=1
    t2 = time.time()-t1
    if t2 > 60 and t2 < 3600:
        t2 = t2/60
        print(f'time elapsed since start: {t2:0.2f} minutes\n------------------\n')
    elif t2 > 3600:
        t2 = t2/3600
        print(f'time elapsed since start: {t2:0.2f} hours\n------------------\n')
    else:
        print(f'time elapsed since start: {t2:0.2f} seconds\n------------------\n')
    


print(df)
df.to_excel('team_stats.xlsx')

elapsed_time = time.time()-t1
if elapsed_time > 60:
    elapsed_time = elapsed_time/60
    print(f'time elapsed: {elapsed_time:0.2f} minutes')
elif elapsed_time > 3600:
    elapsed_time = elapsed_time/3600
    print(f'time elapsed: {elapsed_time:0.2f} hours')


# compiled_df = pd.DataFrame.from_dict(storage, orient = 'columns')
# print(compiled_df)

    
