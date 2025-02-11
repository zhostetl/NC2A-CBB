from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading
import queue
import os
import time 

from geopy.geocoders import Nominatim
from geopy.distance import geodesic

from webscraping.web_scrapper import Scraper
from database.database import *


"""
This code looks up the game stats for a given game id and inputs the data into the database.
It relies on the game ids that are queried from the database in the 'game_id' table

TODO:
add winning and losing streaks as well as conference/non conference opponent/win streak

"""


class Matchup():
    
        def __init__(self, game_stat,session):
            self.rounding = 5
            self.game_id = game_stat.loc[0, 'GameID']
            self.date = game_stat.loc[0, 'Date'].date()
            self.game_location = game_stat.loc[0, 'Location']
            self.game_state = game_stat.loc[0, 'State']
            self.betting_line = game_stat.loc[0, 'Betting Line']
            self.over_under = game_stat.loc[0, 'Over Under']
            self.attendance = int(game_stat.loc[0, 'Attendance']) if game_stat.loc[0, 'Attendance'] != 'NA' else 0
            self.session = session
    
            #check if referees are in the data and not 
            all_refs = game_stat.loc[0, 'Referees'].strip()
            if len(all_refs) == 0:
                self.ref1 = 'NA'
                self.ref2 = 'NA'
                self.ref3 = 'NA'
            else:
                num_refs = len(game_stat.loc[0, 'Referees'].split(','))
                if num_refs == 1:
                    self.ref1 = game_stat.loc[0, 'Referees']
                    self.ref2 = 'NA'
                    self.ref3 = 'NA'
                elif num_refs == 2:
                    self.ref1 = game_stat.loc[0, 'Referees'].split(',')[0]
                    self.ref2 = game_stat.loc[0, 'Referees'].split(',')[1]
                    self.ref3 = 'NA'
                elif num_refs == 3:
                    self.ref1 = game_stat.loc[0, 'Referees'].split(',')[0]
                    self.ref2 = game_stat.loc[0, 'Referees'].split(',')[1]
                    self.ref3 = game_stat.loc[0, 'Referees'].split(',')[2]

            self.team1 = TeamStat(game_stat, 0)
            self.team2 = TeamStat(game_stat, 1)
            if self.date.month > 6:
                year_search = self.date.year + 1
            else:
                year_search = self.date.year
            #look up the foreign key for the team
            self.team1.team_id = self.session.query(Teams).filter(Teams.espn_name == self.team1.team).first().id
            self.team2.team_id = self.session.query(Teams).filter(Teams.espn_name == self.team2.team).first().id
            self.team1.season_id = self.session.query(Season).filter(Season.year == year_search).first().id
            self.team2.season_id = self.session.query(Season).filter(Season.year == year_search).first().id
            self.team1.conference = self.session.query(TeamSeasonConference).filter(TeamSeasonConference.team_id == self.team1.team_id, TeamSeasonConference.season_id == session.query(Season).filter(Season.year == year_search).first().id).first().conference_id
            self.team2.conference = self.session.query(TeamSeasonConference).filter(TeamSeasonConference.team_id == self.team2.team_id, TeamSeasonConference.season_id == session.query(Season).filter(Season.year == year_search).first().id).first().conference_id

            self.team1.offensive_efficiency = round((self.team1.points / self.team1.possessions),self.rounding)
            self.team1.defensive_efficiency = round((self.team2.points / self.team1.possessions),self.rounding)

            self.team2.offensive_efficiency = round((self.team2.points / self.team2.possessions),self.rounding)
            self.team2.defensive_efficiency = round((self.team1.points / self.team2.possessions),self.rounding)

            self.team1.pace = round((40 * (self.team1.possessions + self.team2.possessions) / 80),self.rounding)
            self.team2.pace = round((40 * (self.team2.possessions + self.team1.possessions) / 80),self.rounding)

            self.team1.win = int(self.team1.points > self.team2.points)
            self.team1.loss = 1 - self.team1.win

            self.team2.win = int(self.team2.points > self.team1.points)
            self.team2.loss = 1 - self.team2.win

            self.team1.possessions = self.team1.possessions
            self.team1.off_eff = self.team1.points / self.team1.possessions
            self.team1.def_eff = self.team2.points / self.team1.possessions

            self.team2.off_eff = self.team2.points / self.team2.possessions
            self.team2.def_eff = self.team1.points / self.team2.possessions
            self.team2.possessions = self.team2.possessions
            #FOUR FACTORS FOR OFFENSE
            self.team1.efg = round(((self.team1.fg_made + 0.5 * self.team1.three_point_made) / self.team1.fg_attempted),self.rounding) # effective field goal percentage
            self.team2.efg = round(((self.team2.fg_made + 0.5 * self.team2.three_point_made) / self.team2.fg_attempted),self.rounding)

            self.team1.tov = round((self.team1.total_turnovers) / (self.team1.fg_attempted + 0.44 * self.team1.free_throws_attempted + self.team1.total_turnovers),self.rounding) # turnover percentage
            self.team2.tov = round((self.team2.total_turnovers) / (self.team2.fg_attempted + 0.44 * self.team2.free_throws_attempted + self.team2.total_turnovers),self.rounding) # turnover percentage

            self.team1.orb = round((self.team1.offensive_rebounds / (self.team1.offensive_rebounds + self.team2.defensive_rebounds)),self.rounding) # offensive rebound percentage
            self.team2.orb = round((self.team2.offensive_rebounds / (self.team2.offensive_rebounds + self.team1.defensive_rebounds)),self.rounding)

            self.team1.ft_rate = round((self.team1.free_throws_made / self.team1.fg_attempted),self.rounding) # free throw rate
            self.team2.ft_rate = round((self.team2.free_throws_made / self.team2.fg_attempted),self.rounding)

            #FOUR FACTORS FOR DEFENSE
            self.team1.opp_efg = self.team2.efg
            self.team2.opp_efg = self.team1.efg

            self.team1.opp_tov = self.team2.tov
            self.team2.opp_tov = self.team1.tov

            self.team1.dreb_per = self.team1.defensive_rebounds / (self.team1.defensive_rebounds + self.team2.defensive_rebounds)
            self.team2.dreb_per = self.team2.defensive_rebounds / (self.team2.defensive_rebounds + self.team1.defensive_rebounds)

            self.team1.opp_ft_rate = self.team2.ft_rate
            self.team2.opp_ft_rate = self.team1.ft_rate

            #determine home and away teams + calculate distance traveled
            self.game_distance(self.team1.team, self.team2.team)

            self.team1.oppoent = self.team2.team
            self.team2.oppoent = self.team1.team

            #add game data to the database
            # self.add_game_data(team_of_interest = self.team1, opponent = self.team2)
            # self.add_game_data(team_of_interest = self.team2, opponent = self.team1)

        def add_game_data(self, team_of_interest = None, opponent = None):
            #add game data to the database
            self.session.add(Games(date = self.date, game_id = self.game_id, 
                              team_id = team_of_interest.team_id,
                              season_id = team_of_interest.season_id,
                              conference_id = team_of_interest.conference,
                              game_num = self.session.query(Games).filter(Games.season_id==team_of_interest.season_id).filter(Games.team_id == team_of_interest.team_id).count() + 1,
                              game_location = self.game_location, game_state = self.game_state,
                              opponent_id = self.session.query(Teams).filter(Teams.espn_name == opponent.team).first().id,
                              opponent_conference_id = opponent.conference,
                              two_point_field_goals_made = team_of_interest.two_point_made, two_point_field_goals_attempted = team_of_interest.two_point_attempted,
                              two_point_field_goal_percentage = team_of_interest.two_point_percentage, three_point_field_goals_made = team_of_interest.three_point_made,
                              three_point_field_goals_attempted = team_of_interest.three_point_attempted,
                              three_point_field_goal_percentage = team_of_interest.three_point_percentage,
                              field_goals_made = team_of_interest.fg_made,
                              field_goals_attemped = team_of_interest.fg_attempted, field_goal_percentage = team_of_interest.fg_percentage,
                              free_throws_made = team_of_interest.free_throws_made, free_throws_attempted = team_of_interest.free_throws_attempted,
                              free_throw_percentage = team_of_interest.free_throw_percentage, rebounds = team_of_interest.rebounds,
                              offensive_rebounds = team_of_interest.offensive_rebounds,    defensive_rebounds = team_of_interest.defensive_rebounds,
                              assists = team_of_interest.assists,
                              steals = team_of_interest.steals,
                              blocks = team_of_interest.blocks,
                              total_turnovers = team_of_interest.total_turnovers,
                              points_off_turnovers = team_of_interest.points_off_turnovers,
                              fast_break_points = team_of_interest.fast_break_points,
                              points_in_paint = team_of_interest.points_in_paint,
                              fouls = team_of_interest.fouls,
                              technical_fouls = team_of_interest.technical_fouls,
                              flagrant_fouls = team_of_interest.flagrant_fouls,
                              largest_lead = team_of_interest.largest_lead,
                              points = team_of_interest.points,
                              win = team_of_interest.win,
                              loss = team_of_interest.loss,
                              home = team_of_interest.home,
                              away = team_of_interest.away,
                              neutral = team_of_interest.neutral,
                              eFG = team_of_interest.efg,
                              TO_rate = team_of_interest.tov,
                              OREB_per = team_of_interest.orb,
                              FT_rate = team_of_interest.ft_rate,
                              opp_eFG = team_of_interest.opp_efg,
                              opp_TO_rate = team_of_interest.opp_tov,
                              DREB_per = team_of_interest.dreb_per,
                              opp_FT_rate = team_of_interest.opp_ft_rate,
                              possessions = team_of_interest.possessions,
                              offensive_efficiency = team_of_interest.offensive_efficiency,
                              defensive_efficiency = team_of_interest.defensive_efficiency,

                              pace = team_of_interest.pace,
                              distance_traveled = team_of_interest.distance,
                              attendance = self.attendance,
                              betting_line = self.betting_line,
                              over_under = self.over_under,
                              referee1 = self.ref1,
                              referee2 = self.ref2,
                              referee3 = self.ref3))
         
            # session.commit()

        def game_distance(self, team1, team2):
            #calculate the distance between the two teams
            #lookup team locations 
            team1_location = self.session.query(Teams).filter(Teams.espn_name == team1).first().location
            team2_location = self.session.query(Teams).filter(Teams.espn_name == team2).first().location
            #check if distance is already in the database
            self.team1.distance = self.distance_check(team1_location)
            self.team2.distance = self.distance_check(team2_location)

            self.team1.home = int(self.team1.distance == 0)
            self.team1.away = int(self.team1.distance != 0 and self.team2.distance == 0)
            self.team1.neutral = int(self.team1.distance != 0 and self.team2.distance != 0)

            self.team2.home = int(self.team2.distance == 0)
            self.team2.away = int(self.team2.distance != 0 and self.team1.distance == 0)
            self.team2.neutral = int(self.team1.distance != 0 and self.team2.distance != 0)

        def distance_check(self, team):
            #check if distance is already in the database
            distance = self.session.query(GameLocations).filter(GameLocations.location == self.game_state, GameLocations.team_location== team).first()
            if distance: 
                return distance.distance
            else:
                distance_traveled = self.calculate_distance(team, self.game_state)
                self.session.add(GameLocations(location = self.game_state, team_location = team, distance = distance_traveled))
                self.session.commit()
                return distance_traveled

        def summarize_game(self):
            #summarize the game data
            summary = f"Game Summary:\n"
            summary += f"Game locations: {self.game_location}, {self.game_state}\n"
            summary += f"{self.team1.team} traveled {self.team1.distance:.2f} miles\n"
            summary += f"{self.team2.team} traveled {self.team2.distance:.2f} miles\n"
            summary += f"Teams: {self.team1.team} vs {self.team2.team}\n"
            summary += f"Final Score: {self.team1.team} {self.team1.points} - {self.team2.team} {self.team2.points}\n"
            summary += f"Efficiencies:\n"
            summary += f"{self.team1.team} - Offensive Efficiency: {self.team1.offensive_efficiency:.2f}, Defensive Efficiency: {self.team1.defensive_efficiency:.2f}\n"
            summary += f"{self.team2.team} - Offensive Efficiency: {self.team2.offensive_efficiency:.2f}, Defensive Efficiency: {self.team2.defensive_efficiency:.2f}\n"
            print(summary)
        
        def get_coordinates(self, city_name):
            geolocator = Nominatim(user_agent="city_distance_calculator")
            location = geolocator.geocode(city_name, timeout=5)
            if location:
                return (location.latitude, location.longitude)
            else:
                return None

        def calculate_distance(self, city1, city2):
            coordinates1 = self.get_coordinates(city1)
            coordinates2 = self.get_coordinates(city2)

            if coordinates1 and coordinates2:
                distance = round(geodesic(coordinates1, coordinates2).miles)
                return distance
            else:
                return None
    

class TeamStat():

    def __init__(self, game_stat, row):
        self.outlier_value = 999
        self.team = game_stat.loc[row, 'Team']
        
        self.fg_made = int(game_stat.loc[row, 'FG'].split('-')[0])
        self.fg_attempted = int(game_stat.loc[row, 'FG'].split('-')[1])
        self.fg_percentage = round(float(game_stat.loc[row, 'Field Goal %']), 2)

        self.three_point_made = int(game_stat.loc[row, '3PT'].split('-')[0])
        self.three_point_attempted = int(game_stat.loc[row, '3PT'].split('-')[1])
        self.three_point_percentage = round(float(game_stat.loc[row, 'Three Point %']),2)

        self.two_point_made = self.fg_made - self.three_point_made
        self.two_point_attempted = self.fg_attempted - self.three_point_attempted
        self.two_point_percentage = round(float(self.two_point_made / self.two_point_attempted),2)
        
        self.free_throws_made = int(game_stat.loc[row, 'FT'].split('-')[0])
        self.free_throws_attempted = int(game_stat.loc[row, 'FT'].split('-')[1])
        self.free_throw_percentage = round(float(game_stat.loc[row, 'Free Throw %']),2)
        
        self.rebounds = int(game_stat.loc[row, 'Rebounds'])
        self.offensive_rebounds = int(game_stat.loc[row, 'Offensive Rebounds'])
        self.defensive_rebounds = int(game_stat.loc[row, 'Defensive Rebounds'])
        
        self.assists = int(game_stat.loc[row, 'Assists'])
        self.steals = int(game_stat.loc[row, 'Steals'])
        self.blocks = int(game_stat.loc[row, 'Blocks'])
        
        self.total_turnovers = int(game_stat.loc[row, 'Total Turnovers'])
        self.points_off_turnovers = int(game_stat.loc[row, 'Points Off Turnovers'])
        
        self.fast_break_points = int(game_stat.loc[row, 'Fast Break Points'])
        self.points_in_paint = int(game_stat.loc[row, 'Points in Paint'])
        
        self.fouls = int(game_stat.loc[row, 'Fouls'])
        self.technical_fouls = int(game_stat.loc[row, 'Technical Fouls'])
        self.flagrant_fouls = int(game_stat.loc[row, 'Flagrant Fouls'])
        #check if largest lead column exists
        if 'Largest Lead' in game_stat.columns:
            self.largest_lead = int(game_stat.loc[row, 'Largest Lead'])
        else:
            self.largest_lead = self.outlier_value

        self.points = int((self.two_point_made * 2)+ (self.three_point_made * 3) + self.free_throws_made)
        # self.win_loss = game_stat.loc[row, 'Win/Loss']
        self.possessions = 0.96 * (self.fg_attempted + self.total_turnovers + 0.44 * self.free_throws_attempted - self.offensive_rebounds)
        


def threaded_scrape(game_queue,bad_games):
        session = Session()
        while not game_queue.empty():
            try:
                scraper = Scraper()
                lock = threading.Lock()
                game = game_queue.get_nowait()
                # print(f"Scraping game {game.game_id}")
                game_stat = scraper.scrape_teamstats(game_id=game.game_id)
                game_data = Matchup(game_stat)
                # game_data.summarize_game()
                with lock:
                    game_data.add_game_data(team_of_interest = game_data.team1, opponent = game_data.team2)
                    game_data.add_game_data(team_of_interest = game_data.team2, opponent = game_data.team1)
                    session.commit()

                game_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error scraping game {game.game_id}: {e}")
                bad_games.append(game.game_id)
                session.rollback()
                game_queue.task_done()
        
        session.close()

# def main_threaded():
#     pass



def connect_db():
    # Construct the relative path to the database file
    db_folder = os.path.join(os.path.dirname(__file__), '..', 'database')
    db_name = 'ncaa_basketball.db'
    db_path = os.path.join(db_folder, db_name)

# Create the SQLAlchemy engine using the constructed database path
    engine = create_engine(f'sqlite:///{db_path}')
    Base = declarative_base()
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def single_game(gameid):
    # session = connect_db()
    scraper = Scraper()
    game_stat = scraper.scrape_teamstats(game_id = gameid)

    game_data = Matchup(game_stat)
    game_data.summarize_game()
    game_data.add_game_data(team_of_interest = game_data.team1, opponent = game_data.team2)
    game_data.add_game_data(team_of_interest = game_data.team2, opponent = game_data.team1)
    session.commit()

if __name__ == '__main__':
    # Construct the relative path to the database file
    db_folder = os.path.join(os.path.dirname(__file__), '..', 'database')
    db_name = 'ncaa_basketball.db'
    db_path = os.path.join(db_folder, db_name)

# Create the SQLAlchemy engine using the constructed database path
    engine = create_engine(f'sqlite:///{db_path}')
    Base = declarative_base()
    Session = sessionmaker(bind=engine)
    session = Session()

    # single_game(400597730) #no refs 
    # single_game(400587755) #has refs

    MAX_WORKERS = 5

    season = 2025 
    # scraper = Scraper()

    db_season = session.query(Season).filter(Season.year == season).first()

    game_ids = session.query(GameID).filter(GameID.season_id == season).all()

    current_games = session.query(Games).filter(Games.season_id == db_season.id).all()
    current_game_ids = [game.game_id for game in current_games]
    print(f"\n**********Starting Analysis for {season} season**********\n")
    print(f"Number of games: {len(game_ids)}")
    bad_games = []
    #1st hundred games
    subset = game_ids[:20]
    bad_games_csv = pd.read_csv(r'C:\Users\zhostetl\Documents\11_CBB\99_git\NC2A-CBB\bad_games_2025.csv')
    
    game_queue = queue.Queue()
    for game in game_ids:
        if game.game_id not in current_game_ids:
            if game.game_id not in bad_games_csv['GameID'].values:
                game_queue.put(game)
    
    print(game_queue.qsize())
    # print the first game id in the queue 
    

    t1 = time.time()
   
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(threaded_scrape, game_queue, bad_games) for _ in range(MAX_WORKERS)]
        
    print(f"\n**********Finished Analysis for {season} season**********\n")
    print(f"Number of bad games: {len(bad_games)} out of {len(game_ids)}")

    bad_games = pd.DataFrame(bad_games, columns = ['GameID'])
    bad_games.to_csv(f'bad_games_{season}.csv', index = False)

    t2 = time.time()
    print(f"\nTime to scrape: {t2-t1:0.2f} seconds\n")
