from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
import pandas as pd
import time 
import os

from webscraping.web_scrapper import Scraper
from database.database import *

t1 = time.time()


# Construct the relative path to the database file
db_folder = os.path.join(os.path.dirname(__file__))
db_name = 'ncaa_basketball.db'
db_path = os.path.join(db_folder, db_name)

# Create the SQLAlchemy engine using the constructed database path
engine = create_engine(f'sqlite:///{db_path}')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()


# engine = create_engine('sqlite:///ncaa_basketball.db')
# Base = declarative_base()
# Session = sessionmaker(bind=engine)
# session = Session()

conference_mapper = {
    'A-10':3,
    'ACC':2,
    'ASUN':46,
    'AM. EAST':1,
    'American':62,
    'Big 12':8,
    'Big East':4,
    'Big Sky':5,
    'Big South':6,
    'Big Ten':7,
    'Big West':9,
    'CAA':10,
    'CUSA':11,
    'Horizon':45,
    'Ivy':12,
    'MAAC':13,
    'MAC':14,
    'MEAC':16,
    'MVC':18,
    'Mountain West':44,
    'NEC':19,
    'OVC':20,
    'Pac-12':21,
    'Patriot':22,
    'SEC':23,
    'SWAC':26,
    'Southern':24,
    'Southland':25,
    'Summit':49,
    'Sun Belt':27,
    'WAC':30,
    'WCC':29,
}


# Add data to the tables
def init_db():

    seasons = [Season(year=year) for year in range(2015, 2025)]
    session.add_all(seasons)

    for key, value in conference_mapper.items():
        conference = Conferences(name=key, espn_group=value)
        session.add(conference)

    teams = pd.read_csv('ncaa_info.csv')

    for idx, row in teams.iterrows():
        team = Teams(name=row['name'], espn_name=row['ESPN_Name'])
        session.add(team)
    
    session.commit()

conference_page = Scraper()

#get list of seasons in the database 
seasons = session.query(Season).all()

missing_conference = {}

for season in seasons:
    
    year = season.year
    if year != 2025:
        continue
    missing_teams = []
    break_outer_loop = False

    for conf_name, conf_val in conference_mapper.items():

        if break_outer_loop:
            break

        print(f"\nanalyzing {conf_name} for {year}\n")

        #check if conference info is already in the database
        conference_check = session.query(TeamSeasonConference).join(Conferences).join(Season).filter(Season.year == year, Conferences.name == conf_name).first()
        # conference_check = session.query(Conferences).filter(Conferences.name == conf_name).first()
        if conference_check is not None:
            continue
        
        conference_info = conference_page.scrape_conferences(season = year, conference=conf_val)
        
        if len(conference_info) == 0:
            print(f"no conference info found for {conf_name} in {year}")
            if conf_name not in missing_conference:
                missing_conference[conf_name] = []
                missing_conference[conf_name].append(year)
            else:
                missing_conference[conf_name].append(year)
            print(missing_conference)
            continue
        # if len(conference_info)>0:
        #     continue

        for team in conference_info:
            team_name = session.query(Teams).filter(Teams.espn_name == team).first()

            if team_name is None:
                print(f"{team} not found in database")
                missing_teams.append(team)
                break_outer_loop = True
                break

            season_name = session.query(Season).filter(Season.year == year).first()
            conference_name = session.query(Conferences).filter(Conferences.name == conf_name).first()
            
            new_entry = TeamSeasonConference(team_id=team_name.id, season_id=season_name.id, conference_id=conference_name.id)
            # print(f"adding {team} to {conf_name} in {year}")
            session.add(new_entry)
        session.commit()

missing_conf = pd.DataFrame.from_dict(missing_conference, orient='index', columns=['year'])
missing_conf.to_csv(f'missing_conferences_{year}.csv')

t2 = time.time()
print(f"Time elapsed: {t2-t1:0.2f} seconds")

