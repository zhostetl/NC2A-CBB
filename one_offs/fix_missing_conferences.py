from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
import pandas as pd
import time 



from webscraping.web_scrapper import Scraper
from database.database import *

t1 = time.time()

engine = create_engine('sqlite:///ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

missing_file = 'missing-conferences.xlsx'

corrected = {
    'UTMUT Martin Skyhawks':'UT Martin Skyhawks',
    'SIUESIU Edwardsville Cougars':'SIU Edwardsville Cougars',
    'UABUAB Blazers':'UAB Blazers',
    'UTSAUTSA Roadrunners':'UTSA Roadrunners',
    'UTEPUTEP Miners':'UTEP Miners',
    'UTAUT Arlington Mavericks':'UT Arlington Mavericks',
    'ULMUL Monroe Warhawks':'UL Monroe Warhawks',
    'UNCAUNC Asheville Bulldogs':'UNC Asheville Bulldogs',
}

df = pd.read_excel(missing_file)

for idx, row in df.iterrows():
    year = row['Year']
    conf_name  = row['Conference']
    #find first lowercase letter in team name
    first_lower = 0
    for i, char in enumerate(row['Team']):
        if char.islower():
            first_lower = i
            break
    team = row['Team'][first_lower-1:]
    # print(team)
    team_name = session.query(Teams).filter(Teams.espn_name == team).first()
    if team_name is None:
        team = corrected[row['Team']]
        team_name = session.query(Teams).filter(Teams.espn_name == team).first()
        if team_name is None:
            print(f"Could not find team: {team}")
            continue
    season_name = session.query(Season).filter(Season.year == year).first()
    conference_name = session.query(Conferences).filter(Conferences.name == conf_name).first()
            
    new_entry = TeamSeasonConference(team_id=team_name.id, season_id=season_name.id, conference_id=conference_name.id)
#     session.add(new_entry)
# session.commit()