from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
import pandas as pd
import time 

from webscraping.web_scrapper import Scraper
from database.database import *

engine = create_engine('sqlite:///database/ncaa_basketball.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

csv_info_file = r'ncaa_info.csv'

data = pd.read_csv(csv_info_file)

for idx, row in data.iterrows():
    # if idx > 3:
    #     continue

    db_team = session.query(Teams).filter(Teams.name == row['name']).first()
    db_team.location = row['address']
    # session.commit()