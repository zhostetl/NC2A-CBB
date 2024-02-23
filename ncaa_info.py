import os
import glob 
import numpy as np 
import pandas as pd
from utils import web_scraper 

# url = r'https://www.ncsasports.org/division-1-colleges'
url = r'https://www.espn.com/mens-college-basketball/teams'

# Create a web_scraper object
scraper = web_scraper(url)

rows = scraper.soup.find_all('div', class_ = 'mt7')
print(len(rows))
data = pd.DataFrame(columns = ['ESPN_Name','Conference'])
counter = 0 
for row in rows:
    confernces = row.find_all('div', class_="headline headline pb4 n8 fw-heavy clr-gray-01")
    conference_name = confernces[0].text
    
    teams = row.find_all('div', class_="ContentList__Item")
    for team in teams: 
        team.find_all('div', class_ = 'p13')
        for t in team:
            tname = t.find_all('h2', class_ = "di clr-gray-01 h5")
            data.loc[counter,'ESPN_Name'] = tname[0].text
            data.loc[counter,'Conference'] = conference_name
            counter += 1

print(data)
data.to_csv('espn_info.csv', index = False)
#scrapes university names from ncasports.org
# rows = scraper.soup.find_all('div', class_ = 'container')

# data = []

# item_props = ['name','address','member']

# for idx, row in enumerate(rows):
#     # if idx >3:
#     #     continue
#     row_data = {}
#     for item in item_props:
#         content = row.find(attrs = {'itemprop': item})
#         if content:
#             row_data[item]=content.text
#     others = row.find_all('div')
#     ot = [div for div in others if not div.attrs]
#     for i, o in enumerate(ot):
#         # print(o.text)
#         if i == 0:
#             row_data['School_Type'] = o.text
#         else:
#             row_data['Division'] = o.text
        
    
#     data.append(row_data)

# df = pd.DataFrame(data)

# df.to_csv('ncaa_info.csv', index = False)
   