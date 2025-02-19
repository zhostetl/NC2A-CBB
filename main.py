from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

from database.database import *
from database.app_db import AppDB


db = AppDB()

# db_season = db.games_by_season(2025)

conf_db = db.session.query(Conferences).all()
conferences = [conference.name for conference in conf_db]


seasons = [2025]

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children = 'NCAA College Basketball Team Stats', style = {'textAlign': 'center'}),
    html.H2(children = 'Select a season to view game stats',style = {'textAlign': 'center'}),
    html.Div(
        dcc.Dropdown(
            id = 'conference-dropdown',
            options = [{'label': str(conference), 'value': conference} for conference in conferences],
            placeholder = 'Select a conference',
        ), style = {'width': '50%', 'margin': 'auto'}),
    
    dcc.Dropdown(
        id = 'season-dropdown',
        options = [{'label': str(season), 'value': season} for season in seasons],
        value = seasons[0], style = {'width': '50%', 'margin': 'auto'}
    ),
    html.Div(id = 'season-output'),
    # dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])


    
    ])

#######################################
### ---------- Callbacks ---------- ###
#######################################

@callback(Output('season-output', 'children'), Input('season-dropdown', 'value'), Input('conference-dropdown', 'value'))
def update_season_output(season_value,conference_value):
    season = season_value
    conference = conference_value
    print(conference)
    print(season)
    db_season = db.games_by_season(season, conference)
    
    return dash_table.DataTable(
    data=db_season.to_dict('records'),
    columns=[{"name": i, "id": i} for i in db_season.columns],
    style_table={'overflowX': 'auto', 'overflowY': 'auto'},
    style_cell={'textAlign': 'left'},
    page_size=20,  # Set the number of visible rows to 20
    fixed_rows={'headers': True}  # Keep the header fixed while scrolling
)




#######################################
### --------- Run Server ---------- ###
#######################################

if __name__ == '__main__':
    app.run(debug=True)