from dash import Dash, html, dcc, callback, Output, Input, dash_table, State
import pandas as pd
import numpy as np
import os
from single_prediction import predict_game
import plotly.express as px

from database.database import *
from database.app_db import AppDB

db = AppDB()

teams_db = db.session.query(Teams).all()
teams = [team.espn_name for team in teams_db]

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='2025 NCAA Basketball games', style={'textAlign': 'center'}),
    html.Div(
        dcc.Dropdown(
            id='teams-dropdown',
            options=[{'label': str(team), 'value': team} for team in teams],
            placeholder='Select a team',
        ),
    ),
    html.Div([
        dash_table.DataTable(
            id='games-table',
            data=[],
            columns=[],
            style_table={'overflowX': 'auto', 'overflowY': 'auto'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'center', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
            page_size=20,
        )
    ]),
    html.Button('Predict Games', id='predict-button', n_clicks=0),
    # html.Div(id = 'predicted-outcome'),
    html.Div([
        dcc.Graph(id='predicted-outcome')
    ],),
    
    
])

#######################################
### ---------- Callbacks ---------- ###
#######################################

@callback(
    Output('games-table', 'data'),
    Output('games-table', 'columns'),
    Input('teams-dropdown', 'value'),
)
def update_games_table(team_of_interest):
    season = 2025
    db_season = db.games_by_season(season)
    
    if team_of_interest:
        db_season = db_season[db_season['team_name'] == team_of_interest]
    
    columns = [{"name": i, "id": i} for i in db_season.columns]
    data = db_season.to_dict('records')
    
    return data, columns

@callback(
    # Output('predicted-outcome', 'children'),
    Output('predicted-outcome', 'figure'),
    Input('predict-button', 'n_clicks'),
    Input('games-table', 'active_cell'),
    State('games-table', 'data'),
)
def predict_games(n_clicks, active_cell, data):
    if active_cell:
        #print the team information from the selected cell using the active cell 
        print(data[active_cell['row']])

        team1, team2, game_location = db.matchup_data(data[active_cell['row']])

        print('\n********** Analyzing the following game **********\n')
        print(f"{team1.espn_name} vs {team2.espn_name} at {game_location.location}\n")
        
        games = []
        points = []
        num_games = np.linspace(100, 10000, 500)
        # num_games = [500,1000]
        for i in num_games:
            game_winner, win_pct, win_pts, game_loser, loser_pts, over_under, win_margin, total_pts = predict_game(team1 = team1.espn_name, team2 = team2.espn_name, game_location=game_location.location, db_session=db.session, num_games=int(i), season=2025, over_under=None)
            games.append(i)
            points.append(win_pts)
        
        fig = px.scatter(x=games, y=points, labels={'x':'Number of games', 'y':'Winning team points'}, title='Predicted winning team points')
        fig.update_layout(
            xaxis_title='Number of games',
            yaxis_title='Winning team points',
            title='Predicted winning team points',
        )

            
        print('Done!')
        # fig = px.scatter(x=num_games, y=points, labels={'x':'Number of games', 'y':'Winning team points'}, title='Predicted winning team points')
        return fig

    return Dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)