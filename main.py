from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

#https://www.rockmnation.com/2021/1/4/22211605/advanced-analytics-understanding-efficiency-margin-adjusted


bp = r'C:\Users\zhostetl\Documents\11_CBB\01_rawdata' 

seasons = ['2022_2023', '2023_2024']

metric_options = ["Field Goal %", "Three Point %", "Free Throw %"]

compiled_df = pd.DataFrame()
for season in seasons: 
    season_file = f'{season}_team_stats.xlsx'
    temp_file = os.path.join(bp,season_file)
    temp_df = pd.read_excel(temp_file)
    compiled_df = pd.concat([compiled_df, temp_df])

print(len(compiled_df))


# sdf = [df['Team']=='Baylor Bears']

def generate_table(dataframe):
    return html.Div(
        style = {'maxHeight':'50vh','overflowY':'scroll'},
        children = [
        html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(len(dataframe))
        ])
    ])
        ]
    )

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children = 'NCAA College Basketball Team Stats', style = {'textAlign': 'center'}),
    html.H2(children = 'Select a team to view their stats',style = {'textAlign': 'center'}),
    html.Div([
        "Team Input:", dcc.Input(id='team_input', value='Baylor Bears', type='text')
        ]),
    html.Div([
        "Metric Input:", dcc.Dropdown(id='metric_input', value='Field Goal %',
                                       options=[opt for opt in metric_options],
                                       style={'width':'50%'})
    ]),
    html.H4(children='Team data'),
    html.Div(id = 'table-container'),
    html.Div(id = 'graph-container'),
    

    ])

#######################################
### ---------- Callbacks ---------- ###
#######################################

@app.callback(
    [Output(component_id='table-container', component_property='children'),
     Output(component_id='graph-container', component_property='children')],
    [Input(component_id='team_input', component_property='value'),
     Input(component_id='metric_input', component_property='value')]
)

def update_content(selected_team, selected_metric):
    filtered_df = compiled_df[compiled_df['Team'] == selected_team]
    table = generate_table(filtered_df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[selected_metric], mode='lines+markers', name='Field Goal %'))
    
    graph = dcc.Graph(figure=fig)
    return table, graph




#######################################
### --------- Run Server ---------- ###
#######################################

if __name__ == '__main__':
    app.run(debug=True)