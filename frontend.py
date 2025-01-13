import threading

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from pymupdf.mupdf import metadata_keys

from graphs import get_graphs
from embedding_analysis.article_analysis import ArticleAnalysis

from util import get_topic_names, get_topic_ids

# Initialize the Dash app
app = dash.Dash(__name__)

metadata_keys = [
    "Published Dates",
    "Source",
    "Source Category",
]

# Get all topic IDs
all_topic_ids = get_topic_ids()

app.layout = html.Div([
    html.H1("Comparing Articles", style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'marginBottom': '20px'}),
    html.Div([
        html.Label('Select topics:', style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='topic-dropdown',
            options=[{'label': topic_name, 'value': topic_id} for topic_name, topic_id in zip(get_topic_names(), all_topic_ids)],
            multi=True,
            placeholder="Select topics",
            value=all_topic_ids,
            style={'marginBottom': '10px'}
        ),
        html.Label('Select clustering/coloring option:', style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='color-dropdown',
            options=[{'label': mk, 'value': mk} for mk in metadata_keys],
            placeholder="Select clustering/coloring option",
            value="Source",
            style={'marginBottom': '10px'}
        ),
        html.Label('Select projection algorithm:', style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='projection-dropdown',
            options=[
                {'label': 'PCA', 'value': 'PCA'},
                {'label': 'T-SNE', 'value': 'T-SNE'},
                {'label': 'MDS', 'value': 'MDS'}
            ],
            placeholder="Select projection algorithm",
            value="T-SNE",
            style={'marginBottom': '10px'}
        ),
        html.Label('Enter query text:', style={'fontWeight': 'bold'}),
        dcc.Input(
            id='query-input',
            type='text',
            placeholder='Enter query text',
            value='',
            style={
                'marginBottom': '10px',
                'width': '100%',
                'padding': '12px 20px',
                'fontSize': '16px',
                'border': '2px solid #ccc',
                'borderRadius': '4px',
                'boxSizing': 'border-box'
            }
        ),
        html.Label('Color by Query Similarity:', style={'fontWeight': 'bold'}),
        dcc.Checklist(
            id='color-query-toggle',
            options=[{'label': 'Color by Query Similarity', 'value': 'on'}],
            value=[],
            style={'marginBottom': '10px'}
        ),
        html.Button(
            'Submit',
            id='submit-button',
            n_clicks=0,
            style={
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'textAlign': 'center',
                'textDecoration': 'none',
                'display': 'inline-block',
                'fontSize': '16px',
                'margin': '4px 2px',
                'cursor': 'pointer',
                'borderRadius': '12px'
            }
        )
    ], style={'maxWidth': '600px', 'margin': 'auto'}),
    dcc.Loading(
        id="loading",
        type="default",
        children=[
            html.Hr(),
            html.Div(id='scatter-div', style={'marginTop': '20px'}),
            html.Div(id='graphs-container')
        ]
    )
], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})

# Define the callback to update the output based on the input
@app.callback(
    [Output('scatter-div', 'children'),
     Output('graphs-container', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('topic-dropdown', 'value'),
     State('color-dropdown', 'value'),
     State('projection-dropdown', 'value'),
     State('query-input', 'value'),
     State('color-query-toggle', 'value')]
)
def update_output(n_clicks, selected_topics, selected_filter, projection_algorithm, query, color_query_value):
    if n_clicks > 0:
        result = {}
        thread = threading.Thread(target=get_graphs, args=(selected_topics, selected_filter, projection_algorithm, query, color_query_value, result))
        thread.start()
        thread.join()
        scatter = result.get('scatter', None)
        similarity_matrix = result.get('similarity', None)
        similarity_graph = result.get('similarity_graph', None)
        if scatter is not None and similarity_matrix is not None:
            return (
                dcc.Graph(figure=scatter),
                html.Div([
                    html.Hr(),
                    html.Div([
                        html.Div(id='similarity-matrix-div', children=dcc.Graph(figure=similarity_matrix),
                                 style={'flex': '1'}),
                        html.Div(style={'width': '1px', 'backgroundColor': '#ccc', 'margin': '0 10px'}),
                        html.Div(id='similarity-graph-div', children=dcc.Graph(figure=similarity_graph),
                                 style={'flex': '2'})
                    ], style={'display': 'flex', 'marginTop': '20px'}),
                ])
            )
    return html.Div(), html.Div()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)