import dash
from dash.dependencies import Input, Output
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from pandas_datareader import data as web
from lstm import build_model
from datetime import datetime as dt

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}
# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}
sidebar = html.Div(
    id="sidebar",
    children=[
        html.H5("FYP ", className="display-4"),
        html.H5("Ho Yin Tse", className="display-4"),
        html.Hr(),
        html.P(
            "Interim Findngs", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Predictive Model Comparison", href="/", active="exact"),
                dbc.NavLink("Trade Simulation", href="/page-1", active="exact"),
                dbc.NavLink("To Be Added", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)
content = html.Div(id="page-content", children=[dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'HSBC', 'value': '0005.hk'},
            {'label': 'Tesla', 'value': 'TSLA'},
            {'label': 'Apple', 'value': 'AAPL'}
        ],
        value='0005.hk'
    ),
    dcc.Graph(id="my-graph", figure=build_model())],
    style=CONTENT_STYLE)
app.layout = html.Div([
    sidebar,
    content
])




@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
                html.H1('Kindergarten in Iran',
                        style={'textAlign':'center'}),
                dcc.Graph(id='bargraph',
                         figure=px.bar(df, barmode='group', x='Years',
                         y=['Girls Kindergarten', 'Boys Kindergarten']))
                ]
    elif pathname == "/page-1":
        return [
                html.H1('Grad School in Iran',
                        style={'textAlign':'center'}),
                dcc.Graph(id='bargraph',
                         figure=px.bar(df, barmode='group', x='Years',
                         y=['Girls Grade School', 'Boys Grade School']))
                ]
    elif pathname == "/page-2":
        return [
                html.H1('High School in Iran',
                        style={'textAlign':'center'}),
                dcc.Graph(id='bargraph',
                         figure=px.bar(df, barmode='group', x='Years',
                         y=['Girls High School', 'Boys High School']))
                ]
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

if __name__ == '__main__':
    app.run_server(debug=True,host = '0.0.0.0')

