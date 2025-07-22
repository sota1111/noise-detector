"""
タブコンテンツレイアウト
"""

from dash import dcc, html
import dash_bootstrap_components as dbc

def create_statistics_content():
    """統計分析タブのコンテンツ"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    dcc.Graph(id="accuracy-bar-chart"),
                    type="default"
                )
            ], width=6),
            dbc.Col([
                dcc.Loading(
                    dcc.Graph(id="confidence-histogram"),
                    type="default"
                )
            ], width=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    dcc.Graph(id="response-time-boxplot"),
                    type="default"
                )
            ], width=6),
            dbc.Col([
                dcc.Loading(
                    dcc.Graph(id="confusion-matrix"),
                    type="default"
                )
            ], width=6),
        ]),
    ], fluid=True)

def create_details_content():
    """詳細分析タブのコンテンツ"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    dcc.Graph(id="performance-timeline"),
                    type="default"
                )
            ], width=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.H5("テスト結果詳細"),
                html.Div(id="test-results-table")
            ], width=12)
        ])
    ], fluid=True)

def create_waveform_content():
    """波形表示タブのコンテンツ"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H5("音声波形表示"),
                html.P("選択されたセッションの最初の音声ファイルの波形を表示します"),
            ], width=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    dcc.Graph(id="waveform-plot"),
                    type="default"
                )
            ], width=12),
        ]),
    ], fluid=True)