"""
タブコンテンツレイアウト
"""

from dash import dcc, html
import dash_bootstrap_components as dbc


def create_distributions_content():
    """分布分析タブのコンテンツ"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    dcc.Graph(id="response-time-boxplot"),
                    type="default"
                )
            ], width=6),
            dbc.Col([
                dcc.Loading(
                    dcc.Graph(id="confidence-histogram"),
                    type="default"
                )
            ], width=6),
        ]),
    ], fluid=True)

def create_accuracy_content():
    """精度バーチャートタブのコンテンツ"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    dcc.Graph(id="accuracy-bar-chart"),
                    type="default"
                )
            ], width=12),
        ]),
    ], fluid=True)

def create_confusion_content():
    """混合行列タブのコンテンツ"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    dcc.Graph(id="confusion-matrix", style={"height": "1000px"}),
                    type="default"
                )
            ], width=12),
        ]),
    ], fluid=True)


def create_waveform_content():
    """波形表示タブのコンテンツ"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H5("音声波形表示"),
                html.P("波形を表示する音声ファイルを選択してください"),
            ], width=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Label("音声ファイル選択:"),
                dcc.Dropdown(
                    id="waveform-file-dropdown",
                    options=[],
                    value="",
                    placeholder="波形ファイルを選択してください"
                ),
            ], width=6),
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