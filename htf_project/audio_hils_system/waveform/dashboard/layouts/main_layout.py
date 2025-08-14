"""
メインレイアウトコンポーネント
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from typing import List, Dict
from ..config import config

def create_navbar():
    """ナビゲーションバーを作成"""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("概要", href="#overview", external_link=True)),
            dbc.NavItem(dbc.NavLink("統計", href="#statistics", external_link=True)),
            dbc.NavItem(dbc.NavLink("詳細", href="#details", external_link=True)),
        ],
        brand=config.BRAND_TEXT,
        brand_href="#",
        color="primary",
        dark=True,
        fluid=True,
    )

def create_summary_cards(stats_dict: Dict):
    """サマリーカードを作成"""
    cards = [
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{stats_dict.get('total_tests', 0):,}", className=f"text-{config.PRIMARY_COLOR}"),
                html.P("総テスト数", className="card-text"),
            ])
        ], className="text-center"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{stats_dict.get('accuracy', 0):.1f}%", className=f"text-{config.SUCCESS_COLOR}"),
                html.P("識別精度", className="card-text"),
            ])
        ], className="text-center"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{stats_dict.get('avg_confidence', 0):.3f}", className=f"text-{config.INFO_COLOR}"),
                html.P("平均信頼度", className="card-text"),
            ])
        ], className="text-center"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{stats_dict.get('avg_response_time', 0):.3f}s", className=f"text-{config.WARNING_COLOR}"),
                html.P("平均応答時間", className="card-text"),
            ])
        ], className="text-center"),
    ]
    
    return dbc.Row([dbc.Col(card, width=3) for card in cards], className="mb-4")

def create_control_panel(session_options: List[Dict]):
    """コントロールパネルを作成"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("表示設定", className="card-title"),
                    html.Div([
                        html.Label("テストセッション選択:"),
                        dcc.Dropdown(
                            id="session-dropdown",
                            options=session_options,
                            value=session_options[0]["value"] if session_options else "",
                            placeholder="セッションを選択してください"
                        ),
                    ], className="mb-3"),
                    
                    html.Div([
                        html.Label("表示するカテゴリ:"),
                        dcc.Checklist(
                            id="category-checklist",
                            options=[],
                            value=[],
                            inline=True
                        ),
                    ], className="mb-3"),
                    
                    dbc.Button(
                        "データ更新", 
                        id="refresh-button", 
                        color="primary", 
                        size="sm",
                        className="me-2"
                    ),
                    
                    html.Small(
                        id="last-update",
                        className="text-muted"
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4")

def create_main_tabs():
    """メインタブを作成"""
    return dbc.Tabs([
        dbc.Tab(label="識別精度", tab_id="accuracy-tab"),
        dbc.Tab(label="混合行列", tab_id="confusion-tab"),
        dbc.Tab(label="分布分析", tab_id="distributions-tab"),
        dbc.Tab(label="波形表示", tab_id="waveform-tab"),
    ], id="main-tabs", active_tab="accuracy-tab")

def create_main_layout(session_options: List[Dict]):
    """メインレイアウトを作成"""
    return dbc.Container([
        # ナビゲーションバー
        create_navbar(),
        
        # コントロールパネル
        create_control_panel(session_options),
        
        # サマリーカード
        html.Div(id="summary-cards"),
        
        # メインコンテンツタブ
        create_main_tabs(),
        
        # タブコンテンツ
        html.Div(id="tab-content", className="mt-4"),
        
        # 自動更新用のInterval
        dcc.Interval(
            id="interval-component",
            interval=config.AUTO_REFRESH_INTERVAL,
            n_intervals=0
        ),
        
        # データ保存用
        dcc.Store(id="current-data"),
        dcc.Store(id="session-summary"),
        
    ], fluid=True)