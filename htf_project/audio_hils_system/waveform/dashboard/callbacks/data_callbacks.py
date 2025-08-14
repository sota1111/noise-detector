"""
データ関連のコールバック関数
"""

import pandas as pd
from dash import Input, Output
from datetime import datetime
from typing import Dict, List, Tuple, Any

from ..utils.data_loader import data_loader
from ..components.graphs import graph_generator

def register_data_callbacks(app):
    """データ関連のコールバックを登録"""
    
    @app.callback(
        [Output("current-data", "data"),
         Output("session-summary", "data"),
         Output("category-checklist", "options"),
         Output("category-checklist", "value"),
         Output("last-update", "children")],
        [Input("session-dropdown", "value"),
         Input("refresh-button", "n_clicks")]
    )
    def update_data(selected_session: str, refresh_clicks: int) -> Tuple:
        """データを更新"""
        if not selected_session:
            return {}, {}, [], [], f"最終更新: {datetime.now().strftime('%H:%M:%S')}"
        
        # セッション統計データを読み込み
        df = data_loader.load_session_statistics(selected_session)
        session_summary = data_loader.get_session_summary(selected_session)
        
        if df is None or df.empty:
            return {}, {}, [], [], f"最終更新: {datetime.now().strftime('%H:%M:%S')}"
        
        # カテゴリオプション
        categories = sorted(df['category'].unique().tolist())
        category_options = [{"label": cat, "value": cat} for cat in categories]
        
        return (
            df.to_dict('records'),
            session_summary,
            category_options,
            categories,  # 全カテゴリを選択状態に
            f"最終更新: {datetime.now().strftime('%H:%M:%S')}"
        )

    @app.callback(
        Output("summary-cards", "children"),
        [Input("current-data", "data"),
         Input("category-checklist", "value")]
    )
    def update_summary_cards(data: List[Dict], selected_categories: List[str]):
        """サマリーカードを更新"""
        from ..layouts.main_layout import create_summary_cards
        
        if not data:
            return create_summary_cards({})
        
        df = pd.DataFrame(data)
        
        if selected_categories:
            df = df[df['category'].isin(selected_categories)]
        
        stats = graph_generator.create_test_summary_cards(df)
        return create_summary_cards(stats)

    @app.callback(
        [Output("waveform-file-dropdown", "options"),
         Output("waveform-file-dropdown", "value")],
        [Input("current-data", "data")]
    )
    def update_waveform_file_options(data: List[Dict]) -> Tuple:
        """波形ファイル選択肢を更新"""
        if not data:
            return [], ""
        
        df = pd.DataFrame(data)
        
        # ファイル名とカテゴリのペアを作成
        file_info = df[['filename', 'category']].drop_duplicates()
        file_options = [
            {"label": f"{row['filename']} ({row['category']})", "value": row['filename']} 
            for _, row in file_info.iterrows()
        ]
        
        # 最初のファイルをデフォルト選択
        default_value = file_info.iloc[0]['filename'] if len(file_info) > 0 else ""
        
        return file_options, default_value