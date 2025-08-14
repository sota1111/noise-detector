"""
グラフ関連のコールバック関数
"""

import pandas as pd
from dash import Input, Output
from typing import Dict, List, Tuple

from ..utils.data_loader import data_loader
from ..components.graphs import graph_generator

def register_graph_callbacks(app):
    """グラフ関連のコールバックを登録"""
    
    @app.callback(
        Output("accuracy-bar-chart", "figure"),
        [Input("current-data", "data"),
         Input("category-checklist", "value")]
    )
    def update_accuracy_graph(data: List[Dict], selected_categories: List[str]):
        """識別精度グラフを更新"""
        if not data:
            return graph_generator._create_empty_figure("データがありません")
        
        df = pd.DataFrame(data)
        
        if selected_categories:
            df = df[df['category'].isin(selected_categories)]
        
        return graph_generator.create_accuracy_by_category_bar(df)

    @app.callback(
        Output("confusion-matrix", "figure"),
        [Input("current-data", "data"),
         Input("category-checklist", "value")]
    )
    def update_confusion_matrix(data: List[Dict], selected_categories: List[str]):
        """混合行列を更新"""
        if not data:
            return graph_generator._create_empty_figure("データがありません")
        
        df = pd.DataFrame(data)
        
        if selected_categories:
            df = df[df['category'].isin(selected_categories)]
        
        return graph_generator.create_confusion_matrix(df)

    @app.callback(
        [Output("response-time-boxplot", "figure"),
         Output("confidence-histogram", "figure")],
        [Input("current-data", "data"),
         Input("category-checklist", "value")]
    )
    def update_distributions_graphs(data: List[Dict], selected_categories: List[str]) -> Tuple:
        """分布グラフを更新"""
        if not data:
            empty_fig = graph_generator._create_empty_figure("データがありません")
            return empty_fig, empty_fig
        
        df = pd.DataFrame(data)
        
        if selected_categories:
            df = df[df['category'].isin(selected_categories)]
        
        response_fig = graph_generator.create_response_time_boxplot(df)
        confidence_fig = graph_generator.create_confidence_histogram(df)
        
        return response_fig, confidence_fig


    @app.callback(
        Output("waveform-plot", "figure"),
        [Input("waveform-file-dropdown", "value")],
        prevent_initial_call=True
    )
    def update_waveform_plot(selected_filename: str):
        """波形プロットを更新"""
        if not selected_filename:
            return graph_generator._create_empty_figure("ファイルが選択されていません")
        
        # 選択されたファイルの波形データを取得
        waveform_data = data_loader.get_waveform_data(selected_filename)
        
        if not waveform_data:
            return graph_generator._create_empty_figure(f"波形データが見つかりません: {selected_filename}")
        
        return graph_generator.create_waveform_plot(
            waveform_data, 
            title=f"音声波形: {selected_filename}"
        )