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
        [Output("accuracy-bar-chart", "figure"),
         Output("confidence-histogram", "figure"),
         Output("response-time-boxplot", "figure"),
         Output("confusion-matrix", "figure")],
        [Input("current-data", "data"),
         Input("category-checklist", "value")]
    )
    def update_statistics_graphs(data: List[Dict], selected_categories: List[str]) -> Tuple:
        """統計グラフを更新"""
        if not data:
            empty_fig = graph_generator._create_empty_figure("データがありません")
            return empty_fig, empty_fig, empty_fig, empty_fig
        
        df = pd.DataFrame(data)
        
        if selected_categories:
            df = df[df['category'].isin(selected_categories)]
        
        accuracy_fig = graph_generator.create_accuracy_by_category_bar(df)
        confidence_fig = graph_generator.create_confidence_histogram(df)
        response_fig = graph_generator.create_response_time_boxplot(df)
        confusion_fig = graph_generator.create_confusion_matrix(df)
        
        return accuracy_fig, confidence_fig, response_fig, confusion_fig

    @app.callback(
        Output("performance-timeline", "figure"),
        [Input("interval-component", "n_intervals")]
    )
    def update_performance_timeline(n_intervals: int):
        """性能推移を更新"""
        timeline_df = data_loader.get_test_performance_over_time()
        return graph_generator.create_performance_timeline(timeline_df)

    @app.callback(
        Output("waveform-plot", "figure"),
        [Input("current-data", "data")],
        prevent_initial_call=True
    )
    def update_waveform_plot(data: List[Dict]):
        """波形プロットを更新"""
        if not data:
            return graph_generator._create_empty_figure("データがありません")
        
        # 最初のファイルの波形を表示
        df = pd.DataFrame(data)
        if len(df) > 0:
            first_file = df.iloc[0]['filename']
            waveform_data = data_loader.get_waveform_data(first_file)
            return graph_generator.create_waveform_plot(
                waveform_data, 
                title=f"音声波形: {first_file}"
            )
        
        return graph_generator._create_empty_figure("波形データがありません")