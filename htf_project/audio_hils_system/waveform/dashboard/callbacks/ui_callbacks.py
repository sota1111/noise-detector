"""
UI関連のコールバック関数
"""

from dash import Input, Output, html

def register_ui_callbacks(app):
    """UI関連のコールバックを登録"""
    
    @app.callback(
        Output("tab-content", "children"),
        [Input("main-tabs", "active_tab")]
    )
    def update_tab_content(active_tab: str):
        """アクティブタブに応じてコンテンツを切り替え"""
        from ..layouts.tab_contents import (
            create_statistics_content,
            create_details_content, 
            create_waveform_content
        )
        
        if active_tab == "statistics-tab":
            return create_statistics_content()
        elif active_tab == "details-tab":
            return create_details_content()
        elif active_tab == "waveform-tab":
            return create_waveform_content()
        return html.Div("タブが選択されていません")