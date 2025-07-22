"""
ダッシュボード設定
"""

from dataclasses import dataclass
from typing import List

@dataclass
class DashboardConfig:
    """ダッシュボード設定クラス"""
    
    # サーバー設定
    HOST: str = "0.0.0.0"
    PORT: int = 8050
    DEBUG: bool = False
    
    # UI設定
    APP_TITLE: str = "音声識別評価ダッシュボード"
    BRAND_TEXT: str = "🎵 音声識別評価ダッシュボード"
    THEME: str = "BOOTSTRAP"
    
    # 更新間隔
    AUTO_REFRESH_INTERVAL: int = 30 * 1000  # 30秒
    
    # カラーテーマ
    PRIMARY_COLOR: str = "primary"
    SUCCESS_COLOR: str = "success"
    INFO_COLOR: str = "info"
    WARNING_COLOR: str = "warning"
    
    # グラフ設定
    GRAPH_TEMPLATE: str = "plotly_white"
    DEFAULT_FIGURE_HEIGHT: int = 400
    TIMELINE_FIGURE_HEIGHT: int = 600
    
    # データ設定
    MAX_HISTOGRAM_BINS: int = 20
    
    @property
    def external_stylesheets(self) -> List[str]:
        """外部スタイルシート"""
        import dash_bootstrap_components as dbc
        return [getattr(dbc.themes, self.THEME)]

# グローバル設定インスタンス
config = DashboardConfig()