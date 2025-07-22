"""
メインDashアプリケーション（リファクタリング版）
"""

import dash
import os
import sys

# 相対importのためのパス設定
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from audio_hils_system.waveform.dashboard.config import config
from audio_hils_system.waveform.dashboard.utils.data_loader import data_loader
from audio_hils_system.waveform.dashboard.layouts.main_layout import create_main_layout
from audio_hils_system.waveform.dashboard.callbacks.data_callbacks import register_data_callbacks
from audio_hils_system.waveform.dashboard.callbacks.graph_callbacks import register_graph_callbacks
from audio_hils_system.waveform.dashboard.callbacks.ui_callbacks import register_ui_callbacks

# Dashアプリケーションの初期化
app = dash.Dash(
    __name__,
    external_stylesheets=config.external_stylesheets,
    title=config.APP_TITLE
)

# レスポンシブ対応
app.config.suppress_callback_exceptions = True

def initialize_app():
    """アプリケーションを初期化"""
    # 利用可能なセッションを取得
    sessions = data_loader.get_available_sessions()
    session_options = [
        {"label": f"Session {session['session_id']}", "value": session['session_id']}
        for session in sessions
    ]
    
    if not session_options:
        session_options = [{"label": "データがありません", "value": ""}]
    
    # レイアウトを設定
    app.layout = create_main_layout(session_options)
    
    # コールバックを登録
    register_data_callbacks(app)
    register_graph_callbacks(app)
    register_ui_callbacks(app)

# アプリケーションを初期化
initialize_app()

if __name__ == "__main__":
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)