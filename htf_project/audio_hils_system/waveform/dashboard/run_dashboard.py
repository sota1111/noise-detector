#!/usr/bin/env python3
"""
ダッシュボード起動スクリプト
"""

import os
import sys
import subprocess

def install_requirements():
    """必要なライブラリをインストール"""
    requirements_file = os.path.join(os.path.dirname(__file__), '..', 'dashboard_requirements.txt')
    
    if os.path.exists(requirements_file):
        print("Dashboard用ライブラリをインストール中...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        print("インストール完了!")
    else:
        print("requirements.txtが見つかりません。手動でライブラリをインストールしてください:")
        print("pip install dash dash-bootstrap-components plotly pandas")

def run_dashboard():
    """ダッシュボードを起動"""
    try:
        from app import app
        print("\n" + "="*60)
        print("🎵 音声識別評価ダッシュボードを起動中...")
        print("="*60)
        print("URL: http://localhost:8050")
        print("ブラウザでアクセスしてください")
        print("停止するにはCtrl+Cを押してください")
        print("="*60 + "\n")
        
        app.run(debug=True, host="0.0.0.0", port=8050)
        
    except ImportError as e:
        print(f"ライブラリのインポートエラー: {e}")
        print("必要なライブラリをインストールしてください:")
        install_requirements()
        print("インストール後、再度実行してください。")
        
    except Exception as e:
        print(f"ダッシュボード起動エラー: {e}")

if __name__ == "__main__":
    run_dashboard()