"""
ESC-50データセットのラベル管理モジュール
音声ファイル名から音声の種類を取得する機能
"""

import pandas as pd
import os
from pathlib import Path


class ESC50Labels:
    """ESC-50データセットのラベル管理クラス"""
    
    def __init__(self, esc50_meta_path=None):
        """
        ESC-50ラベルマネージャーを初期化
        
        Args:
            esc50_meta_path (str, optional): esc50.csvファイルのパス
        """
        self.meta_data = None
        self.esc50_meta_path = esc50_meta_path or self._find_esc50_csv()
        
        if self.esc50_meta_path and os.path.exists(self.esc50_meta_path):
            self._load_meta_data()
        else:
            print("警告: ESC-50メタデータファイルが見つかりません。ラベル機能は無効になります。")
    
    def _find_esc50_csv(self):
        """ESC-50のCSVファイルを自動検索"""
        possible_paths = [
            "ESC-50/meta/esc50.csv",
            "../ESC-50/meta/esc50.csv",
            "../../ESC-50/meta/esc50.csv",
            "esc50.csv",
            "meta/esc50.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_meta_data(self):
        """メタデータCSVを読み込み"""
        try:
            self.meta_data = pd.read_csv(self.esc50_meta_path)
            print(f"ESC-50メタデータを読み込みました: {self.esc50_meta_path}")
            print(f"登録済み音声ファイル数: {len(self.meta_data)}")
        except Exception as e:
            print(f"ESC-50メタデータ読み込みエラー: {e}")
            self.meta_data = None
    
    def get_category_by_filename(self, audio_filename):
        """
        音声ファイル名からカテゴリを取得
        
        Args:
            audio_filename (str): 音声ファイル名（パス可）
            
        Returns:
            str or None: カテゴリ名（見つからない場合はNone）
        """
        if self.meta_data is None:
            return None
        
        # ファイル名のみを抽出
        filename_only = Path(audio_filename).name
        
        # メタデータから検索
        matching_rows = self.meta_data[self.meta_data['filename'] == filename_only]
        
        if not matching_rows.empty:
            return matching_rows.iloc[0]['category']
        else:
            return None
    
    
    def create_labeled_filename(self, audio_filename, base_name=None, suffix="continuous"):
        """
        音声ファイル名とカテゴリからラベル付きファイル名を生成
        
        Args:
            audio_filename (str): 音声ファイル名（パス可）
            base_name (str, optional): ベース名（指定されない場合は音声ファイル名から生成）
            suffix (str): ファイル名の接尾辞
            
        Returns:
            str: ラベル付きファイル名
        """
        if base_name is None:
            base_name = Path(audio_filename).stem
        
        category = self.get_category_by_filename(audio_filename)
        
        if category:
            return f"{base_name}_{category}_{suffix}"
        else:
            return f"{base_name}_{suffix}"
