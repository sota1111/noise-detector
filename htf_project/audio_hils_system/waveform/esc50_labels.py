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
    
    def get_audio_info(self, audio_filename):
        """
        音声ファイル名から詳細情報を取得
        
        Args:
            audio_filename (str): 音声ファイル名（パス可）
            
        Returns:
            dict or None: 音声情報の辞書（見つからない場合はNone）
        """
        if self.meta_data is None:
            return None
        
        # ファイル名のみを抽出
        filename_only = Path(audio_filename).name
        
        # メタデータから検索
        matching_rows = self.meta_data[self.meta_data['filename'] == filename_only]
        
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            return {
                'filename': row['filename'],
                'category': row['category'],
                'target': row['target'],
                'fold': row['fold'],
                'esc10': row['esc10'],
                'src_file': row['src_file'],
                'take': row['take']
            }
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
    
    def get_all_categories(self):
        """
        データセット内の全カテゴリを取得
        
        Returns:
            list: カテゴリ名のリスト
        """
        if self.meta_data is None:
            return []
        
        return sorted(self.meta_data['category'].unique().tolist())
    
    def get_category_count(self):
        """
        各カテゴリの音声ファイル数を取得
        
        Returns:
            dict: カテゴリ名をキー、ファイル数を値とする辞書
        """
        if self.meta_data is None:
            return {}
        
        return self.meta_data['category'].value_counts().to_dict()
    
    def is_esc10_subset(self, audio_filename):
        """
        音声ファイルがESC-10サブセットに含まれるかチェック
        
        Args:
            audio_filename (str): 音声ファイル名（パス可）
            
        Returns:
            bool: ESC-10サブセットに含まれる場合True
        """
        info = self.get_audio_info(audio_filename)
        return info['esc10'] if info else False
    
    def print_dataset_info(self):
        """データセット情報を表示"""
        if self.meta_data is None:
            print("ESC-50メタデータが利用できません")
            return
        
        print("=== ESC-50データセット情報 ===")
        print(f"総ファイル数: {len(self.meta_data)}")
        print(f"カテゴリ数: {len(self.meta_data['category'].unique())}")
        print(f"ESC-10サブセット: {len(self.meta_data[self.meta_data['esc10'] == True])}ファイル")
        
        print("\nカテゴリ別ファイル数:")
        category_counts = self.get_category_count()
        for category, count in sorted(category_counts.items()):
            esc10_mark = "★" if category in self.meta_data[self.meta_data['esc10'] == True]['category'].unique() else ""
            print(f"  {category}: {count}ファイル {esc10_mark}")