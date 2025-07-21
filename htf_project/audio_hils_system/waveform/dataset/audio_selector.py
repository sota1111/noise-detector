"""
音声ファイル選択モジュール
音声識別評価システム用の音声ファイル選択機能
"""

import os
import glob
import random
from pathlib import Path
from .esc50_labels import ESC50Labels


class AudioSelector:
    """音声ファイル選択クラス"""
    
    def __init__(self):
        """音声ファイル選択器を初期化"""
        self.esc50_labels = ESC50Labels()
        self.available_files = self._scan_audio_files()
        
    def _scan_audio_files(self):
        """利用可能な音声ファイルをスキャン"""
        audio_files = []
        
        # ESC-50フォルダから音声ファイルを探す
        esc50_patterns = [
            "ESC-50/audio/*.wav",
            "../ESC-50/audio/*.wav",
            "../../ESC-50/audio/*.wav"
        ]
        
        for pattern in esc50_patterns:
            files = glob.glob(pattern)
            if files:
                audio_files.extend(files)
                break
        
        # inputフォルダからも探す
        input_patterns = [
            "input/*.wav",
            "../input/*.wav"
        ]
        
        for pattern in input_patterns:
            files = glob.glob(pattern)
            audio_files.extend(files)
        
        return sorted(list(set(audio_files)))  # 重複除去とソート
    
    def get_available_files(self):
        """利用可能なファイルリストを取得"""
        return self.available_files.copy()
    
    def get_files_by_category(self, category=None):
        """カテゴリ別にファイルを取得"""
        if category is None:
            return self.available_files.copy()
        
        filtered_files = []
        for file_path in self.available_files:
            file_category = self.esc50_labels.get_category_by_filename(file_path)
            if file_category == category:
                filtered_files.append(file_path)
        
        return filtered_files
    
    def get_available_categories(self):
        """利用可能なカテゴリ一覧を取得"""
        categories = set()
        for file_path in self.available_files:
            category = self.esc50_labels.get_category_by_filename(file_path)
            if category:
                categories.add(category)
        
        return sorted(list(categories))
    
    def select_random_files(self, count=1, category=None):
        """ランダムにファイルを選択"""
        candidate_files = self.get_files_by_category(category)
        
        if not candidate_files:
            return []
        
        if count >= len(candidate_files):
            return candidate_files.copy()
        
        return random.sample(candidate_files, count)
    
    def select_files_by_pattern(self, pattern):
        """パターンに一致するファイルを選択"""
        import fnmatch
        
        matched_files = []
        for file_path in self.available_files:
            filename = Path(file_path).name
            if fnmatch.fnmatch(filename, pattern):
                matched_files.append(file_path)
        
        return matched_files
    
    def validate_file_selection(self, file_paths):
        """選択されたファイルの妥当性を検証"""
        results = []
        
        for file_path in file_paths:
            result = {
                'file_path': file_path,
                'valid': False,
                'error': None,
                'info': {}
            }
            
            if not os.path.exists(file_path):
                result['error'] = f"ファイルが存在しません: {file_path}"
            elif file_path not in self.available_files:
                result['error'] = f"利用可能ファイルリストに含まれていません: {file_path}"
            else:
                # AudioProcessorを使用して詳細検証
                from audio_processor import AudioProcessor
                validation = AudioProcessor.validate_audio_file(file_path)
                result.update(validation)
            
            results.append(result)
        
        return results
    
    def create_test_set(self, test_config):
        """テスト設定に基づいてファイルセットを作成"""
        test_set = []
        
        # カテゴリ別選択
        if 'categories' in test_config:
            for category_config in test_config['categories']:
                category = category_config.get('name')
                count = category_config.get('count', 1)
                
                files = self.select_random_files(count, category)
                test_set.extend(files)
        
        # ランダム選択
        if 'random_count' in test_config:
            random_count = test_config['random_count']
            exclude_files = set(test_set)  # 既に選択されたファイルを除外
            
            available_for_random = [f for f in self.available_files if f not in exclude_files]
            random_files = random.sample(available_for_random, 
                                       min(random_count, len(available_for_random)))
            test_set.extend(random_files)
        
        # パターン選択
        if 'patterns' in test_config:
            for pattern in test_config['patterns']:
                pattern_files = self.select_files_by_pattern(pattern)
                test_set.extend(pattern_files)
        
        # 明示的なファイル指定
        if 'explicit_files' in test_config:
            test_set.extend(test_config['explicit_files'])
        
        # 重複除去
        test_set = list(set(test_set))
        
        return test_set
    
    def get_file_info(self, file_path):
        """ファイル情報を取得"""
        info = {
            'file_path': file_path,
            'filename': Path(file_path).name,
            'exists': os.path.exists(file_path),
            'category': None,
            'size_bytes': None
        }
        
        if info['exists']:
            info['size_bytes'] = os.path.getsize(file_path)
            info['category'] = self.esc50_labels.get_category_by_filename(file_path)
        
        return info
    
    def print_summary(self):
        """選択可能なファイルの要約を表示"""
        print("=== 音声ファイル選択器サマリー ===")
        print(f"利用可能ファイル数: {len(self.available_files)}")
        
        categories = self.get_available_categories()
        print(f"利用可能カテゴリ数: {len(categories)}")
        
        if categories:
            print("カテゴリ別ファイル数:")
            for category in categories[:10]:  # 最初の10カテゴリのみ表示
                count = len(self.get_files_by_category(category))
                print(f"  {category}: {count}ファイル")
            
            if len(categories) > 10:
                print(f"  ... 他{len(categories) - 10}カテゴリ")
        
        # ファイル例を表示
        if self.available_files:
            print(f"\nファイル例（最大5件）:")
            for file_path in self.available_files[:5]:
                info = self.get_file_info(file_path)
                category_str = f" ({info['category']})" if info['category'] else ""
                print(f"  {info['filename']}{category_str}")