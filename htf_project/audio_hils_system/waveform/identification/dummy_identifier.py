"""
ダミー音声識別コンポーネント
音声識別デバイスの動作をシミュレート（正解率80%）
"""

import random
import numpy as np
import time
from pathlib import Path
from ..dataset.esc50_labels import ESC50Labels


class DummyIdentifier:
    """ダミー音声識別デバイスクラス（正解率80%）"""
    
    def __init__(self, accuracy_rate=0.8):
        """
        ダミー識別器を初期化
        
        Args:
            accuracy_rate (float): 正解率（0.0-1.0）
        """
        self.accuracy_rate = accuracy_rate
        self.esc50_labels = ESC50Labels()
        self.response_delay = 0.1  # 識別応答遅延（秒）
        
        # ESC-50カテゴリ一覧（実際のESC-50データセットに基づく）
        self.esc50_categories = [
            'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow',
            'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 
            'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',
            'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps', 
            'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
            'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 
            'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking',
            'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train', 'church_bells', 
            'airplane', 'fireworks', 'hand_saw'
        ]
        
        print(f"ダミー音声識別器初期化完了（正解率: {accuracy_rate*100:.1f}%）")
    
    def identify_audio(self, audio_file_path, input_voltage_data=None):
        """
        音声ファイルまたは入力電圧データを識別
        
        Args:
            audio_file_path (str): 音声ファイルパス（正解ラベル取得用）
            input_voltage_data (np.ndarray, optional): AD3で記録した入力電圧データ
            
        Returns:
            dict: 識別結果
        """
        # 応答遅延をシミュレート
        time.sleep(self.response_delay)
        
        # 正解ラベルを取得
        true_category = self.esc50_labels.get_category_by_filename(audio_file_path)
        
        # 識別結果を決定
        if random.random() < self.accuracy_rate and true_category:
            # 正解の場合
            predicted_category = true_category
            is_correct = True
        else:
            # 不正解の場合：ランダムに他のカテゴリを選択
            if true_category and true_category in self.esc50_categories:
                # 正解カテゴリ以外からランダム選択
                available_categories = [cat for cat in self.esc50_categories if cat != true_category]
                predicted_category = random.choice(available_categories)
            else:
                # 正解カテゴリが不明な場合は完全ランダム
                predicted_category = random.choice(self.esc50_categories)
            is_correct = False
        
        # 信頼度スコアを生成（正解時は高め、不正解時は低め）
        if is_correct:
            confidence = random.uniform(0.7, 0.95)
        else:
            confidence = random.uniform(0.3, 0.8)
        
        # その他の候補も生成（上位3候補）
        other_candidates = [cat for cat in self.esc50_categories if cat != predicted_category]
        top_candidates = [predicted_category] + random.sample(other_candidates, min(2, len(other_candidates)))
        
        # 候補の信頼度を生成
        candidate_scores = {}
        remaining_confidence = 1.0 - confidence
        
        candidate_scores[predicted_category] = confidence
        for i, candidate in enumerate(top_candidates[1:]):
            score = remaining_confidence * random.uniform(0.1, 0.5) / (i + 1)
            candidate_scores[candidate] = score
            remaining_confidence -= score
        
        result = {
            'predicted_category': predicted_category,
            'true_category': true_category,
            'is_correct': is_correct,
            'confidence': confidence,
            'candidate_scores': candidate_scores,
            'top_candidates': top_candidates,
            'response_time': self.response_delay,
            'audio_file': Path(audio_file_path).name,
            'timestamp': time.time()
        }
        
        return result
    
    def identify_batch(self, audio_files, input_voltage_list=None):
        """
        複数音声ファイルの一括識別
        
        Args:
            audio_files (list): 音声ファイルパスのリスト
            input_voltage_list (list, optional): 入力電圧データのリスト
            
        Returns:
            list: 識別結果のリスト
        """
        results = []
        
        for i, audio_file in enumerate(audio_files):
            input_data = input_voltage_list[i] if input_voltage_list else None
            result = self.identify_audio(audio_file, input_data)
            results.append(result)
            
            # プログレス表示
            print(f"識別進捗: {i+1}/{len(audio_files)} - {result['predicted_category']} "
                  f"({'✓' if result['is_correct'] else '×'})")
        
        return results
    
    def get_statistics(self, identification_results):
        """
        識別結果の統計を計算
        
        Args:
            identification_results (list): 識別結果のリスト
            
        Returns:
            dict: 統計情報
        """
        if not identification_results:
            return {
                'total_count': 0,
                'correct_count': 0,
                'accuracy': 0.0,
                'average_confidence': 0.0,
                'category_accuracy': {}
            }
        
        total_count = len(identification_results)
        correct_count = sum(1 for r in identification_results if r['is_correct'])
        accuracy = correct_count / total_count
        
        # 平均信頼度
        average_confidence = np.mean([r['confidence'] for r in identification_results])
        
        # カテゴリ別正解率
        category_stats = {}
        for result in identification_results:
            true_cat = result['true_category']
            if true_cat:
                if true_cat not in category_stats:
                    category_stats[true_cat] = {'total': 0, 'correct': 0}
                category_stats[true_cat]['total'] += 1
                if result['is_correct']:
                    category_stats[true_cat]['correct'] += 1
        
        category_accuracy = {}
        for cat, stats in category_stats.items():
            category_accuracy[cat] = stats['correct'] / stats['total']
        
        return {
            'total_count': total_count,
            'correct_count': correct_count,
            'accuracy': accuracy,
            'average_confidence': average_confidence,
            'category_stats': category_stats,
            'category_accuracy': category_accuracy,
            'average_response_time': np.mean([r['response_time'] for r in identification_results])
        }
    
    def print_statistics(self, identification_results):
        """識別統計を表示"""
        stats = self.get_statistics(identification_results)
        
        print(f"\n=== ダミー識別器 統計結果 ===")
        print(f"総テスト数: {stats['total_count']}")
        print(f"正解数: {stats['correct_count']}")
        print(f"正解率: {stats['accuracy']*100:.1f}%")
        print(f"平均信頼度: {stats['average_confidence']:.3f}")
        print(f"平均応答時間: {stats['average_response_time']:.3f}秒")
        
        if stats['category_accuracy']:
            print(f"\nカテゴリ別正解率:")
            for category, accuracy in sorted(stats['category_accuracy'].items()):
                total = stats['category_stats'][category]['total']
                correct = stats['category_stats'][category]['correct']
                print(f"  {category}: {accuracy*100:.1f}% ({correct}/{total})")
    
    def simulate_device_malfunction(self, malfunction_rate=0.05):
        """
        デバイス故障をシミュレート
        
        Args:
            malfunction_rate (float): 故障率
            
        Returns:
            bool: 故障発生フラグ
        """
        if random.random() < malfunction_rate:
            print("⚠️ ダミー識別器：デバイス故障をシミュレート")
            return True
        return False
    
    def set_accuracy_rate(self, new_accuracy_rate):
        """正解率を変更"""
        self.accuracy_rate = max(0.0, min(1.0, new_accuracy_rate))
        print(f"ダミー識別器の正解率を {self.accuracy_rate*100:.1f}% に変更")
    
    def get_device_info(self):
        """デバイス情報を取得"""
        return {
            'device_type': 'Dummy Audio Identifier',
            'version': '1.0.0',
            'accuracy_rate': self.accuracy_rate,
            'supported_categories': len(self.esc50_categories),
            'response_delay': self.response_delay,
            'categories': self.esc50_categories.copy()
        }