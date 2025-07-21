"""
音声識別評価テスト実行エンジン
既存のaudio_hils_systemコンポーネントを活用
"""

import os
import time
from datetime import datetime
from pathlib import Path

from ..core.audio_processor import AudioProcessor
from ..core.ad3_controller import AD3Controller
from ..core.data_handler import DataHandler
from ..dataset.esc50_labels import ESC50Labels
from ..dataset.audio_selector import AudioSelector
from ..identification.dummy_identifier import DummyIdentifier


class TestEngine:
    """音声識別評価テスト実行エンジン"""
    
    def __init__(self):
        """テストエンジンを初期化"""
        self.audio_processor = AudioProcessor()
        self.ad3_controller = AD3Controller()
        self.data_handler = DataHandler()
        self.esc50_labels = ESC50Labels()
        self.audio_selector = AudioSelector()
        self.dummy_identifier = DummyIdentifier(accuracy_rate=0.8)
        
        self.is_connected = False
        self.test_session_id = None
        
    def initialize(self):
        """テストエンジンを初期化し、AD3デバイスに接続"""
        try:
            print("=== 音声識別評価システム初期化 ===")
            
            # AD3デバイス接続
            print("AD3デバイス接続中...")
            self.is_connected = self.ad3_controller.connect()
            
            if not self.is_connected:
                raise RuntimeError("AD3デバイス接続に失敗しました")
            
            print("✓ AD3デバイス接続成功")
            
            # 音声ファイル情報表示
            print("\n--- 利用可能音声ファイル情報 ---")
            self.audio_selector.print_summary()
            
            # テストセッションID生成
            self.test_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"\n✓ テストセッション開始: {self.test_session_id}")
            
            return True
            
        except Exception as e:
            print(f"× 初期化エラー: {e}")
            return False
    
    def shutdown(self):
        """テストエンジンを終了し、リソースを解放"""
        if self.is_connected:
            self.ad3_controller.disconnect()
            self.is_connected = False
        print("テストエンジン終了")
    
    def run_single_file_test(self, audio_file_path, test_duration=5, test_name=None):
        """
        単一音声ファイルのテストを実行
        
        Args:
            audio_file_path (str): 音声ファイルのパス
            test_duration (int): テスト時間（秒）
            test_name (str, optional): テスト名
            
        Returns:
            dict: テスト結果
        """
        if not self.is_connected:
            return {'success': False, 'error': 'AD3デバイスが接続されていません'}
        
        try:
            start_time = time.time()
            
            print(f"\n=== 単一ファイルテスト開始 ===")
            print(f"ファイル: {audio_file_path}")
            print(f"テスト時間: {test_duration}秒")
            
            # ファイル検証
            validation = self.audio_processor.validate_audio_file(audio_file_path)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': f"音声ファイル検証失敗: {validation['error']}",
                    'file_path': audio_file_path
                }
            
            print(f"✓ 音声ファイル検証OK")
            
            # 音声処理
            chunks, file_info = self.audio_processor.process_audio_file(audio_file_path)
            
            # テスト時間に応じてチャンクを制限
            test_chunks = chunks[:test_duration]
            print(f"✓ 音声処理完了 ({len(test_chunks)}秒分)")
            
            # AD3正規化パラメータ設定
            self.ad3_controller.set_global_normalization_params(test_chunks)
            
            # 音声出力・記録
            print(f"\n--- AD3出力・記録実行 ---")
            output_voltages = []
            input_voltages = []
            
            for idx, chunk in enumerate(test_chunks):
                print(f"{idx+1}/{len(test_chunks)}秒目処理中...")
                result = self.ad3_controller.output_waveform_and_record(chunk, idx)
                
                if result is None:
                    return {
                        'success': False,
                        'error': f'{idx+1}秒目の処理に失敗',
                        'file_path': audio_file_path,
                        'partial_results': {
                            'output_voltages': output_voltages,
                            'input_voltages': input_voltages
                        }
                    }
                
                output_voltages.append(result['output_voltage'])
                input_voltages.append(result['input_voltage'])
            
            print(f"✓ 全チャンク処理完了")
            
            # 音声識別実行
            print(f"\n--- 音声識別実行 ---")
            identification_result = self.dummy_identifier.identify_audio(
                audio_file_path, input_voltages
            )
            
            print(f"識別結果: {identification_result['predicted_category']} "
                  f"({'✓正解' if identification_result['is_correct'] else '×不正解'})")
            print(f"信頼度: {identification_result['confidence']:.3f}")
            
            # データ保存
            print(f"\n--- データ保存 ---")
            
            # ファイル名ベース作成
            audio_filename_base = self.data_handler.get_audio_filename_base(audio_file_path)
            
            # ESC-50ラベル取得
            category = self.esc50_labels.get_category_by_filename(audio_file_path)
            
            # テスト名の決定
            if test_name is None:
                test_name = f"test_{self.test_session_id}"
            
            # ラベル付きファイル名作成
            if category:
                labeled_filename_base = self.esc50_labels.create_labeled_filename(
                    audio_file_path, audio_filename_base, test_name
                )
            else:
                labeled_filename_base = f"{audio_filename_base}_{test_name}"
            
            # 保存データ準備
            save_data = {
                'output_voltage': output_voltages,
                'input_voltage': input_voltages,
                'time_start': 0.0,
                'total_duration': len(test_chunks),
                'num_chunks': len(test_chunks)
            }
            
            # 連続データ保存
            csv_path, graph_path = self.data_handler.save_continuous_data_with_label(
                save_data, labeled_filename_base
            )
            
            # 高解像度データ保存（最初の1秒）
            highres_csv, highres_graph = self.data_handler.save_first_second_highres(
                save_data, labeled_filename_base
            )
            
            execution_time = time.time() - start_time
            
            # 結果まとめ
            result = {
                'success': True,
                'file_path': audio_file_path,
                'test_name': test_name,
                'category': category,
                'session_id': self.test_session_id,
                'file_info': file_info,
                'test_duration': len(test_chunks),
                'execution_time': execution_time,
                'output_files': {
                    'csv': csv_path,
                    'graph': graph_path,
                    'highres_csv': highres_csv,
                    'highres_graph': highres_graph
                },
                'labeled_filename_base': labeled_filename_base,
                'data_summary': {
                    'chunks_processed': len(test_chunks),
                    'total_samples': sum(len(v) for v in output_voltages),
                    'sample_rate': file_info['sample_rate']
                },
                'identification_result': identification_result
            }
            
            print(f"\n✓ テスト完了 (実行時間: {execution_time:.1f}秒)")
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'テスト実行エラー: {e}',
                'file_path': audio_file_path,
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def run_multiple_files_test(self, file_paths, test_duration_per_file=5, test_name=None):
        """
        複数音声ファイルのバッチテストを実行
        
        Args:
            file_paths (list): 音声ファイルパスのリスト
            test_duration_per_file (int): ファイルあたりのテスト時間（秒）
            test_name (str, optional): テスト名
            
        Returns:
            dict: バッチテスト結果
        """
        if not self.is_connected:
            return {'success': False, 'error': 'AD3デバイスが接続されていません'}
        
        start_time = time.time()
        results = []
        
        print(f"\n=== 複数ファイルバッチテスト開始 ===")
        print(f"対象ファイル数: {len(file_paths)}")
        print(f"ファイルあたりテスト時間: {test_duration_per_file}秒")
        
        if test_name is None:
            test_name = f"batch_{self.test_session_id}"
        
        for idx, file_path in enumerate(file_paths):
            print(f"\n--- ファイル {idx+1}/{len(file_paths)}: {Path(file_path).name} ---")
            
            # 個別テスト実行
            individual_test_name = f"{test_name}_file{idx+1:03d}"
            result = self.run_single_file_test(file_path, test_duration_per_file, individual_test_name)
            results.append(result)
            
            # エラー時の処理
            if not result['success']:
                print(f"× ファイル {idx+1} 失敗: {result.get('error', '不明なエラー')}")
                # 継続するかの判定（現在は継続）
                continue
            else:
                print(f"✓ ファイル {idx+1} 完了")
        
        # バッチテスト統計
        successful_tests = [r for r in results if r['success']]
        failed_tests = [r for r in results if not r['success']]
        
        total_execution_time = time.time() - start_time
        
        # バッチ識別統計を追加
        identification_results = [r.get('identification_result') for r in successful_tests 
                                 if r.get('identification_result')]
        identification_statistics = None
        if identification_results:
            identification_statistics = self.dummy_identifier.get_statistics(identification_results)
            print(f"\n--- バッチ識別統計 ---")
            print(f"識別正解率: {identification_statistics['accuracy']*100:.1f}%")
            print(f"平均信頼度: {identification_statistics['average_confidence']:.3f}")

        batch_result = {
            'success': len(failed_tests) == 0,
            'test_name': test_name,
            'session_id': self.test_session_id,
            'total_files': len(file_paths),
            'successful_files': len(successful_tests),
            'failed_files': len(failed_tests),
            'results': results,
            'execution_time': total_execution_time,
            'statistics': {
                'total_test_duration': sum(r.get('test_duration', 0) for r in successful_tests),
                'average_execution_time': (sum(r.get('execution_time', 0) for r in successful_tests) / 
                                         len(successful_tests)) if successful_tests else 0,
                'total_samples_processed': sum(r.get('data_summary', {}).get('total_samples', 0) 
                                             for r in successful_tests)
            },
            'identification_statistics': identification_statistics
        }
        
        print(f"\n=== バッチテスト完了 ===")
        print(f"成功: {len(successful_tests)}/{len(file_paths)}ファイル")
        print(f"総実行時間: {total_execution_time:.1f}秒")
        
        return batch_result
    
    def run_category_test(self, category, max_files=None, test_duration_per_file=5):
        """
        特定カテゴリの音声ファイルをテスト
        
        Args:
            category (str): ESC-50カテゴリ名
            max_files (int, optional): 最大ファイル数（Noneの場合は全て）
            test_duration_per_file (int): ファイルあたりのテスト時間（秒）
            
        Returns:
            dict: カテゴリテスト結果
        """
        # カテゴリのファイルを取得
        category_files = self.audio_selector.get_files_by_category(category)
        
        if not category_files:
            return {
                'success': False,
                'error': f'カテゴリ "{category}" のファイルが見つかりません',
                'category': category
            }
        
        # ファイル数制限
        if max_files is not None and len(category_files) > max_files:
            category_files = category_files[:max_files]
        
        print(f"\n=== カテゴリテスト: {category} ===")
        print(f"対象ファイル数: {len(category_files)}")
        
        # バッチテスト実行
        test_name = f"category_{category}_{self.test_session_id}"
        result = self.run_multiple_files_test(category_files, test_duration_per_file, test_name)
        result['category'] = category
        
        return result
    
    def run_random_test(self, file_count=5, test_duration_per_file=5):
        """
        ランダム選択ファイルのテスト
        
        Args:
            file_count (int): テストするファイル数
            test_duration_per_file (int): ファイルあたりのテスト時間（秒）
            
        Returns:
            dict: ランダムテスト結果
        """
        # ランダムファイル選択
        random_files = self.audio_selector.select_random_files(file_count)
        
        if not random_files:
            return {
                'success': False,
                'error': 'ランダム選択用のファイルが見つかりません'
            }
        
        print(f"\n=== ランダムファイルテスト ===")
        print(f"選択ファイル数: {len(random_files)}")
        
        # バッチテスト実行
        test_name = f"random_{self.test_session_id}"
        result = self.run_multiple_files_test(random_files, test_duration_per_file, test_name)
        result['test_type'] = 'random'
        
        return result