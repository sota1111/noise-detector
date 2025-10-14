"""
音声識別評価テスト実行エンジン
既存のaudio_hils_systemコンポーネントを活用
"""

import time
from datetime import datetime
from pathlib import Path
import serial
import serial.tools.list_ports

from ..core.audio_processor import AudioProcessor
from ..core.ad3_controller import AD3Controller
from ..core.data_handler import DataHandler
from ..dataset.esc50_labels import ESC50Labels
from ..dataset.audio_selector import AudioSelector


# ESC-50の34クラス名定義（M5Stackのコードと対応）
ESC50_CLASS_NAMES = [
    "dog", "cat", "rain", "sea_waves", "crackling_fire", "crickets",
    "wind", "water_drops", "thunderstorm", "crying_baby", "sneezing",
    "clapping", "breathing", "coughing", "footsteps", "laughing",
    "brushing_teeth", "snoring", "door_wood_knock", "mouse_click",
    "keyboard_typing", "can_opening", "washing_machine", "vacuum_cleaner",
    "toilet_flush", "clock_alarm", "clock_tick", "glass_breaking",
    "door_wood_creaks", "siren", "car_horn", "engine", "train", "silent"
]


class TestEngine:
    """音声識別評価テスト実行エンジン"""

    def __init__(self):
        """テストエンジンを初期化"""
        self.audio_processor = AudioProcessor()
        self.ad3_controller = AD3Controller()
        self.data_handler = DataHandler()
        self.esc50_labels = ESC50Labels()
        self.audio_selector = AudioSelector()

        self.is_connected = False
        self.test_session_id = None
        self.serial_connection = None
        self.m5stack_port = None
        
    def find_m5stack_port(self):
        """M5Stackのシリアルポートを自動検出"""
        ports = serial.tools.list_ports.comports()

        # デバッグ: 利用可能なポートを表示
        print("利用可能なシリアルポート:")
        for p in ports:
            print(f"  {p.device}: {p.description} (製造元: {p.manufacturer})")

        for port in ports:
            # M5Stack（CP2104 USB to UART Bridge）を検出
            description = port.description if port.description else ""
            manufacturer = port.manufacturer if port.manufacturer else ""

            # CP2104, Silicon Labs, またはACM（M5Stackの可能性）を検出
            if ('CP210' in description or 'Silicon Labs' in manufacturer or
                'ACM' in port.device or 'USB Serial' in description or
                'USB Single Serial' in description):
                print(f"M5Stack候補を検出: {port.device}")
                return port.device

        return None

    def connect_m5stack(self):
        """M5Stackとのシリアル接続を確立"""
        try:
            print("\n--- M5Stackシリアル接続 ---")

            # ポート自動検出
            self.m5stack_port = self.find_m5stack_port()
            if not self.m5stack_port:
                print("× M5Stackが見つかりません")
                return False

            print(f"M5Stack検出: {self.m5stack_port}")

            # シリアル接続確立
            self.serial_connection = serial.Serial(
                port=self.m5stack_port,
                baudrate=115200,
                timeout=1.0
            )

            # バッファをクリア
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()

            print("✓ M5Stackシリアル接続成功")
            return True

        except Exception as e:
            print(f"× M5Stackシリアル接続エラー: {e}")
            return False

    def disconnect_m5stack(self):
        """M5Stackとのシリアル接続を切断"""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
                print("M5Stackシリアル接続を切断しました")
        except Exception as e:
            print(f"× M5Stackシリアル切断エラー: {e}")

    def read_m5stack_classification(self, timeout=10):
        """
        M5Stackから分類結果を読み取る

        Args:
            timeout (int): タイムアウト時間（秒）

        Returns:
            dict: 分類結果 {'class_index': int, 'db_value': int, 'class_name': str}
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            print("× シリアル接続が確立されていません")
            return None

        start_time = time.time()
        result = {'class_index': None, 'db_value': None, 'class_name': None}
        in_result_block = False

        print("M5Stackからの分類結果を待機中...")

        while time.time() - start_time < timeout:
            try:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()

                    # デバッグ出力（結果ブロック外のみ）
                    if line and not in_result_block:
                        print(f"[M5Stack] {line}")

                    # 結果ブロックの開始を検出
                    if "CLASSIFICATION_RESULT_START" in line:
                        in_result_block = True
                        result = {'class_index': None, 'db_value': None, 'class_name': None}
                        print("[M5Stack] 分類結果ブロック開始")
                        continue

                    # 結果ブロックの終了を検出
                    if "CLASSIFICATION_RESULT_END" in line:
                        in_result_block = False
                        # すべての情報が揃っているか確認
                        if all(v is not None for v in result.values()):
                            print(f"✓ 分類結果受信: クラス={result['class_index']}, dB={result['db_value']}, 名前={result['class_name']}")
                            return result
                        else:
                            print("× 分類結果が不完全です")
                        continue

                    # 結果ブロック内でデータをパース
                    if in_result_block:
                        if line.startswith("DETECTED_CLASS:"):
                            result['class_index'] = int(line.split(":")[1])
                        elif line.startswith("DB_VALUE:"):
                            result['db_value'] = int(line.split(":")[1])
                        elif line.startswith("CLASS_NAME:"):
                            result['class_name'] = line.split(":")[1]

            except Exception as e:
                print(f"× シリアル読み取りエラー: {e}")
                return None

        print(f"× タイムアウト: {timeout}秒以内に結果を受信できませんでした")
        return None

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

            # M5Stackシリアル接続
            if not self.connect_m5stack():
                print("警告: M5Stack接続に失敗しました。分類結果の取得はできません。")

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
        try:
            # M5Stackシリアル接続を切断
            self.disconnect_m5stack()

            if self.is_connected:
                self.ad3_controller.disconnect()
                self.is_connected = False
            print("テストエンジン終了")
        except Exception as e:
            print(f"× テストエンジン終了時にエラー: {e}")
            self.is_connected = False  # 強制的にフラグをリセット
    
    def run_single_file_test(self, audio_file_path, test_duration=5, use_peak_segment=False, test_name=None):
        """
        単一音声ファイルのテストを実行
        
        Args:
            audio_file_path (str): 音声ファイルのパス
            test_duration (int): テスト時間（秒）
            use_peak_segment (bool): 最大音量部分の前後0.5秒を使用するかどうか
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
            chunks, file_info = self.audio_processor.process_audio_file(audio_file_path, use_peak_segment)
            
            # ピークセグメント使用時は時間制限不要、通常時は時間制限適用
            if use_peak_segment:
                test_chunks = chunks  # 全て使用（約1秒）
                print(f"✓ 音声処理完了 ({len(test_chunks)}秒分 - ピークセグメント)")
            else:
                test_chunks = chunks[:test_duration]
                print(f"✓ 音声処理完了 ({len(test_chunks)}秒分)")
            
            # チャンクが空の場合のエラーハンドリング
            if not test_chunks:
                return {
                    'success': False,
                    'error': 'ピークセグメント処理でチャンクが生成されませんでした',
                    'file_path': audio_file_path,
                    'processing_mode': file_info.get('processing_mode', 'unknown')
                }
            
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

            # M5Stackから分類結果を取得
            m5stack_result = None
            expected_index = None
            if self.serial_connection and self.serial_connection.is_open:
                print(f"\n--- M5Stack分類結果取得 ---")

                # 正解カテゴリを取得
                category = self.esc50_labels.get_category_by_filename(audio_file_path)
                if category and category in ESC50_CLASS_NAMES:
                    expected_index = ESC50_CLASS_NAMES.index(category)
                    print(f"期待する分類: {category} (インデックス: {expected_index})")

                # M5Stackから分類結果を読み取る
                m5stack_result = self.read_m5stack_classification(timeout=15)

                if m5stack_result:
                    detected_index = m5stack_result['class_index']
                    is_match = (expected_index == detected_index) if expected_index is not None else None

                    print(f"\n--- 分類結果比較 ---")
                    print(f"送信したカテゴリ: {category} (インデックス: {expected_index})")
                    print(f"M5Stack検出結果: {m5stack_result['class_name']} (インデックス: {detected_index})")
                    print(f"一致判定: {'✓ 一致' if is_match else '× 不一致' if is_match is not None else '判定不可'}")
                else:
                    print("× M5Stackから分類結果を取得できませんでした")

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

            # M5Stack分類結果の一致判定
            classification_match = None
            if m5stack_result and expected_index is not None:
                classification_match = (expected_index == m5stack_result['class_index'])

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
                'm5stack_classification': {
                    'expected_index': expected_index,
                    'detected_index': m5stack_result['class_index'] if m5stack_result else None,
                    'detected_class_name': m5stack_result['class_name'] if m5stack_result else None,
                    'db_value': m5stack_result['db_value'] if m5stack_result else None,
                    'is_match': classification_match
                }
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
    
    def run_multiple_files_test(self, file_paths, test_duration_per_file=5, use_peak_segment=False, test_name=None):
        """
        複数音声ファイルのバッチテストを実行
        
        Args:
            file_paths (list): 音声ファイルパスのリスト
            test_duration_per_file (int): ファイルあたりのテスト時間（秒）
            use_peak_segment (bool): 最大音量部分の前後0.5秒を使用するかどうか
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
            result = self.run_single_file_test(file_path, test_duration_per_file, use_peak_segment, individual_test_name)
            results.append(result)
            
            # エラー時の処理
            if not result['success']:
                print(f"× ファイル {idx+1} 失敗: {result.get('error', '不明なエラー')}")
                # 継続するかの判定（現在は継続）
                continue
            else:
                print(f"✓ ファイル {idx+1} 完了")
            
            # 次のファイル処理前に3秒待機（最後のファイルは除く）
            if idx < len(file_paths) - 1:
                # 待機前にAWG出力を停止
                self.ad3_controller.stop_output()
                print("次のファイル処理まで3秒待機中...")
                time.sleep(3)
        
        # バッチテスト統計
        successful_tests = [r for r in results if r['success']]
        failed_tests = [r for r in results if not r['success']]

        total_execution_time = time.time() - start_time

        # M5Stack分類統計を追加
        m5stack_classifications = [r.get('m5stack_classification') for r in successful_tests
                                   if r.get('m5stack_classification')]
        m5stack_statistics = None
        if m5stack_classifications:
            # 有効な分類結果のみをカウント
            valid_classifications = [c for c in m5stack_classifications if c.get('is_match') is not None]

            if valid_classifications:
                total_valid = len(valid_classifications)
                matches = sum(1 for c in valid_classifications if c['is_match'])
                accuracy = matches / total_valid if total_valid > 0 else 0

                # dB値の統計
                db_values = [c['db_value'] for c in valid_classifications if c.get('db_value') is not None]
                avg_db = sum(db_values) / len(db_values) if db_values else 0

                m5stack_statistics = {
                    'total_tests': total_valid,
                    'matches': matches,
                    'mismatches': total_valid - matches,
                    'accuracy': accuracy,
                    'average_db': avg_db
                }

                print(f"\n--- M5Stack分類統計 ---")
                print(f"M5Stack分類正解率: {accuracy*100:.1f}% ({matches}/{total_valid})")
                print(f"平均dB値: {avg_db:.1f}dB")

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
            'm5stack_statistics': m5stack_statistics
        }
        
        print(f"\n=== バッチテスト完了 ===")
        print(f"成功: {len(successful_tests)}/{len(file_paths)}ファイル")
        print(f"総実行時間: {total_execution_time:.1f}秒")
        
        return batch_result
    
    def run_category_test(self, category, max_files=None, test_duration_per_file=5, use_peak_segment=False):
        """
        特定カテゴリの音声ファイルをテスト
        
        Args:
            category (str): ESC-50カテゴリ名
            max_files (int, optional): 最大ファイル数（Noneの場合は全て）
            test_duration_per_file (int): ファイルあたりのテスト時間（秒）
            use_peak_segment (bool): 最大音量部分の前後0.5秒を使用するかどうか
            
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
        result = self.run_multiple_files_test(category_files, test_duration_per_file, use_peak_segment, test_name)
        result['category'] = category
        
        return result
    
    def run_random_test(self, file_count=5, test_duration_per_file=5, use_peak_segment=False):
        """
        ランダム選択ファイルのテスト
        
        Args:
            file_count (int): テストするファイル数
            test_duration_per_file (int): ファイルあたりのテスト時間（秒）
            use_peak_segment (bool): 最大音量部分の前後0.5秒を使用するかどうか
            
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
        result = self.run_multiple_files_test(random_files, test_duration_per_file, use_peak_segment, test_name)
        result['test_type'] = 'random'
        
        return result