"""
1秒のみの処理テスト
実機でのAD3動作確認用
"""

import sys
import os
from audio_processor import AudioProcessor
from ad3_controller import AD3Controller
from data_handler import DataHandler
from esc50_labels import ESC50Labels
from config import DEFAULT_AUDIO_FILES


def main():
    """1秒のみの処理テスト"""
    print("=== 1秒処理テスト - AD3実機動作確認 ===")
    
    # 音声ファイルパスの決定
    if len(sys.argv) > 1:
        wav_file_path = sys.argv[1]
    else:
        # デフォルトファイルを探す
        audio_processor = AudioProcessor()
        wav_file_path = audio_processor.find_default_audio_file()
        
        if wav_file_path is None:
            print("音声ファイルが見つかりません。")
            return 1
    
    # ファイル存在確認と妥当性検証
    audio_processor = AudioProcessor()
    validation_result = audio_processor.validate_audio_file(wav_file_path)
    
    if not validation_result['valid']:
        print(f"音声ファイルエラー: {validation_result['error']}")
        return 1
    
    print(f"音声ファイル検証OK: {wav_file_path}")
    print(f"ファイル情報: {validation_result['info']}")
    
    try:
        # 音声ファイル処理（1秒分のみ）
        print(f"\n--- 音声ファイル処理開始（1秒のみ） ---")
        chunks, file_info = audio_processor.process_audio_file(wav_file_path)
        
        # 1チャンクのみを使用
        first_chunk = chunks[0]
        print(f"1秒分のデータを準備:")
        print(f"- サンプル数: {len(first_chunk)}")
        print(f"- サンプリングレート: {file_info['sample_rate']}Hz")
        print(f"- 処理時間: 1.0秒")
        
        # AD3コントローラー初期化
        print(f"\n--- AD3デバイス初期化 ---")
        ad3 = AD3Controller()
        
        # デバイス接続
        if not ad3.connect():
            print("AD3デバイス接続に失敗しました。処理を中止します。")
            return 1
        
        # データハンドラー初期化
        data_handler = DataHandler()
        
        # ESC-50ラベル機能初期化
        esc50_labels = ESC50Labels()
        
        # 1秒処理
        print(f"\n--- 1秒データ処理開始 ---")
        
        # 音声ファイル名からベース名を取得
        audio_filename_base = data_handler.get_audio_filename_base(wav_file_path)
        print(f"音声ファイルベース名: {audio_filename_base}")
        
        # ESC-50ラベル取得
        category = esc50_labels.get_category_by_filename(wav_file_path)
        if category:
            print(f"ESC-50カテゴリ: {category}")
            labeled_filename_base = esc50_labels.create_labeled_filename(
                wav_file_path, audio_filename_base, "1s_test"
            )
            print(f"ラベル付きファイル名ベース: {labeled_filename_base}")
        else:
            labeled_filename_base = f"{audio_filename_base}_1s_test"
            print(f"ファイル名ベース: {labeled_filename_base}")
        
        # 1チャンクのみを処理
        print("1秒分の出力・記録を実行中...")
        result = ad3.output_waveform_and_record(first_chunk, 0)
        
        if result is None:
            print("1秒処理に失敗しました。")
            return 1
        
        print("✓ 1秒処理成功!")
        
        # データ保存
        print(f"\n--- データ保存 ---")
        
        # 保存用データを準備
        save_data = {
            'output_voltage': result['output_voltage'],
            'input_voltage': result['input_voltage'],
            'time_start': 0.0,
            'total_duration': 1.0,
            'num_chunks': 1
        }
        
        # データ保存
        csv_path, graph_path = data_handler.save_continuous_data_with_label(
            save_data, labeled_filename_base
        )
        
        print(f"1秒テスト保存完了:")
        print(f"- CSV: {csv_path}")
        print(f"- グラフ: {graph_path}")
        
        # デバイス切断
        print(f"\n--- デバイス切断 ---")
        ad3.disconnect()
        
        print(f"\n--- 1秒テスト完了 ---")
        print("1秒分の処理が正常に完了しました！")
        
    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())