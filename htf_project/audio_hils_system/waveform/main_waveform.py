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
    """5秒のみの処理テスト"""
    print("=== 5秒処理テスト - AD3実機動作確認 ===")
    
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
        # 音声ファイル処理（5秒分）
        print(f"\n--- 音声ファイル処理開始（5秒のみ） ---")
        chunks, file_info = audio_processor.process_audio_file(wav_file_path)
        
        # 最初の5チャンクを使用（1チャンク=1秒）
        selected_chunks = chunks[:5]
        print(f"5秒分のデータを準備:")
        print(f"- チャンク数: {len(selected_chunks)}")
        print(f"- サンプリングレート: {file_info['sample_rate']}Hz")
        print(f"- 総処理時間: {len(selected_chunks)}秒")
        
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
        
        # AD3コントローラーに5秒間全体の正規化パラメータを設定
        ad3.set_global_normalization_params(selected_chunks)
        
        # 5秒分のデータ処理
        print(f"\n--- 5秒データ処理開始 ---")
        
        # 音声ファイル名からベース名を取得
        audio_filename_base = data_handler.get_audio_filename_base(wav_file_path)
        print(f"音声ファイルベース名: {audio_filename_base}")
        
        # ESC-50ラベル取得
        category = esc50_labels.get_category_by_filename(wav_file_path)
        if category:
            print(f"ESC-50カテゴリ: {category}")
            labeled_filename_base = esc50_labels.create_labeled_filename(
                wav_file_path, audio_filename_base, "5s_test"
            )
            print(f"ラベル付きファイル名ベース: {labeled_filename_base}")
        else:
            labeled_filename_base = f"{audio_filename_base}_5s_test"
            print(f"ファイル名ベース: {labeled_filename_base}")
        
        # 各チャンクを逐次処理
        output_voltages = []
        input_voltages  = []
        for idx, chunk in enumerate(selected_chunks):
            print(f"{idx+1}秒目の出力・記録を実行中...")
            result = ad3.output_waveform_and_record(chunk, idx)
            if result is None:
                print(f"{idx+1}秒処理に失敗しました。")
                return 1
            output_voltages.append(result['output_voltage'])
            input_voltages.append(result['input_voltage'])
        print("✓ 5秒処理成功!")
        
        # データ保存
        print(f"\n--- データ保存 ---")
        save_data = {
            'output_voltage': output_voltages,
            'input_voltage':  input_voltages,
            'time_start':     0.0,
            'total_duration': len(selected_chunks),
            'num_chunks':     len(selected_chunks)
        }
        csv_path, graph_path = data_handler.save_continuous_data_with_label(
            save_data, labeled_filename_base
        )
        
        print(f"5秒テスト保存完了:")
        print(f"- CSV: {csv_path}")
        print(f"- グラフ: {graph_path}")
        
        # デバイス切断
        print(f"\n--- デバイス切断 ---")
        ad3.disconnect()
        
        print(f"\n--- 5秒テスト完了 ---")
        print("5秒分の処理が正常に完了しました！")
        
    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
