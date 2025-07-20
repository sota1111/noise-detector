"""
メイン実行モジュール
音声ファイル処理とAD3制御を統合したメイン処理
"""

import sys
import os
from audio_processor import AudioProcessor
from ad3_controller import AD3Controller
from data_handler import DataHandler
from esc50_labels import ESC50Labels
from config import DEFAULT_AUDIO_FILES


def print_usage():
    """使用方法を表示"""
    print("使用方法: python main.py [音声ファイルパス]")
    print("または、以下のいずれかのファイルを配置してください:")
    for file_path in DEFAULT_AUDIO_FILES:
        print(f"  - {file_path}")


def main():
    """メイン処理"""
    print("=== 音声ファイル処理 & AD3アナログ入出力システム ===")
    
    # 音声ファイルパスの決定
    if len(sys.argv) > 1:
        wav_file_path = sys.argv[1]
    else:
        # デフォルトファイルを探す
        audio_processor = AudioProcessor()
        wav_file_path = audio_processor.find_default_audio_file()
        
        if wav_file_path is None:
            print("音声ファイルが見つかりません。")
            print_usage()
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
        # 音声ファイル処理
        print(f"\n--- 音声ファイル処理開始 ---")
        chunks, file_info = audio_processor.process_audio_file(wav_file_path)
        
        print(f"ファイル読み込み完了:")
        print(f"- ファイル長: {file_info['file_duration']:.1f}秒")
        print(f"- サンプル数: {file_info['total_samples']}")
        print(f"- サンプリングレート: {file_info['sample_rate']}Hz")
        print(f"- チャンク数: {file_info['num_chunks']}個")
        print(f"- チャンク長: {file_info['chunk_duration']}秒")
        
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
        
        # 連続処理
        print(f"\n--- 連続データ処理開始 ---")
        
        # 音声ファイル名からベース名を取得
        audio_filename_base = data_handler.get_audio_filename_base(wav_file_path)
        print(f"音声ファイルベース名: {audio_filename_base}")
        
        # ESC-50ラベル取得
        category = esc50_labels.get_category_by_filename(wav_file_path)
        if category:
            print(f"ESC-50カテゴリ: {category}")
            # ラベル付きファイル名を作成
            labeled_filename_base = esc50_labels.create_labeled_filename(
                wav_file_path, audio_filename_base, "continuous"
            )
            print(f"ラベル付きファイル名ベース: {labeled_filename_base}")
            
            # 音声詳細情報も表示
            audio_info = esc50_labels.get_audio_info(wav_file_path)
            if audio_info:
                esc10_mark = "ESC-10サブセット" if audio_info['esc10'] else "ESC-50のみ"
                print(f"音声情報: {esc10_mark}, Fold {audio_info['fold']}, Target {audio_info['target']}")
        else:
            print("ESC-50カテゴリ: 不明（メタデータに見つかりません）")
            labeled_filename_base = f"{audio_filename_base}_continuous"
        
        # AD3で連続出力・記録
        result_data = ad3.output_continuous_and_record(chunks)
        
        # デバイス切断
        print(f"\n--- デバイス切断 ---")
        ad3.disconnect()
        
        if result_data is not None:
            # 連続データ保存（ラベル付きファイル名で）
            print(f"\n--- 連続データ保存 ---")
            csv_path, plot_path = data_handler.save_continuous_data_with_label(result_data, labeled_filename_base)
            
            # 処理結果表示
            print(f"\n--- 処理完了 ---")
            print(f"生成されたファイル:")
            print(f"- CSV: {csv_path}")
            print(f"- グラフ: {plot_path}")
            if category:
                print(f"音声種類「{category}」として記録されました。")
            print(f"連続処理が正常に完了しました！")
            return 0
        else:
            print("連続処理が失敗しました。")
            return 1
            
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        return 1
    except Exception as e:
        print(f"\n予期しないエラーが発生しました: {e}")
        return 1


def test_without_ad3():
    """AD3なしでのテスト実行"""
    print("=== AD3なしテストモード ===")
    
    # 設定でダミーモードを有効化
    import config
    
    # ダミーファイルを作成してテスト
    import numpy as np
    import wave
    
    # テスト用ダミー音声ファイルを作成
    test_file = "test_audio.wav"
    sample_rate = 44100
    duration = 3.0  # 3秒
    frequency = 440  # A4音
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    audio_data_int = (audio_data * 32767).astype(np.int16)
    
    with wave.open(test_file, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data_int.tobytes())
    
    print(f"テスト用音声ファイルを作成しました: {test_file}")
    
    # テスト実行
    sys.argv = ['main.py', test_file]
    result = main()
    
    # テストファイル削除
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"テスト用ファイルを削除しました: {test_file}")
    
    return result


if __name__ == "__main__":
    # --test オプションでテストモード
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        exit_code = test_without_ad3()
    else:
        exit_code = main()
    
    sys.exit(exit_code)