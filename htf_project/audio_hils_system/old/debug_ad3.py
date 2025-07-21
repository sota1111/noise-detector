"""
AD3エラーの詳細デバッグ
具体的なAPI呼び出しエラーを特定
"""

import numpy as np
import ctypes
from audio_processor import AudioProcessor
from ad3_controller import AD3Controller
from config import OUTPUT_AMP, AD3_SCOPE_RANGE, AD3_OFFSET_VOLTAGE, SAMPLE_RATE

def debug_ad3_api():
    """AD3 APIの詳細デバッグ"""
    print("=== AD3 API詳細デバッグ ===")
    
    # 音声データの準備
    audio_processor = AudioProcessor()
    wav_file_path = audio_processor.find_default_audio_file()
    
    if wav_file_path is None:
        print("音声ファイルが見つかりません")
        return
    
    chunks, file_info = audio_processor.process_audio_file(wav_file_path)
    waveform_chunk = chunks[0]  # 1秒分
    
    print(f"テストデータ:")
    print(f"- サンプル数: {len(waveform_chunk)}")
    print(f"- データ型: {type(waveform_chunk)}")
    print(f"- 要素型: {waveform_chunk.dtype}")
    print(f"- 値の範囲: {np.min(waveform_chunk):.6f} ~ {np.max(waveform_chunk):.6f}")
    
    # AD3接続
    ad3 = AD3Controller()
    if not ad3.connect():
        print("AD3接続失敗")
        return
    
    # ダミーモードは削除されたので、直接実機デバッグに進む
    
    print(f"\n--- 実機デバッグ開始 ---")
    print(f"hdwfオブジェクト型: {type(ad3.hdwf)}")
    print(f"hdwfオブジェクト: {ad3.hdwf}")
    
    try:
        # 波形正規化
        min_val = np.min(waveform_chunk)
        max_val = np.max(waveform_chunk)
        if max_val > min_val:
            waveform_normalized = (waveform_chunk - min_val) / (max_val - min_val) * 3.3
        else:
            waveform_normalized = np.ones_like(waveform_chunk) * AD3_OFFSET_VOLTAGE
        
        print(f"\n正規化後データ:")
        print(f"- 値の範囲: {np.min(waveform_normalized):.6f} ~ {np.max(waveform_normalized):.6f}")
        
        # AWG用に-1~1の範囲に変換
        awg_waveform = (waveform_normalized - AD3_OFFSET_VOLTAGE) / OUTPUT_AMP
        awg_waveform = np.clip(awg_waveform, -1.0, 1.0)
        
        print(f"AWG用データ:")
        print(f"- 値の範囲: {np.min(awg_waveform):.6f} ~ {np.max(awg_waveform):.6f}")
        print(f"- データ型: {awg_waveform.dtype}")
        
        # ライブラリとハンドル準備
        lib = ad3.dwf._lib
        channel = 0
        
        # hdwfハンドルの詳細デバッグ
        print(f"\nhdwfハンドル処理:")
        if isinstance(ad3.hdwf, int):
            hdwf_c = ctypes.c_int(ad3.hdwf)
            print(f"- 整数型ハンドル: {ad3.hdwf}")
        elif hasattr(ad3.hdwf, 'value'):
            hdwf_c = ctypes.c_int(ad3.hdwf.value)
            print(f"- value属性: {ad3.hdwf.value}")
        elif hasattr(ad3.hdwf, 'hdwf'):
            hdwf_c = ctypes.c_int(ad3.hdwf.hdwf)
            print(f"- hdwf属性: {ad3.hdwf.hdwf}")
        else:
            try:
                hdwf_c = ctypes.c_int(ad3.hdwf._hdwf)
                print(f"- _hdwf属性: {ad3.hdwf._hdwf}")
            except AttributeError:
                print(f"- 利用可能属性: {dir(ad3.hdwf)}")
                raise Exception("hdwfハンドルを取得できません")
        
        print(f"ctypes.c_int(hdwf): {hdwf_c}")
        print(f"ctypes.c_int(hdwf).value: {hdwf_c.value}")
        
        # 各API呼び出しを個別にテスト
        print(f"\n--- API呼び出しテスト ---")
        
        print("1. デバイス自動設定無効化...")
        try:
            result = lib.FDwfDeviceAutoConfigureSet(hdwf_c, ctypes.c_int(0))
            print(f"   成功: 戻り値 = {result}")
        except Exception as e:
            print(f"   失敗: {e}")
            print(f"   引数型: hdwf_c={type(hdwf_c)}, 0={type(ctypes.c_int(0))}")
        
        print("2. AWGリセット...")
        try:
            result = lib.FDwfAnalogOutReset(hdwf_c, ctypes.c_int(channel))
            print(f"   成功: 戻り値 = {result}")
        except Exception as e:
            print(f"   失敗: {e}")
            print(f"   引数型: hdwf_c={type(hdwf_c)}, channel={type(ctypes.c_int(channel))}")
        
        print("3. AWGノード有効化...")
        try:
            result = lib.FDwfAnalogOutNodeEnableSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_int(1))
            print(f"   成功: 戻り値 = {result}")
        except Exception as e:
            print(f"   失敗: {e}")
            print(f"   引数型: hdwf_c={type(hdwf_c)}, channel={type(ctypes.c_int(channel))}, node={type(ctypes.c_int(0))}, enable={type(ctypes.c_int(1))}")
        
        print("4. AWG波形関数設定...")
        try:
            result = lib.FDwfAnalogOutNodeFunctionSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_int(31))
            print(f"   成功: 戻り値 = {result}")
        except Exception as e:
            print(f"   失敗: {e}")
            print(f"   引数型: hdwf_c={type(hdwf_c)}, channel={type(ctypes.c_int(channel))}, node={type(ctypes.c_int(0))}, func={type(ctypes.c_int(31))}")
        
        print("5. データ配列作成...")
        try:
            # 小さなテストデータで試行
            test_data = awg_waveform[:100]  # 最初の100サンプルのみ
            rgdData = (ctypes.c_double * len(test_data))(*test_data)
            print(f"   データ配列作成成功: 長さ={len(test_data)}, 型={type(rgdData)}")
            print(f"   最初の5要素: {[rgdData[i] for i in range(min(5, len(test_data)))]}")
            
            print("6. AWGデータ設定...")
            result = lib.FDwfAnalogOutNodeDataSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), rgdData, ctypes.c_int(len(test_data)))
            print(f"   成功: 戻り値 = {result}")
            
        except Exception as e:
            print(f"   データ設定失敗: {e}")
            print(f"   引数型: hdwf_c={type(hdwf_c)}, channel={type(ctypes.c_int(channel))}, node={type(ctypes.c_int(0))}")
            if 'rgdData' in locals():
                print(f"   データ型: {type(rgdData)}, 長さ={type(ctypes.c_int(len(test_data)))}")
        
        print("\n✓ API呼び出しテスト完了")
        
    except Exception as e:
        print(f"\nデバッグ中にエラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        ad3.disconnect()

if __name__ == "__main__":
    debug_ad3_api()