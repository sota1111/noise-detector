import numpy as np
import wave
import pydwf
from scipy import signal
import time
import ctypes

# --- 設定 ---
wav_file = "1-137-A-32.wav"      # 元のWAVファイル (モノラル, 44.1kHz, 5秒間)
output_amp = 1.65                # ±1.65V (3.3V p-p)
sample_rate = 10000              # 10kHz
total_duration = 5.0             # 全体の出力時間（5秒）

# WAVファイル読み込み・全データ読み込み
with wave.open(wav_file, 'rb') as wf:
    assert wf.getnchannels() == 1, "モノラルWAVのみ対応"
    original_rate = wf.getframerate()
    total_frames = wf.getnframes()
    file_duration = total_frames / original_rate
    assert file_duration >= total_duration, "WAVファイルが5秒より短いです"
    
    # 全5秒分のデータを読み込み
    wf.setpos(0)
    all_audio_data = wf.readframes(total_frames)
    
    waveform_orig_all = np.frombuffer(all_audio_data, dtype=np.int16).astype(np.float64) / 32768.0
    waveform_orig_all = np.clip(waveform_orig_all, -1.0, 1.0)
    
    # 10kHzにリサンプリング
    num_samples_all = int(len(waveform_orig_all) * sample_rate / original_rate)
    waveform_all = signal.resample(waveform_orig_all, num_samples_all)

# 正規化：0~3.3Vの範囲になるように調整
min_val = np.min(waveform_all)
max_val = np.max(waveform_all)
if max_val > min_val:
    # -1~1の範囲を0~3.3Vに変換
    waveform_normalized = (waveform_all - min_val) / (max_val - min_val) * 3.3
else:
    waveform_normalized = np.ones_like(waveform_all) * 1.65  # 一定値の場合は中点

print(f"5秒間の波形データ準備完了:")
print(f"- サンプル数: {len(waveform_normalized)}")
print(f"- サンプリングレート: {sample_rate}Hz")
print(f"- 電圧範囲: {np.min(waveform_normalized):.3f}V ~ {np.max(waveform_normalized):.3f}V")

# Analog Discovery3デバイス接続
try:
    dwf = pydwf.DwfLibrary()
    device_count = dwf.deviceEnum.enumerateDevices()
    print(f"接続されているデバイス数: {device_count}")

    if device_count == 0:
        print("Analog Discovery3デバイスが検出されませんでした。")
        print("デバイスが接続されているか確認してください。")
        exit(1)

    # デバイスを開く - 複数の方法を試行
    hdwf = None
    
    # 方法1: deviceControl.open()
    try:
        hdwf = dwf.deviceControl.open(0)
        print("デバイス接続成功 (方法1)")
    except Exception as e1:
        print(f"方法1失敗: {e1}")
        
        # 方法2: 低レベルAPI直接使用
        try:
            hdwf = ctypes.c_int()
            result = dwf._lib.FDwfDeviceOpen(ctypes.c_int(0), ctypes.byref(hdwf))
            if result == 1:  # Success
                hdwf = hdwf.value
                print("デバイス接続成功 (方法2)")
            else:
                raise Exception("FDwfDeviceOpen failed")
        except Exception as e2:
            print(f"方法2失敗: {e2}")
            raise Exception("すべてのデバイス接続方法が失敗しました")
    
    if hdwf is None:
        raise Exception("デバイスハンドル取得失敗")
    
    # 正規化された波形を-1~1の範囲に戻す（AWGの入力用）
    awg_waveform = (waveform_normalized - 1.65) / output_amp
    
    # データを1秒ずつのチャンクに分割
    chunk_size = sample_rate  # 1秒 = 10000サンプル
    chunks = []
    for i in range(0, len(awg_waveform), chunk_size):
        chunks.append(awg_waveform[i:i+chunk_size])
    
    print(f"波形データを{len(chunks)}個のチャンクに分割しました（各{chunk_size}サンプル）")
    
    # 低レベルAPIを使用してAWGを設定
    lib = dwf._lib
    
    print("AWG設定中...")
    channel = 0
    
    # データ型変換関数
    def to_hdwf(handle):
        if isinstance(handle, int):
            return ctypes.c_int(handle)
        elif hasattr(handle, 'value'):
            return ctypes.c_int(handle.value)
        else:
            return ctypes.c_int(int(handle))
    
    hdwf_c = to_hdwf(hdwf)
    
    # AWGリセット
    result = lib.FDwfAnalogOutReset(hdwf_c, ctypes.c_int(channel))
    print(f"AWGリセット結果: {result}")
    
    # Playモード設定
    result = lib.FDwfAnalogOutNodeEnableSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_bool(True))
    print(f"ノード有効化結果: {result}")
    
    result = lib.FDwfAnalogOutNodeFunctionSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_int(31))  # funcPlay = 31
    print(f"Play機能設定結果: {result}")
    
    result = lib.FDwfAnalogOutNodeFrequencySet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(sample_rate))
    print(f"周波数設定結果: {result}")
    
    result = lib.FDwfAnalogOutNodeAmplitudeSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(output_amp))
    print(f"振幅設定結果: {result}")
    
    result = lib.FDwfAnalogOutNodeOffsetSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(1.65))
    print(f"オフセット設定結果: {result}")
    
    # 最初のチャンクをロード
    first_chunk = chunks[0]
    rgdData = (ctypes.c_double * len(first_chunk))(*first_chunk)
    result = lib.FDwfAnalogOutNodeDataSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), rgdData, ctypes.c_int(len(first_chunk)))
    print(f"初期データ設定結果: {result}")
    
    print("5秒間の連続アナログ出力を開始します...")
    print("Playモードで出力中...")
    
    # AWG出力開始
    result = lib.FDwfAnalogOutConfigure(hdwf_c, ctypes.c_int(channel), ctypes.c_bool(True))
    print(f"出力開始結果: {result}")
    
    # 残りのチャンクを順次送信
    chunk_index = 1
    start_time = time.time()
    
    while chunk_index < len(chunks):
        try:
            # バッファの空き状況をチェック
            cdDataFree = ctypes.c_int()
            cdDataLost = ctypes.c_int()
            cdDataCorrupted = ctypes.c_int()
            
            result = lib.FDwfAnalogOutNodePlayStatus(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), 
                                          ctypes.byref(cdDataFree), 
                                          ctypes.byref(cdDataLost), 
                                          ctypes.byref(cdDataCorrupted))
            
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            print(f"\r経過時間: {elapsed_time:.1f}s / {total_duration:.1f}s チャンク: {chunk_index}/{len(chunks)} バッファ空き: {cdDataFree.value}", end="", flush=True)
            
            # バッファに空きがあれば次のチャンクを送信
            if cdDataFree.value >= len(chunks[chunk_index]):
                next_chunk = chunks[chunk_index]
                rgdData = (ctypes.c_double * len(next_chunk))(*next_chunk)
                result = lib.FDwfAnalogOutNodePlayData(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), rgdData, ctypes.c_int(len(next_chunk)))
                if result == 1:
                    chunk_index += 1
                else:
                    print(f"\nデータ送信失敗: result={result}")
            
            # データロストやコラプションをチェック
            if cdDataLost.value > 0:
                print(f"\n警告: {cdDataLost.value}サンプルがロストしました")
            if cdDataCorrupted.value > 0:
                print(f"\n警告: {cdDataCorrupted.value}サンプルが破損しました")
            
        except Exception as e:
            print(f"\nチャンク送信エラー: {e}")
            break
            
        time.sleep(0.01)  # 10ms待機
    
    # すべてのデータが送信されるまで待機
    print("\n出力完了を待機中...")
    while True:
        try:
            cdDataFree = ctypes.c_int()
            cdDataLost = ctypes.c_int()
            cdDataCorrupted = ctypes.c_int()
            
            lib.FDwfAnalogOutNodePlayStatus(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), 
                                          ctypes.byref(cdDataFree), 
                                          ctypes.byref(cdDataLost), 
                                          ctypes.byref(cdDataCorrupted))
            
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            print(f"\r経過時間: {elapsed_time:.1f}s / {total_duration:.1f}s 出力完了待ち バッファ空き: {cdDataFree.value}", end="", flush=True)
            
            if elapsed_time >= total_duration + 0.5:  # 余裕を持って終了
                break
                
        except Exception as e:
            print(f"\nステータス取得エラー: {e}")
            break
            
        time.sleep(0.1)
    
    print(f"\n5秒間の連続出力が完了しました。")
    
    # AWGを停止
    try:
        lib.FDwfAnalogOutReset(hdwf_c, ctypes.c_int(channel))
        print("AWG停止完了")
    except Exception as e:
        print(f"AWG停止エラー: {e}")
    
    # デバイスを閉じる
    try:
        if isinstance(hdwf, int):
            lib.FDwfDeviceClose(ctypes.c_int(hdwf))
        else:
            dwf.deviceControl.close(hdwf)
        print("デバイス接続を終了しました。")
    except Exception as e:
        print(f"デバイスクローズエラー: {e}")

except Exception as e:
    print(f"エラーが発生しました: {e}")
    print("pydwfライブラリまたはAnalog Discovery3との通信に問題があります。")
    print("WaveFormsソフトウェアでデバイスが正常に動作するか確認してください。")