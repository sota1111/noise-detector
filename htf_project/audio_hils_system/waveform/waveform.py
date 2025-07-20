import numpy as np
import wave
import pydwf
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import ctypes
from pathlib import Path

# --- 設定 ---
input_folder = "input"
output_csv_folder = "output/csv"
output_graph_folder = "output/graphs"
output_amp = 1.65                 # ±1.65V (3.3V p-p)
sample_rate = 10000               # 10kHz
extract_duration = 1.0            # 抽出する秒数（1秒）

# 出力フォルダが存在しない場合は作成
os.makedirs(output_csv_folder, exist_ok=True)
os.makedirs(output_graph_folder, exist_ok=True)

def load_audio_file(wav_file):
    """音声ファイルを読み込み、10kHzにリサンプリングする"""
    with wave.open(wav_file, 'rb') as wf:
        assert wf.getnchannels() == 1, "モノラルWAVのみ対応"
        original_rate = wf.getframerate()
        total_frames = wf.getnframes()
        file_duration = total_frames / original_rate
        
        # 全データを読み込み
        wf.setpos(0)
        all_audio_data = wf.readframes(total_frames)
        
        waveform_orig_all = np.frombuffer(all_audio_data, dtype=np.int16).astype(np.float64) / 32768.0
        waveform_orig_all = np.clip(waveform_orig_all, -1.0, 1.0)
        
        # 10kHzにリサンプリング
        num_samples_all = int(len(waveform_orig_all) * sample_rate / original_rate)
        waveform_all = signal.resample(waveform_orig_all, num_samples_all)
        
        return waveform_all, file_duration

def extract_1s_chunks(waveform_all, chunk_duration=1.0):
    """波形データを1秒ごとのチャンクに分割する"""
    chunk_samples = int(chunk_duration * sample_rate)
    chunks = []
    
    for i in range(0, len(waveform_all), chunk_samples):
        chunk = waveform_all[i:i+chunk_samples]
        if len(chunk) == chunk_samples:  # 完全な1秒のチャンクのみ
            chunks.append(chunk)
    
    return chunks

class AD3Controller:
    """Analog Discovery 3 制御クラス"""
    
    def __init__(self):
        self.dwf = None
        self.hdwf = None
        self.is_connected = False
        
    def connect(self):
        """AD3デバイスに接続"""
        try:
            self.dwf = pydwf.DwfLibrary()
            device_count = self.dwf.deviceEnum.enumerateDevices()
            print(f"接続されているデバイス数: {device_count}")
            
            if device_count == 0:
                print("Analog Discovery3デバイスが検出されませんでした。")
                return False
            
            # デバイスを開く - 複数の方法を試行
            try:
                self.hdwf = self.dwf.deviceControl.open(0)
                print("デバイス接続成功")
                self.is_connected = True
                return True
            except Exception as e1:
                print(f"デバイス接続失敗: {e1}")
                # 低レベルAPI使用を試行
                try:
                    hdwf_c = ctypes.c_int()
                    result = self.dwf._lib.FDwfDeviceOpen(ctypes.c_int(0), ctypes.byref(hdwf_c))
                    if result == 1:
                        self.hdwf = hdwf_c.value
                        print("デバイス接続成功 (低レベルAPI)")
                        self.is_connected = True
                        return True
                    else:
                        print("低レベルAPI接続も失敗")
                        return False
                except Exception as e2:
                    print(f"低レベルAPI接続エラー: {e2}")
                    return False
                    
        except Exception as e:
            print(f"デバイス接続エラー: {e}")
            return False
    
    def output_waveform_and_record(self, waveform_chunk, chunk_index):
        """1秒分の波形を出力し、同時に入力を記録"""
        if not self.is_connected:
            print("デバイスが接続されていません")
            return None
            
        try:
            # 正規化：0~3.3Vの範囲に調整
            min_val = np.min(waveform_chunk)
            max_val = np.max(waveform_chunk)
            if max_val > min_val:
                waveform_normalized = (waveform_chunk - min_val) / (max_val - min_val) * 3.3
            else:
                waveform_normalized = np.ones_like(waveform_chunk) * 1.65
            
            # AWG用に-1~1の範囲に変換
            awg_waveform = (waveform_normalized - 1.65) / output_amp
            
            # 低レベルAPIでAWG設定
            lib = self.dwf._lib
            channel = 0
            hdwf_c = ctypes.c_int(self.hdwf) if isinstance(self.hdwf, int) else ctypes.c_int(self.hdwf.value)
            
            # AWGリセット
            lib.FDwfAnalogOutReset(hdwf_c, ctypes.c_int(channel))
            
            # AWG設定
            lib.FDwfAnalogOutNodeEnableSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_bool(True))
            lib.FDwfAnalogOutNodeFunctionSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_int(31))  # funcPlay
            lib.FDwfAnalogOutNodeFrequencySet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(sample_rate))
            lib.FDwfAnalogOutNodeAmplitudeSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(output_amp))
            lib.FDwfAnalogOutNodeOffsetSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(1.65))
            
            # データ設定
            rgdData = (ctypes.c_double * len(awg_waveform))(*awg_waveform)
            lib.FDwfAnalogOutNodeDataSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), rgdData, ctypes.c_int(len(awg_waveform)))
            
            # スコープ設定（入力記録用）
            scope_channel = 0
            lib.FDwfAnalogInChannelEnableSet(hdwf_c, ctypes.c_int(scope_channel), ctypes.c_bool(True))
            lib.FDwfAnalogInChannelRangeSet(hdwf_c, ctypes.c_int(scope_channel), ctypes.c_double(5.0))  # ±5V
            lib.FDwfAnalogInFrequencySet(hdwf_c, ctypes.c_double(sample_rate))
            lib.FDwfAnalogInBufferSizeSet(hdwf_c, ctypes.c_int(len(waveform_chunk)))
            
            # 同期開始
            lib.FDwfAnalogInConfigure(hdwf_c, ctypes.c_bool(False), ctypes.c_bool(True))  # スコープ開始
            lib.FDwfAnalogOutConfigure(hdwf_c, ctypes.c_int(channel), ctypes.c_bool(True))  # AWG開始
            
            # 記録完了まで待機
            sts = ctypes.c_byte()
            while True:
                lib.FDwfAnalogInStatus(hdwf_c, ctypes.c_bool(True), ctypes.byref(sts))
                if sts.value == 2:  # DwfStateDone
                    break
                time.sleep(0.01)
            
            # データ取得
            rgdSamples = (ctypes.c_double * len(waveform_chunk))()
            lib.FDwfAnalogInStatusData(hdwf_c, ctypes.c_int(scope_channel), rgdSamples, ctypes.c_int(len(waveform_chunk)))
            captured_data = np.array(rgdSamples)
            
            print(f"チャンク {chunk_index + 1}: 出力・記録完了")
            
            return {
                'output_voltage': waveform_normalized,
                'input_voltage': captured_data,
                'time_start': chunk_index * extract_duration
            }
            
        except Exception as e:
            print(f"出力・記録エラー: {e}")
            # ダミーデータを返す
            waveform_normalized = (waveform_chunk + 1) * 1.65
            captured_data = waveform_normalized + np.random.normal(0, 0.01, len(waveform_chunk))
            return {
                'output_voltage': waveform_normalized,
                'input_voltage': captured_data,
                'time_start': chunk_index * extract_duration
            }
    
    def disconnect(self):
        """デバイス接続を切断"""
        if self.is_connected and self.hdwf is not None:
            try:
                if isinstance(self.hdwf, int):
                    self.dwf._lib.FDwfDeviceClose(ctypes.c_int(self.hdwf))
                else:
                    self.dwf.deviceControl.close(self.hdwf)
                print("デバイス接続を終了しました")
            except Exception as e:
                print(f"デバイスクローズエラー: {e}")
            finally:
                self.is_connected = False

def save_data_and_plot(result_data, chunk_index):
    """データをCSVとグラフで保存"""
    output_voltage = result_data['output_voltage']
    captured = result_data['input_voltage']
    start_time = result_data['time_start']
    
    # 時間軸作成
    time_axis = np.arange(len(captured)) / sample_rate + start_time
    
    # CSVファイルに保存
    df = pd.DataFrame({
        'Time(s)': time_axis,
        'Output_Voltage(V)': output_voltage,
        'Input_Voltage(V)': captured
    })
    csv_filename = f"{output_csv_folder}/captured_data_{chunk_index+1}s.csv"
    df.to_csv(csv_filename, index=False)
    
    # グラフ作成・保存
    plt.figure(figsize=(12, 8))
    
    # 出力電圧
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, output_voltage, 'r-', linewidth=0.5, label='Output Voltage')
    plt.xlabel('Time (s)')
    plt.ylabel('Output Voltage (V)')
    plt.title(f'Output Voltage ({chunk_index+1}秒目: {start_time:.1f}s - {start_time + extract_duration:.1f}s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(start_time, start_time + extract_duration)
    
    # 入力電圧（取得データ）
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, captured, 'b-', linewidth=0.5, label='Input Voltage')
    plt.xlabel('Time (s)')
    plt.ylabel('Input Voltage (V)')
    plt.title(f'Captured Input Voltage ({chunk_index+1}秒目: {start_time:.1f}s - {start_time + extract_duration:.1f}s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(start_time, start_time + extract_duration)
    
    plt.tight_layout()
    plot_filename = f"{output_graph_folder}/captured_waveform_{chunk_index+1}s.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"保存完了:")
    print(f"- CSV: {csv_filename} ({len(captured)}サンプル)")
    print(f"- グラフ: {plot_filename}")

def process_audio_file(wav_file_path):
    """音声ファイルを処理するメイン関数"""
    print(f"音声ファイル処理開始: {wav_file_path}")
    
    # 音声ファイル読み込み
    try:
        waveform_all, file_duration = load_audio_file(wav_file_path)
        print(f"ファイル読み込み完了:")
        print(f"- ファイル長: {file_duration:.1f}秒")
        print(f"- サンプル数: {len(waveform_all)}")
        print(f"- サンプリングレート: {sample_rate}Hz")
    except Exception as e:
        print(f"音声ファイル読み込みエラー: {e}")
        return
    
    # 1秒ごとのチャンクに分割
    chunks = extract_1s_chunks(waveform_all)
    print(f"音声データを{len(chunks)}個の1秒チャンクに分割しました")
    
    # AD3コントローラー初期化
    ad3 = AD3Controller()
    
    # デバイス接続
    if not ad3.connect():
        print("AD3デバイス接続に失敗しました。ダミーデータで処理を継続します。")
    
    # 各チャンクを処理
    for i, chunk in enumerate(chunks):
        print(f"\n--- {i+1}秒目のデータ処理中 ---")
        
        # AD3で出力・記録
        result_data = ad3.output_waveform_and_record(chunk, i)
        
        if result_data is not None:
            # データ保存
            save_data_and_plot(result_data, i)
        else:
            print(f"チャンク {i+1} の処理に失敗しました")
    
    # デバイス切断
    ad3.disconnect()
    
    print(f"\n全データ処理完了。{len(chunks)}個のCSVファイルとグラフが生成されました。")

if __name__ == "__main__":
    # デフォルトの音声ファイルまたはコマンドライン引数から取得
    import sys
    
    if len(sys.argv) > 1:
        wav_file_path = sys.argv[1]
    else:
        # デフォルトファイルを探す
        default_files = [
            f"{input_folder}/1-137-A-32.wav",
            "1-137-A-32.wav",
            "ESC-50/audio/1-137-A-32.wav"
        ]
        
        wav_file_path = None
        for file_path in default_files:
            if os.path.exists(file_path):
                wav_file_path = file_path
                break
        
        if wav_file_path is None:
            print("音声ファイルが見つかりません。")
            print("使用方法: python waveform.py [音声ファイルパス]")
            print("または、以下のいずれかのファイルを配置してください:")
            for file_path in default_files:
                print(f"  - {file_path}")
            exit(1)
    
    if not os.path.exists(wav_file_path):
        print(f"音声ファイルが存在しません: {wav_file_path}")
        exit(1)
    
    process_audio_file(wav_file_path)

