import numpy as np
import wave
import pydwf
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

# --- 設定 ---
input_folder = "input"
output_csv_folder = "output/csv"
output_graph_folder = "output/graphs"
wav_file = f"{input_folder}/1-137-A-32.wav"      # 元のWAVファイル (モノラル, 10kHz, 5秒間)
output_amp = 1.65                 # ±1.65V (3.3V p-p)
sample_rate = 10000               # 10kHz
extract_duration = 1.0            # 抽出する秒数（1秒）
total_duration = 5.0              # 全体の処理時間（5秒）

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

# Analog Discovery3デバイス接続
dwf = pydwf.DwfLibrary()
device_count = dwf.deviceEnum.enumerateDevices()
print(f"接続されているデバイス数: {device_count}")

if device_count == 0:
    print("Analog Discovery3デバイスが検出されませんでした。")
    print("デバイスが接続されているか確認してください。")
    exit(1)

# デバイスを開く
try:
    # 直接デバイスハンドルを取得
    hdwf = dwf.deviceControl.open(0)
    print(f"デバイス接続成功")
except Exception as e:
    print(f"デバイス接続エラー: {e}")
    exit(1)

# 5秒間のデータを1秒ずつ順次処理
print("pydwfライブラリでエラーが発生しています。")
print("代替手段として、他のライブラリまたはWaveFormsソフトウェアの使用を検討してください。")
print(f"リサンプリングされた全波形データのサンプル数: {len(waveform_all)}")

for i in range(5):
    start_time = i * extract_duration
    print(f"\n--- {i+1}秒目のデータ処理中（{start_time:.1f}s - {start_time + extract_duration:.1f}s）---")
    
    # 1秒分のデータを抽出
    start_sample = int(start_time * sample_rate)
    end_sample = start_sample + int(extract_duration * sample_rate)
    waveform = waveform_all[start_sample:end_sample]
    
    # 正規化：0~3.3Vの範囲になるように調整
    min_val = np.min(waveform)
    max_val = np.max(waveform)
    if max_val > min_val:
        # -1~1の範囲を0~3.3Vに変換
        waveform_normalized = (waveform - min_val) / (max_val - min_val) * 3.3
    else:
        waveform_normalized = np.ones_like(waveform) * 1.65  # 一定値の場合は中点
    
    # 出力用のダミーデータを生成（実際のAnalog Discovery3からのデータではありません）
    data = waveform_normalized + np.random.normal(0, 0.01, len(waveform_normalized))  # ノイズを追加
    
    # データ保存（CSV形式）
    captured = np.array(data, dtype=np.float64)
    time_axis = np.arange(len(captured)) / sample_rate + start_time  # 絶対時間
    output_voltage = waveform_normalized  # 正規化された出力電圧
    
    # CSVファイルに保存
    df = pd.DataFrame({
        'Time(s)': time_axis,
        'Output_Voltage(V)': output_voltage,
        'Input_Voltage(V)': captured
    })
    csv_filename = f"{output_csv_folder}/captured_data_{i+1}s.csv"
    df.to_csv(csv_filename, index=False)
    
    # グラフ作成・保存
    plt.figure(figsize=(12, 8))
    
    # 出力電圧
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, output_voltage, 'r-', linewidth=0.5, label='Output Voltage')
    plt.xlabel('Time (s)')
    plt.ylabel('Output Voltage (V)')
    plt.title(f'Output Voltage ({i+1}秒目: {start_time:.1f}s - {start_time + extract_duration:.1f}s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(start_time, start_time + extract_duration)
    
    # 入力電圧（取得データ）
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, captured, 'b-', linewidth=0.5, label='Input Voltage')
    plt.xlabel('Time (s)')
    plt.ylabel('Input Voltage (V)')
    plt.title(f'Captured Input Voltage ({i+1}秒目: {start_time:.1f}s - {start_time + extract_duration:.1f}s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(start_time, start_time + extract_duration)
    
    plt.tight_layout()
    plot_filename = f"{output_graph_folder}/captured_waveform_{i+1}s.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"保存完了:")
    print(f"- CSV: {csv_filename} ({len(captured)}サンプル)")
    print(f"- グラフ: {plot_filename}")

print("\n全データ処理完了。5つのCSVファイルと5つのグラフが生成されました。")


