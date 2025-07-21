"""
データ保存・可視化モジュール
pydwfに依存しないデータ処理機能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from config import (
    OUTPUT_CSV_FOLDER, OUTPUT_GRAPH_FOLDER, SAMPLE_RATE, EXTRACT_DURATION,
    GRAPH_DPI, GRAPH_FIGSIZE, GRAPH_LINEWIDTH, GRAPH_ALPHA
)


class DataHandler:
    """データ保存・可視化クラス"""
    
    def __init__(self):
        self.csv_folder = OUTPUT_CSV_FOLDER
        self.graph_folder = OUTPUT_GRAPH_FOLDER
        self.sample_rate = SAMPLE_RATE
        self.extract_duration = EXTRACT_DURATION
        
        # 出力フォルダを作成
        self._ensure_output_directories()
    
    def _ensure_output_directories(self):
        """出力ディレクトリが存在しない場合は作成"""
        os.makedirs(self.csv_folder, exist_ok=True)
        os.makedirs(self.graph_folder, exist_ok=True)
    
    def get_audio_filename_base(self, audio_file_path):
        """音声ファイルパスからベースファイル名を取得（拡張子なし）"""
        return Path(audio_file_path).stem
    
    def save_continuous_data_with_label(self, result_data, labeled_filename_base):
        """
        ラベル付きファイル名で連続データを保存

        Args:
            result_data (dict): 連続データ
            labeled_filename_base (str): ラベル付きファイル名のベース

        Returns:
            tuple: (CSVファイルパス, グラフファイルパス)
        """
        # unpack
        output_voltage = result_data['output_voltage']
        input_voltage  = result_data['input_voltage']
        time_start     = result_data['time_start']
        total_duration = result_data['total_duration']
        num_chunks     = result_data['num_chunks']

        # CSVファイル保存
        # 時間軸作成（フラット配列化含む）
        if isinstance(output_voltage, list):
            output_voltage = np.concatenate(output_voltage)
            input_voltage  = np.concatenate(input_voltage)
        time_axis_csv = np.arange(len(input_voltage)) / self.sample_rate

        df = pd.DataFrame({
            'Time(s)':             time_axis_csv,
            'Output_Voltage(V)':   output_voltage,
            'Input_Voltage(V)':    input_voltage
        })
        csv_filename = os.path.join(self.csv_folder, f"{labeled_filename_base}.csv")
        df.to_csv(csv_filename, index=False)

        # グラフ作成
        # ダウンサンプリング（最大10000点）
        max_plot_points = 10000
        step = max(1, len(output_voltage) // max_plot_points)
        time_axis = np.linspace(time_start,
                                time_start + total_duration,
                                len(output_voltage))[::step]
        output_voltage = output_voltage[::step]
        input_voltage  = input_voltage[::step]

        plt.figure(figsize=(15, 10))

        # 出力電圧のプロット
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, output_voltage,
                linewidth=GRAPH_LINEWIDTH, label='Output Voltage')
        for i in range(1, num_chunks):
            plt.axvline(x=i * EXTRACT_DURATION,
                        color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Output Voltage (V)')
        plt.title(f'Continuous Output Voltage - {labeled_filename_base} '
                f'({total_duration:.1f}s, {num_chunks} chunks)')
        plt.grid(True, alpha=GRAPH_ALPHA)
        plt.legend()
        plt.xlim(0, total_duration)
        plt.ylim(0, 4)

        # 入力電圧のプロット
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, input_voltage,
                linewidth=GRAPH_LINEWIDTH, label='Input Voltage')
        for i in range(1, num_chunks):
            plt.axvline(x=i * EXTRACT_DURATION,
                        color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Input Voltage (V)')
        plt.title(f'Continuous Captured Input Voltage - {labeled_filename_base} '
                f'({total_duration:.1f}s, {num_chunks} chunks)')
        plt.grid(True, alpha=GRAPH_ALPHA)
        plt.legend()
        plt.xlim(0, total_duration)
        plt.ylim(0, 4)

        plt.tight_layout()

        # グラフファイルに保存
        plot_filename = os.path.join(self.graph_folder, f"{labeled_filename_base}.png")
        plt.savefig(plot_filename, dpi=GRAPH_DPI)
        plt.close()

        print(f"ラベル付き連続データ保存完了:")
        print(f"- CSV: {csv_filename}")
        print(f"- グラフ: {plot_filename}")
        print(f"- 総時間: {total_duration:.1f}秒")
        print(f"- チャンク数: {num_chunks}")
        print(f"- サンプル数: {len(input_voltage)}")

        return csv_filename, plot_filename
    
    def save_first_second_highres(self, result_data, labeled_filename_base):
        """
        最初の1秒を高解像度でCSVとグラフ保存
        
        Args:
            result_data (dict): 連続データ
            labeled_filename_base (str): ラベル付きファイル名のベース
            
        Returns:
            tuple: (CSVファイルパス, グラフファイルパス) or (None, None) if no data
        """
        output_voltage = result_data['output_voltage']
        input_voltage  = result_data['input_voltage']
        
        # リストの場合は配列に変換
        if isinstance(output_voltage, list):
            output_voltage = np.concatenate(output_voltage)
            input_voltage  = np.concatenate(input_voltage)
        
        # 1秒分のサンプル数を計算
        samples_per_second = int(self.sample_rate * EXTRACT_DURATION)
        
        if len(output_voltage) < samples_per_second:
            print("警告: データが1秒未満のため、高解像度保存をスキップします")
            return None, None
        
        # 最初の1秒のデータを抽出
        first_sec_output = output_voltage[:samples_per_second]
        first_sec_input = input_voltage[:samples_per_second]
        
        # 時間軸作成
        time_axis = np.arange(len(first_sec_output)) / self.sample_rate
        
        # CSV保存
        df = pd.DataFrame({
            'Time(s)': time_axis,
            'Output_Voltage(V)': first_sec_output,
            'Input_Voltage(V)': first_sec_input
        })
        csv_filename = os.path.join(self.csv_folder, f"{labeled_filename_base}_first1s_highres.csv")
        df.to_csv(csv_filename, index=False)
        
        # 高解像度グラフ作成
        plt.figure(figsize=(16, 12))  # より大きなサイズ
        
        # 出力電圧のプロット（高解像度）
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, first_sec_output, 'r-', linewidth=0.8, label='Output Voltage')
        plt.xlabel('Time (s)')
        plt.ylabel('Output Voltage (V)')
        plt.title(f'High-Res Output Voltage (First 1s) - {labeled_filename_base}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, 1.0)
        plt.ylim(0, 3.5)
        
        # より細かい時間軸のティック
        plt.xticks(np.arange(0, 1.1, 0.1))
        
        # 入力電圧のプロット（高解像度）
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, first_sec_input, 'b-', linewidth=0.8, label='Input Voltage')
        plt.xlabel('Time (s)')
        plt.ylabel('Input Voltage (V)')
        plt.title(f'High-Res Captured Input Voltage (First 1s) - {labeled_filename_base}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, 1.0)
        plt.ylim(0, 3.5)
        
        # より細かい時間軸のティック
        plt.xticks(np.arange(0, 1.1, 0.1))
        
        plt.tight_layout()
        
        # 高解像度で保存
        plot_filename = os.path.join(self.graph_folder, f"{labeled_filename_base}_first1s_highres.png")
        plt.savefig(plot_filename, dpi=600, bbox_inches='tight')  # 高DPI
        plt.close()
        
        print(f"最初の1秒高解像度保存完了:")
        print(f"- CSV: {csv_filename}")
        print(f"- グラフ: {plot_filename}")
        print(f"- サンプル数: {len(first_sec_output)}")
        
        return csv_filename, plot_filename
