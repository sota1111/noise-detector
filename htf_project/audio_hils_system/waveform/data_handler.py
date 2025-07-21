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
    
    def save_csv(self, result_data, chunk_index, filename_prefix="captured_data"):
        """
        データをCSVファイルに保存
        
        Args:
            result_data (dict): 保存するデータ（output_voltage, input_voltage, time_start）
            chunk_index (int): チャンクのインデックス
            filename_prefix (str): ファイル名のプレフィックス
            
        Returns:
            str: 保存されたCSVファイルのパス
        """
        output_voltage = result_data['output_voltage']
        input_voltage = result_data['input_voltage']
        start_time = result_data['time_start']
        
        # 時間軸作成
        time_axis = np.arange(len(input_voltage)) / self.sample_rate + start_time
        
        # DataFrameに変換
        df = pd.DataFrame({
            'Time(s)': time_axis,
            'Output_Voltage(V)': output_voltage,
            'Input_Voltage(V)': input_voltage
        })
        
        # CSVファイルに保存
        csv_filename = os.path.join(self.csv_folder, f"{filename_prefix}_{chunk_index+1}s.csv")
        df.to_csv(csv_filename, index=False)
        
        return csv_filename
    
    def save_graph(self, result_data, chunk_index, filename_prefix="captured_waveform"):
        """
        データをグラフとして保存
        
        Args:
            result_data (dict): 描画するデータ（output_voltage, input_voltage, time_start）
            chunk_index (int): チャンクのインデックス
            filename_prefix (str): ファイル名のプレフィックス
            
        Returns:
            str: 保存されたグラフファイルのパス
        """
        output_voltage = result_data['output_voltage']
        input_voltage = result_data['input_voltage']
        start_time = result_data['time_start']
        
        # 時間軸作成
        time_axis = np.arange(len(input_voltage)) / self.sample_rate + start_time
        
        # グラフ作成
        plt.figure(figsize=GRAPH_FIGSIZE)
        
        # 出力電圧のプロット
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, output_voltage, 'r-', linewidth=GRAPH_LINEWIDTH, label='Output Voltage')
        plt.xlabel('Time (s)')
        plt.ylabel('Output Voltage (V)')
        plt.title(f'Output Voltage (Chunk {chunk_index+1}: {start_time:.1f}s - {start_time + self.extract_duration:.1f}s)')
        plt.grid(True, alpha=GRAPH_ALPHA)
        plt.legend()
        plt.xlim(start_time, start_time + self.extract_duration)
        plt.ylim(0, 4)  # 縦軸を0-4Vに固定
        
        # 入力電圧のプロット
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, input_voltage, 'b-', linewidth=GRAPH_LINEWIDTH, label='Input Voltage')
        plt.xlabel('Time (s)')
        plt.ylabel('Input Voltage (V)')
        plt.title(f'Captured Input Voltage (Chunk {chunk_index+1}: {start_time:.1f}s - {start_time + self.extract_duration:.1f}s)')
        plt.grid(True, alpha=GRAPH_ALPHA)
        plt.legend()
        plt.xlim(start_time, start_time + self.extract_duration)
        plt.ylim(0, 4)  # 縦軸を0-4Vに固定
        
        plt.tight_layout()
        
        # グラフファイルに保存
        plot_filename = os.path.join(self.graph_folder, f"{filename_prefix}_{chunk_index+1}s.png")
        plt.savefig(plot_filename, dpi=GRAPH_DPI, bbox_inches='tight')
        plt.close()
        
        return plot_filename
    
    def save_data_and_plot(self, result_data, chunk_index):
        """
        データをCSVとグラフ両方で保存
        
        Args:
            result_data (dict): 保存するデータ
            chunk_index (int): チャンクのインデックス
            
        Returns:
            tuple: (CSVファイルパス, グラフファイルパス)
        """
        # CSV保存
        csv_path = self.save_csv(result_data, chunk_index)
        
        # グラフ保存
        plot_path = self.save_graph(result_data, chunk_index)
        
        # 保存完了メッセージ
        sample_count = len(result_data['input_voltage'])
        print(f"保存完了:")
        print(f"- CSV: {csv_path} ({sample_count}サンプル)")
        print(f"- グラフ: {plot_path}")
        
        return csv_path, plot_path
    
    def create_summary_csv(self, all_results, filename="summary_all_chunks.csv"):
        """
        全チャンクのデータを結合したサマリーCSVを作成
        
        Args:
            all_results (list): 全チャンクの結果データのリスト
            filename (str): 出力ファイル名
            
        Returns:
            str: 保存されたサマリーCSVファイルのパス
        """
        all_data = []
        
        for chunk_index, result_data in enumerate(all_results):
            if result_data is None:
                continue
                
            output_voltage = result_data['output_voltage']
            input_voltage = result_data['input_voltage']
            start_time = result_data['time_start']
            
            # 時間軸作成
            time_axis = np.arange(len(input_voltage)) / self.sample_rate + start_time
            
            # データフレーム作成
            chunk_df = pd.DataFrame({
                'Time(s)': time_axis,
                'Output_Voltage(V)': output_voltage,
                'Input_Voltage(V)': input_voltage,
                'Chunk': chunk_index + 1
            })
            
            all_data.append(chunk_df)
        
        if all_data:
            # 全データを結合
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # ファイル保存
            summary_path = os.path.join(self.csv_folder, filename)
            combined_df.to_csv(summary_path, index=False)
            
            print(f"サマリーCSV保存完了: {summary_path} ({len(combined_df)}サンプル)")
            return summary_path
        else:
            print("保存するデータがありません")
            return None
    
    def create_summary_graph(self, all_results, filename="summary_all_chunks.png"):
        """
        全チャンクのデータを重ねたサマリーグラフを作成
        
        Args:
            all_results (list): 全チャンクの結果データのリスト
            filename (str): 出力ファイル名
            
        Returns:
            str: 保存されたサマリーグラフファイルのパス
        """
        if not all_results or all(r is None for r in all_results):
            print("グラフ作成用のデータがありません")
            return None
        
        plt.figure(figsize=(15, 10))
        
        # 出力電圧の全体プロット
        plt.subplot(2, 1, 1)
        for chunk_index, result_data in enumerate(all_results):
            if result_data is None:
                continue
            
            output_voltage = result_data['output_voltage']
            start_time = result_data['time_start']
            time_axis = np.arange(len(output_voltage)) / self.sample_rate + start_time
            
            plt.plot(time_axis, output_voltage, linewidth=GRAPH_LINEWIDTH, 
                    label=f'Chunk {chunk_index+1}', alpha=0.8)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Output Voltage (V)')
        plt.title('All Output Voltages')
        plt.grid(True, alpha=GRAPH_ALPHA)
        plt.legend()
        
        # 入力電圧の全体プロット
        plt.subplot(2, 1, 2)
        for chunk_index, result_data in enumerate(all_results):
            if result_data is None:
                continue
            
            input_voltage = result_data['input_voltage']
            start_time = result_data['time_start']
            time_axis = np.arange(len(input_voltage)) / self.sample_rate + start_time
            
            plt.plot(time_axis, input_voltage, linewidth=GRAPH_LINEWIDTH, 
                    label=f'Chunk {chunk_index+1}', alpha=0.8)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Input Voltage (V)')
        plt.title('All Captured Input Voltages')
        plt.grid(True, alpha=GRAPH_ALPHA)
        plt.legend()
        
        plt.tight_layout()
        
        # ファイル保存
        summary_plot_path = os.path.join(self.graph_folder, filename)
        plt.savefig(summary_plot_path, dpi=GRAPH_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"サマリーグラフ保存完了: {summary_plot_path}")
        return summary_plot_path
    
    def get_file_stats(self):
        """
        出力フォルダ内のファイル統計を取得
        
        Returns:
            dict: ファイル統計情報
        """
        stats = {
            'csv_files': [],
            'graph_files': [],
            'csv_count': 0,
            'graph_count': 0
        }
        
        # CSVファイル
        if os.path.exists(self.csv_folder):
            csv_files = [f for f in os.listdir(self.csv_folder) if f.endswith('.csv')]
            stats['csv_files'] = csv_files
            stats['csv_count'] = len(csv_files)
        
        # グラフファイル
        if os.path.exists(self.graph_folder):
            graph_files = [f for f in os.listdir(self.graph_folder) if f.endswith('.png')]
            stats['graph_files'] = graph_files
            stats['graph_count'] = len(graph_files)
        
        return stats
    
    def get_audio_filename_base(self, audio_file_path):
        """音声ファイルパスからベースファイル名を取得（拡張子なし）"""
        return Path(audio_file_path).stem
    
    def save_continuous_csv(self, result_data, audio_filename_base):
        """
        連続データをCSVファイルに保存
        
        Args:
            result_data (dict): 連続データ（output_voltage, input_voltage, total_duration, num_chunks）
            audio_filename_base (str): 音声ファイルのベース名
            
        Returns:
            str: 保存されたCSVファイルのパス
        """
        output_voltage = result_data['output_voltage']
        input_voltage = result_data['input_voltage']
        total_duration = result_data['total_duration']
        num_chunks = result_data['num_chunks']
        
        # 時間軸作成
        time_axis = np.arange(len(input_voltage)) / self.sample_rate
        
        # DataFrameに変換
        df = pd.DataFrame({
            'Time(s)': time_axis,
            'Output_Voltage(V)': output_voltage,
            'Input_Voltage(V)': input_voltage
        })
        
        # CSVファイルに保存
        csv_filename = os.path.join(self.csv_folder, f"{audio_filename_base}_continuous.csv")
        df.to_csv(csv_filename, index=False)
        
        print(f"連続データCSV保存完了: {csv_filename}")
        print(f"- 総時間: {total_duration:.1f}秒")
        print(f"- チャンク数: {num_chunks}")
        print(f"- サンプル数: {len(input_voltage)}")
        
        return csv_filename
    
    def save_continuous_graph(self, result_data, audio_filename_base):
        """
        連続データをグラフとして保存
        
        Args:
            result_data (dict): 連続データ
            audio_filename_base (str): 音声ファイルのベース名
            
        Returns:
            str: 保存されたグラフファイルのパス
        """
        output_voltage = result_data['output_voltage']
        input_voltage = result_data['input_voltage']
        total_duration = result_data['total_duration']
        num_chunks = result_data['num_chunks']
        
        # 時間軸作成
        time_axis = np.arange(len(input_voltage)) / self.sample_rate
        
        # グラフ作成
        plt.figure(figsize=(15, 10))
        
        # 出力電圧のプロット
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, output_voltage, 'r-', linewidth=GRAPH_LINEWIDTH, label='Output Voltage')
        
        # チャンク境界を表示
        for i in range(1, num_chunks):
            chunk_time = i * EXTRACT_DURATION
            plt.axvline(x=chunk_time, color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Output Voltage (V)')
        plt.title(f'Continuous Output Voltage - {audio_filename_base} ({total_duration:.1f}s, {num_chunks} chunks)')
        plt.grid(True, alpha=GRAPH_ALPHA)
        plt.legend()
        plt.xlim(0, total_duration)
        plt.ylim(0, 4)  # 縦軸を0-4Vに固定
        
        # 入力電圧のプロット
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, input_voltage, 'b-', linewidth=GRAPH_LINEWIDTH, label='Input Voltage')
        
        # チャンク境界を表示
        for i in range(1, num_chunks):
            chunk_time = i * EXTRACT_DURATION
            plt.axvline(x=chunk_time, color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Input Voltage (V)')
        plt.title(f'Continuous Captured Input Voltage - {audio_filename_base} ({total_duration:.1f}s, {num_chunks} chunks)')
        plt.grid(True, alpha=GRAPH_ALPHA)
        plt.legend()
        plt.xlim(0, total_duration)
        plt.ylim(0, 4)  # 縦軸を0-4Vに固定
        
        plt.tight_layout()
        
        # グラフファイルに保存
        plot_filename = os.path.join(self.graph_folder, f"{audio_filename_base}_continuous.png")
        plt.savefig(plot_filename, dpi=GRAPH_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"連続データグラフ保存完了: {plot_filename}")
        
        return plot_filename
    
    def save_continuous_data(self, result_data, audio_filename_base):
        """
        連続データをCSVとグラフ両方で保存
        
        Args:
            result_data (dict): 連続データ
            audio_filename_base (str): 音声ファイルのベース名
            
        Returns:
            tuple: (CSVファイルパス, グラフファイルパス)
        """
        # CSV保存
        csv_path = self.save_continuous_csv(result_data, audio_filename_base)
        
        # グラフ保存
        plot_path = self.save_continuous_graph(result_data, audio_filename_base)
        
        return csv_path, plot_path
    
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
