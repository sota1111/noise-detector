import os
import pandas as pd
import numpy as np
import librosa
import random
from tqdm import tqdm

def find_loudest_segment(audio_file_path, target_samples=2000, sample_interval_us=100, pre_offset=200):
    """
    ESC-50音声ファイルから最も音が大きい部分を見つけて、
    その200サンプル手前から100μs間隔で2000データポイントを抽出し、
    0-4095の整数値に正規化する
    """
    try:
        # 音声ファイルを読み込み
        y, sr = librosa.load(audio_file_path, sr=None)
        
        # 100μsの間隔をサンプル数に変換
        interval_samples = int(sr * 100e-6)  # 100μs = 100 * 10^-6 秒
        
        # 2000データポイントに必要な総サンプル数
        total_samples_needed = target_samples * interval_samples
        
        if len(y) < total_samples_needed + pre_offset:
            print(f"警告: {audio_file_path} は必要なサンプル数が不足しています")
            return None
        
        # 音声の振幅の絶対値を計算
        abs_y = np.abs(y)
        
        # スライディングウィンドウで最も音が大きい部分を見つける
        window_size = total_samples_needed
        max_energy = 0
        best_start = 0
        
        for start in range(pre_offset, len(y) - window_size + 1):
            window = abs_y[start:start + window_size]
            energy = np.sum(window ** 2)  # RMSエネルギー
            
            if energy > max_energy:
                max_energy = energy
                best_start = start
        
        # 最も音が大きい部分の200サンプル手前から100μs間隔でデータを抽出
        actual_start = best_start - pre_offset
        extracted_data = []
        for i in range(target_samples):
            sample_index = actual_start + i * interval_samples
            if sample_index < len(y):
                extracted_data.append(y[sample_index])
            else:
                break
        
        # 0-4095の整数値に正規化（全データの最大値を使用）
        extracted_data = np.array(extracted_data)
        # 全音声データの最大絶対値を取得
        global_max = np.max(np.abs(y))
        # まず-1～1の範囲に正規化（全データの最大値で）
        if global_max > 0:
            extracted_data = extracted_data / global_max
        # 0～1の範囲に変換
        extracted_data = (extracted_data + 1) / 2
        # 0～4095の整数値に変換
        extracted_data = (extracted_data * 4095).astype(int)
        
        return extracted_data.tolist()
        
    except Exception as e:
        print(f"エラー: {audio_file_path} の処理中にエラーが発生しました: {e}")
        return None

def find_quietest_segment(audio_file_path, target_samples=2000, sample_interval_us=100, pre_offset=200):
    """
    ESC-50音声ファイルから最も音が小さい部分を見つけて、
    その200サンプル手前から100μs間隔で2000データポイントを抽出し、
    0-4095の整数値に正規化する
    """
    try:
        # 音声ファイルを読み込み
        y, sr = librosa.load(audio_file_path, sr=None)
        
        # 100μsの間隔をサンプル数に変換
        interval_samples = int(sr * 100e-6)  # 100μs = 100 * 10^-6 秒
        
        # 2000データポイントに必要な総サンプル数
        total_samples_needed = target_samples * interval_samples
        
        if len(y) < total_samples_needed + pre_offset:
            print(f"警告: {audio_file_path} は必要なサンプル数が不足しています")
            return None
        
        # 音声の振幅の絶対値を計算
        abs_y = np.abs(y)
        
        # スライディングウィンドウで最も音が小さい部分を見つける
        window_size = total_samples_needed
        min_energy = float('inf')
        best_start = 0
        
        for start in range(pre_offset, len(y) - window_size + 1):
            window = abs_y[start:start + window_size]
            energy = np.sum(window ** 2)  # RMSエネルギー
            
            if energy < min_energy:
                min_energy = energy
                best_start = start
        
        # 最も音が小さい部分の200サンプル手前から100μs間隔でデータを抽出
        actual_start = best_start - pre_offset
        extracted_data = []
        for i in range(target_samples):
            sample_index = actual_start + i * interval_samples
            if sample_index < len(y):
                extracted_data.append(y[sample_index])
            else:
                break
        
        # 0-4095の整数値に正規化（全データの最大値を使用）
        extracted_data = np.array(extracted_data)
        # 全音声データの最大絶対値を取得
        global_max = np.max(np.abs(y))
        # まず-1～1の範囲に正規化（全データの最大値で）
        if global_max > 0:
            extracted_data = extracted_data / global_max
        # 0～1の範囲に変換
        extracted_data = (extracted_data + 1) / 2
        # 0～4095の整数値に変換
        extracted_data = (extracted_data * 4095).astype(int)
        
        return extracted_data.tolist()
        
    except Exception as e:
        print(f"エラー: {audio_file_path} の処理中にエラーが発生しました: {e}")
        return None

def create_dataset():
    """
    ESC-50データセットから指定されたカテゴリの音声データをCSVに変換し、
    training/dataset下にlabel番号付きフォルダに格納する
    """
    
    # 対象カテゴリのマッピング（指定された順序でlabel0から番号付け）
    target_categories = [
        'dog', 'cat', 'rain', 'sea_waves', 'crackling_fire', 'crickets', 
        'wind', 'water_drops', 'thunderstorm', 'crying_baby', 'sneezing', 
        'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 
        'brushing_teeth', 'snoring', 'door_wood_knock', 'mouse_click', 
        'keyboard_typing', 'can_opening', 'washing_machine', 'vacuum_cleaner', 
        'toilet_flush', 'clock_alarm', 'clock_tick', 'glass_breaking', 
        'door_wood_creaks', 'siren', 'car_horn', 'engine', 'train'
    ]
    
    # カテゴリ名からlabel番号へのマッピング
    category_to_label = {cat: i for i, cat in enumerate(target_categories)}
    
    # メタデータを読み込み
    meta_df = pd.read_csv('ESC-50/meta/esc50.csv')
    
    # 対象カテゴリのファイルを抽出
    target_files = meta_df[meta_df['category'].isin(target_categories)]
    
    print(f"対象カテゴリ: {len(target_categories)}個")
    print(f"対象ファイル: {len(target_files)}個")
    
    # ディレクトリ構造を作成
    base_dir = "training/dataset"
    os.makedirs(base_dir, exist_ok=True)
    
    for i in range(len(target_categories)):
        label_dir = os.path.join(base_dir, f"label{i}")
        os.makedirs(label_dir, exist_ok=True)
    
    # silentフォルダ（既存カテゴリ数+1）も作成
    silent_label_num = len(target_categories)  # 33個のカテゴリがあるので、label33になる
    silent_dir = os.path.join(base_dir, f"label{silent_label_num}")
    os.makedirs(silent_dir, exist_ok=True)
    
    # 各カテゴリごとの処理状況を追跡
    category_counts = {cat: 0 for cat in target_categories}
    total_processed = 0
    
    print("\\n音声ファイルをCSVに変換中...")
    
    # 処理を高速化するため、librosaの設定を調整
    import warnings
    warnings.filterwarnings('ignore')
    
    for _, row in tqdm(target_files.iterrows(), total=len(target_files)):
        filename = row['filename']
        category = row['category']
        audio_path = os.path.join('ESC-50/audio', filename)
        
        # 音声データを処理
        audio_data = find_loudest_segment(audio_path)
        
        if audio_data is not None:
            # label番号を取得
            label_num = category_to_label[category]
            
            # CSV出力パス
            csv_filename = filename.replace('.wav', '.csv')
            csv_path = os.path.join(base_dir, f"label{label_num}", csv_filename)
            
            # CSVに保存（データのみ、ヘッダーなし）
            df = pd.DataFrame([audio_data])
            df.to_csv(csv_path, index=False, header=False)
            
            category_counts[category] += 1
            total_processed += 1
    
    # 結果を表示
    # 各ラベルからランダムに1つずつ選択してsilentデータを作成
    print("\\nsilentデータ作成中...")
    silent_count = 0
    
    # 各カテゴリの処理済みファイルを取得
    processed_files_by_category = {}
    for category in target_categories:
        label_num = category_to_label[category]
        label_files = []
        label_dir = os.path.join(base_dir, f"label{label_num}")
        if os.path.exists(label_dir):
            csv_files = [f for f in os.listdir(label_dir) if f.endswith('.csv')]
            # CSVファイル名から元のwavファイル名を復元
            for csv_file in csv_files:
                wav_filename = csv_file.replace('.csv', '.wav')
                label_files.append(wav_filename)
        processed_files_by_category[category] = label_files
    
    # 各カテゴリからランダムに1つずつ選択
    for category in target_categories:
        label_files = processed_files_by_category[category]
        if label_files:
            selected_file = random.choice(label_files)
            silent_audio_path = os.path.join('ESC-50/audio', selected_file)
            
            # 最も音が小さい部分からデータを抽出
            silent_data = find_quietest_segment(silent_audio_path)
            
            if silent_data is not None:
                # CSV出力パス
                csv_filename = selected_file.replace('.wav', f'_silent_{category}.csv')
                csv_path = os.path.join(silent_dir, csv_filename)
                
                # CSVに保存（データのみ、ヘッダーなし）
                df = pd.DataFrame([silent_data])
                df.to_csv(csv_path, index=False, header=False)
                
                silent_count += 1
                print(f"silentデータ作成: {selected_file} -> {csv_filename}")
    
    print(f"\\n=== 変換完了 ===")
    print(f"総処理ファイル数: {total_processed}")
    print("\\n各カテゴリの処理数:")
    for i, category in enumerate(target_categories):
        count = category_counts[category]
        print(f"label{i:2d} ({category:20s}): {count:3d}ファイル")
    
    print(f"label{silent_label_num} (silent        ): {silent_count:3d}ファイル")
    
    print(f"\\n出力先: {base_dir}/")
    print("各labelフォルダにCSVファイルが格納されました")

if __name__ == "__main__":
    create_dataset()