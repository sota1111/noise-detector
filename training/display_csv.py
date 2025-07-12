import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# カテゴリ名の定義（classification.pyと同じ）
category_names = [
    'dog', 'cat', 'rain', 'sea_waves', 'crackling_fire', 'crickets', 
    'wind', 'water_drops', 'thunderstorm', 'crying_baby', 'sneezing', 
    'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 
    'brushing_teeth', 'snoring', 'door_wood_knock', 'mouse_click', 
    'keyboard_typing', 'can_opening', 'washing_machine', 'vacuum_cleaner', 
    'toilet_flush', 'clock_alarm', 'clock_tick', 'glass_breaking', 
    'door_wood_creaks', 'siren', 'car_horn', 'engine', 'train', 'silent'
]

num_classes = 34

def get_random_csv_files(folder_path, num_files):
    """
    指定されたフォルダ内からランダムにCSVファイルを選択する
    """
    if not os.path.exists(folder_path):
        print(f"警告: フォルダ {folder_path} が存在しません")
        return []
    
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if len(all_files) == 0:
        print(f"警告: フォルダ {folder_path} にCSVファイルが見つかりません")
        return []
    
    return random.sample(all_files, min(num_files, len(all_files)))

def plot_csv_files_grid(folder_path, title, rows, cols, normalize=True):
    """
    フォルダ内のCSVファイルをランダムに選択し、指定されたグリッドサイズで折れ線グラフを描画する
    
    Args:
        folder_path (str): CSVファイルが格納されているフォルダパス
        title (str): グラフのタイトル
        rows (int): 行数
        cols (int): 列数
        normalize (bool): データを-1~1に正規化するかどうか
    """
    num_files = rows * cols
    files = get_random_csv_files(folder_path, num_files)
    
    if len(files) == 0:
        return
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # axesが1次元の場合は2次元に変換
    if rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for i, ax in enumerate(axes.ravel()):
        if i < len(files):
            file = files[i]
            file_path = os.path.join(folder_path, file)
            try:
                data = pd.read_csv(file_path, header=None).iloc[0].values  # 1行目のデータを取得
                
                if normalize:
                    # 0-4095の値を-1~1に正規化（classification.pyと同じ処理）
                    data = (data - 2047.5) / 2047.5
                
                # データをプロット
                ax.plot(data, linewidth=0.8)
                ax.set_title(f"{file}\n(samples: {len(data)})", fontsize=8)
                ax.set_xlabel('Sample Index', fontsize=8)
                ax.set_ylabel('Amplitude', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=6)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error reading\n{file}", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Error: {file}", fontsize=8)
        else:
            # 空のサブプロットを非表示にする
            ax.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def plot_all_categories_overview(rows=6, cols=6, samples_per_category=1):
    """
    全34カテゴリの概要を一度に表示する
    
    Args:
        rows (int): 行数
        cols (int): 列数  
        samples_per_category (int): カテゴリごとのサンプル数
    """
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle('ESC-50 Audio Dataset Overview (34 Categories + Silent)', fontsize=16)
    
    for i, ax in enumerate(axes.ravel()):
        if i < num_classes:
            folder_path = f"dataset/label{i}"
            category_name = category_names[i]
            
            if os.path.exists(folder_path):
                files = get_random_csv_files(folder_path, samples_per_category)
                if files:
                    file_path = os.path.join(folder_path, files[0])
                    try:
                        data = pd.read_csv(file_path, header=None).iloc[0].values
                        # 正規化
                        data = (data - 2047.5) / 2047.5
                        
                        ax.plot(data, linewidth=0.6)
                        ax.set_title(f"Label {i}: {category_name}", fontsize=8)
                        ax.set_ylabel('Amplitude', fontsize=6)
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(labelsize=5)
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error\n{category_name}", ha='center', va='center', transform=ax.transAxes, fontsize=6)
                else:
                    ax.text(0.5, 0.5, f"No data\n{category_name}", ha='center', va='center', transform=ax.transAxes, fontsize=6)
            else:
                ax.text(0.5, 0.5, f"Missing\n{category_name}", ha='center', va='center', transform=ax.transAxes, fontsize=6)
        else:
            ax.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def plot_specific_categories(category_indices, rows=2, cols=2):
    """
    特定のカテゴリを詳細に表示する
    
    Args:
        category_indices (list): 表示したいカテゴリのインデックスリスト
        rows (int): 行数
        cols (int): 列数
    """
    for category_idx in category_indices:
        if 0 <= category_idx < num_classes:
            folder_path = f"dataset/label{category_idx}"
            category_name = category_names[category_idx]
            title = f"Label {category_idx}: {category_name.upper()} - Detailed View"
            
            print(f"表示中: {title}")
            plot_csv_files_grid(folder_path, title, rows, cols)
        else:
            print(f"警告: カテゴリインデックス {category_idx} は無効です（0-{num_classes-1}の範囲で指定してください）")

def display_dataset_statistics():
    """
    データセットの統計情報を表示する
    """
    print("=== ESC-50 Dataset Statistics ===")
    total_files = 0
    
    for i in range(num_classes):
        folder_path = f"dataset/label{i}"
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            file_count = len(files)
            total_files += file_count
            print(f"Label {i:2d} ({category_names[i]:20s}): {file_count:3d} files")
        else:
            print(f"Label {i:2d} ({category_names[i]:20s}): フォルダが存在しません")
    
    print(f"\n総ファイル数: {total_files}")
    print(f"カテゴリ数: {num_classes}")

def interactive_category_viewer():
    """
    インタラクティブなカテゴリビューア
    """
    print("\n=== インタラクティブ カテゴリビューア ===")
    print("使用可能なカテゴリ:")
    for i, name in enumerate(category_names):
        print(f"  {i:2d}: {name}")
    
    while True:
        try:
            user_input = input("\n表示したいカテゴリ番号を入力してください（複数の場合はカンマ区切り、'q'で終了、'all'で全体概要）: ")
            
            if user_input.lower() == 'q':
                break
            elif user_input.lower() == 'all':
                plot_all_categories_overview()
            else:
                category_indices = [int(x.strip()) for x in user_input.split(',')]
                plot_specific_categories(category_indices)
                
        except ValueError:
            print("無効な入力です。数字をカンマ区切りで入力してください。")
        except KeyboardInterrupt:
            print("\n終了します。")
            break

if __name__ == "__main__":
    # データセット統計を表示
    display_dataset_statistics()
    
    # 全カテゴリの概要を表示
    print("\n全カテゴリの概要を表示中...")
    plot_all_categories_overview()
    
    # いくつかの興味深いカテゴリの詳細を表示
    print("\n特定カテゴリの詳細表示...")
    interesting_categories = [0, 1, 2]  # dog, cat, rain
    plot_specific_categories(interesting_categories, rows=2, cols=2)
    
    # インタラクティブビューアを開始
    interactive_category_viewer()