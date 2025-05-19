import os
import random
import pandas as pd
import matplotlib.pyplot as plt

def get_random_csv_files(folder_path, num_files):
    """
    指定されたフォルダ内からランダムにCSVファイルを選択する
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    return random.sample(all_files, min(num_files, len(all_files)))

def plot_csv_files_grid(folder_path, title, rows, cols):
    """
    フォルダ内のCSVファイルをランダムに選択し、指定されたグリッドサイズで折れ線グラフを描画する
    """
    num_files = rows * cols
    files = get_random_csv_files(folder_path, num_files)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
    fig.suptitle(title, fontsize=16)
    
    for ax, file in zip(axes.ravel(), files):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, header=None).iloc[0]  # 1行目のデータを取得
        ax.plot(data, label=file)
        ax.set_title(file)
        ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__ == "__main__":
    plot_csv_files_grid("dataset/label0", "Label 0: Sample CSV Data", 3, 3)
    plot_csv_files_grid("dataset/label1", "Label 1: Sample CSV Data", 3, 3)