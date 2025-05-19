import os
import pandas as pd

# 元のフォルダと出力先のフォルダのリスト
source_dirs = ['dataset_base/label0', 'dataset_base/label1']
target_dirs = ['dataset/label0', 'dataset/label1']

# 各ラベルごとに処理を実施
for src_dir, tgt_dir in zip(source_dirs, target_dirs):
    # 出力先ディレクトリがなければ作成
    os.makedirs(tgt_dir, exist_ok=True)
    
    # 元ディレクトリ内の全CSVファイルを処理
    for filename in os.listdir(src_dir):
        if filename.endswith('.csv'):
            src_file = os.path.join(src_dir, filename)
            # ヘッダーなしで読み込み
            df = pd.read_csv(src_file, header=None)
            
            # 800列分を抽出（Pythonは0-indexなので）
            base_col = 4799
            extracted_df = df.iloc[:, base_col:base_col+800]
            
            # 出力先のファイルパス
            tgt_file = os.path.join(tgt_dir, filename)
            # インデックスやヘッダーは書き込まない
            extracted_df.to_csv(tgt_file, index=False, header=False)
