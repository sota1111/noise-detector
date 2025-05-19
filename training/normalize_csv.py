import pandas as pd
import os

def normalize_csv(input_path, output_path):
    # CSVファイルを読み込む
    df = pd.read_csv(input_path, header=None)
    
    # 数値データの列に対して処理を行う
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        # 2095を引いて2095.0で割る
        df[col] = (df[col] - 2095) / 2095.0
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 新しいCSVファイルとして保存
    df.to_csv(output_path, index=False, header=False, float_format='%.6f')


if __name__ == '__main__':
    input_path = 'dataset/label1/log_2024-12-31_18-03-14.csv'
    output_path = 'dataset_tmp/log_2024-12-31_18-03-14_tmp.csv'
    
    normalize_csv(input_path, output_path)
    print(f'正規化が完了しました。出力ファイル: {output_path}')
