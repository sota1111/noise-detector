#!/bin/bash
set -e

echo "=== 移動: training ディレクトリ ==="
cd training

echo "=== モデルのトレーニングを開始 ==="
python3 classification.py -c cudnn -n conv1d -o output --batch-size 32 --learning-rate 0.001 --max-iter 200 --val-interval 10 --val-iter 20

echo "=== 軽量化されたモデルを正しいディレクトリにコピー ==="
cp tmp.monitor/conv1d_result.nnp output/conv1d_result.nnp

echo "=== NNablaモデルをM5Stack対応のCSRC形式に変換 ==="
nnabla_cli convert -O CSRC -b 1 ./output/conv1d_result.nnp ./output_csrc

echo "=== M5Stack対応のための定数追加とコード修正を実行 ==="
python3 add_const.py

echo "=== 変換されたファイルの確認 ==="
if [ -f "output_csrc/Validation_inference.c" ] && [ -f "output_csrc/Validation_parameters.c" ] && 
   [ -f "../inference/src/core/Validation/Validation_inference.c" ] && [ -f "../inference/src/core/Validation/Validation_parameters.c" ]; then
    echo "M5Stack対応のファイル変換が完了しました"
    
    # 出力クラス数の確認
    output_size=$(grep "NNABLART_VALIDATION_OUTPUT0_SIZE" ../inference/src/core/Validation/Validation_inference.h | grep -o '[0-9]\+')
    echo "モデルの出力クラス数: $output_size"
    
    if [ "$output_size" = "34" ]; then
        echo "✓ 正しく34クラスモデルが生成されました"
    else
        echo "⚠ 警告: 出力クラス数が$output_size です（期待値: 34）"
        echo "classification.pyで34クラスの学習が正しく行われているか確認してください"
    fi
else
    echo "エラー: 変換ファイルが見つかりません"
    exit 1
fi

echo "=== inference ディレクトリに移動 ==="
cd ../inference

echo "=== M5StackへのPlatformIOアップロードを実行 ==="
pio run -t upload

echo "=== M5Stack対応モデルの作成とアップロードが正常に完了しました ==="
echo "=== モデルはM5Stackで動作可能です ==="
