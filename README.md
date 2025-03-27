# 使い方
## training
trainingディレクトリで実行する

学習
``` bash 
python3 classification.py -c cudnn -n conv1d -o output --batch-size 32 --learning-rate 0.001 --max-iter 300 --val-interval 10 --val-iter 20
```
推論コードの生成
``` bash 
nnabla_cli convert -O CSRC -b 1 ./output/conv1d_result.nnp ./output_csrc
```
m5stack用のコードに編集＆ファイル移動
``` bash 
python3 add_const.py 
```

## inference
``` bash 
python3 add_const.py 
```
