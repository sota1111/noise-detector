# 使い方
## training
trainingディレクトリで実行する

### 学習
``` bash 
python3 classification.py -c cudnn -n conv1d -o output --batch-size 32 --learning-rate 0.001 --max-iter 300 --val-interval 10 --val-iter 20
```
### 推論コードの生成
``` bash 
nnabla_cli convert -O CSRC -b 1 ./output/conv1d_result.nnp ./output_csrc
```
### m5stack用のコードに編集＆ファイル移動
``` bash 
python3 add_const.py 
```

## inference
### ビルド
``` bash 
pio run
```
### 書き込み
``` bash 
pio run -t upload
```
### 書き込めない時に試すこと
``` bash 
pio run -t clean
```
- [書き込めない時に試すこと1](https://scrapbox.io/stack-chan/M5Stack%E3%81%8C%E6%9B%B8%E3%81%8D%E8%BE%BC%E3%82%81%E3%81%AA%E3%81%84%E6%99%82%E3%81%AB%E8%A9%A6%E3%81%99%E3%81%93%E3%81%A8)
- [書き込めない時に試すこと2](https://community.m5stack.com/topic/4821/%E5%88%9D%E5%BF%83%E8%80%85%E5%90%91%E3%81%91-arduinoide%E3%81%A7%E3%82%B3%E3%83%B3%E3%83%91%E3%82%A4%E3%83%AB-%E3%83%93%E3%83%AB%E3%83%89-%E3%82%84%E6%9B%B8%E3%81%8D%E8%BE%BC%E3%81%BF%E3%81%8C%E3%81%A7%E3%81%8D%E3%81%AA%E3%81%84%E6%99%82%E3%81%AB%E8%A6%8B%E3%81%A6%E3%81%8F%E3%81%A0%E3%81%95%E3%81%84)