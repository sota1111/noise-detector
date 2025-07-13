# M5Stack Inference Code - 34クラス分類対応アップグレード

## 変更概要

従来の2クラス分類（VOICE/CLAP）から、34クラス環境音分類に対応させました。

## 主な変更点

### 1. NeuralNetwork.cpp
- **getClassName()関数追加**: 34クラスのクラス名取得
- **Top5表示機能**: 全34クラスの代わりに上位5つの結果のみ表示
- **確信度表示**: 予測確信度を色付きで表示（緑：高確信、黄：中確信、赤：低確信）
- **シリアル出力拡張**: クラス名、確信度も出力

### 2. NeuralNetwork.h
- **getClassName()関数宣言**追加

### 3. config.h
- **NUM_CLASSES 34**: クラス数定義
- **CONFIDENCE_THRESHOLD 0.3**: 確信度閾値
- **NN_BEFORE_LEN 200**: 推論用データ長

## 34クラス一覧

| ID | クラス名 | 説明 |
|----|----------|------|
| 0 | dog | 犬の鳴き声 |
| 1 | cat | 猫の鳴き声 |
| 2 | rain | 雨音 |
| 3 | sea_waves | 波音 |
| 4 | crackling_fire | 焚き火音 |
| 5 | crickets | コオロギの鳴き声 |
| 6 | wind | 風音 |
| 7 | water_drops | 水滴音 |
| 8 | thunderstorm | 雷雨 |
| 9 | crying_baby | 赤ちゃんの泣き声 |
| 10 | sneezing | くしゃみ |
| 11 | clapping | 拍手 |
| 12 | breathing | 呼吸音 |
| 13 | coughing | 咳 |
| 14 | footsteps | 足音 |
| 15 | laughing | 笑い声 |
| 16 | brushing_teeth | 歯磨き音 |
| 17 | snoring | いびき |
| 18 | door_wood_knock | ドアのノック音 |
| 19 | mouse_click | マウスクリック音 |
| 20 | keyboard_typing | キーボード入力音 |
| 21 | can_opening | 缶を開ける音 |
| 22 | washing_machine | 洗濯機音 |
| 23 | vacuum_cleaner | 掃除機音 |
| 24 | toilet_flush | 水洗トイレ音 |
| 25 | clock_alarm | 目覚まし時計 |
| 26 | clock_tick | 時計の針音 |
| 27 | glass_breaking | ガラスが割れる音 |
| 28 | door_wood_creaks | ドアのきしみ音 |
| 29 | siren | サイレン |
| 30 | car_horn | 車のクラクション |
| 31 | engine | エンジン音 |
| 32 | train | 電車音 |
| 33 | silent | 無音/静寂 |

## 新機能

### 1. Top5表示
- 全34クラスの代わりに確信度上位5つを表示
- メモリ効率と視認性を向上

### 2. 確信度による色分け表示
- **緑色**: 確信度70%以上（高確信）
- **黄色**: 確信度40-70%（中確信）
- **赤色**: 確信度40%未満（低確信）

### 3. 拡張シリアル出力
```
DETECTED_CLASS:11
CLASS_NAME:clapping
CONFIDENCE:0.852
DB_VALUE:75
```

## 必要なモデルファイル更新

M5Stack上で動作させるには、以下の更新が必要です：

1. **学習済みモデル**: 79.72%精度の34クラス分類モデル
2. **NNabla-C-Runtime**: 34クラス出力対応のランタイム
3. **パラメータファイル**: 更新されたネットワーク重み

## 使用方法

1. 更新されたコードをM5Stackにアップロード
2. 新しいモデルファイルをSDカード/フラッシュメモリに配置
3. M5Stackを起動すると自動的に34クラス分類モードで動作

## パフォーマンス

- **分類精度**: 79.72%（検証データ）
- **推論時間**: 約XX ms（M5Stack Core2）
- **メモリ使用量**: 既存実装と同等

## 今後の改善案

1. **階層分類**: 動物音→犬・猫のような階層構造
2. **適応的閾値**: 環境に応じた確信度閾値調整
3. **時系列解析**: 連続した音の文脈を考慮
4. **省電力モード**: バッテリー駆動時の最適化