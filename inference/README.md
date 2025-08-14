# Noise Detector - Inference

M5Stack用の音響分類システムです。リアルタイムで環境音を取得し、ニューラルネットワークを使って34種類の音に分類します。

## 概要

このプロジェクトは、M5Stack Fire上で動作する音響分類システムです。マイクから取得した音響データをニューラルネットワークで処理し、環境音を34クラスに分類します。

## 機能

### 利用可能なモード

1. **Noise Detector** - 基本的なノイズ検出
2. **FFT** - フーリエ変換による周波数解析
3. **Waveform Drawer** - 波形の可視化
4. **Annotation** - 音響データの注釈付け
5. **Neural Network** - 34クラス音響分類（メイン機能）

### 34クラス音響分類

システムは以下の34種類の環境音を識別できます：

- **動物音**: dog, cat, crickets
- **自然音**: rain, sea_waves, wind, water_drops, thunderstorm, crackling_fire
- **人間音**: crying_baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing_teeth, snoring
- **生活音**: door_wood_knock, mouse_click, keyboard_typing, can_opening, washing_machine, vacuum_cleaner, toilet_flush, clock_alarm, clock_tick, glass_breaking, door_wood_creaks
- **交通音**: siren, car_horn, engine, train
- **その他**: silent（無音）

### 特徴

- **リアルタイム分類**: 音響データをリアルタイムで処理
- **Top5表示**: 確信度上位5つの結果を表示
- **確信度による色分け表示**:
  - 緑: 70%以上（高確信）
  - 黄: 40-70%（中確信）
  - 赤: 40%未満（低確信）
- **シリアル出力**: デバッグ用の詳細情報出力

## ハードウェア要件

- M5Stack Fire
- 内蔵マイク
- SDカード（オプション：データ保存用）

## ソフトウェア要件

- PlatformIO
- Arduino Framework (ESP32)
- 依存ライブラリ:
  - M5Stack (v0.4.6)
  - arduinoFFT (v2.0.4)

## ビルド・アップロード

```bash
# プロジェクトのビルド
pio run

# M5Stackへのアップロード
pio run -t upload

# シリアルモニタで動作確認
pio device monitor --baud 115200
```

## 使用方法

1. M5Stackの電源を入れる
2. 起動時にメニューが表示される
3. ボタンでモードを選択：
   - **Aボタン**: 下移動
   - **Bボタン**: 決定
   - **Cボタン**: 上移動
4. Neural Networkモードを選択すると音響分類が開始される

## プロジェクト構成

```
inference/
├── src/
│   ├── main.cpp              # メイン制御
│   ├── config/
│   │   ├── config.h          # システム設定
│   │   └── secret.h          # WiFi設定等
│   ├── core/
│   │   ├── NeuralNetwork.cpp # ニューラルネットワーク処理
│   │   ├── NoiseDetector.cpp # ノイズ検出
│   │   ├── FourierTransform.cpp # FFT処理
│   │   └── WaveformDrawer.cpp # 波形描画
│   └── DeviceHandler/        # デバイス制御
├── lib/
│   └── nnabla-c-runtime/     # ニューラルネットワークランタイム
├── platformio.ini            # PlatformIO設定
└── README.md                 # このファイル
```

## 設定

### config.h の主要設定

```cpp
#define NUM_CLASSES 34           // 分類クラス数
#define CONFIDENCE_THRESHOLD 0.3 // 確信度閾値
#define NN_BEFORE_LEN 200        # 推論用データ長
```

### secret.h の設定（WiFi使用時）

```cpp
#define WIFI_SSID "your_ssid"
#define WIFI_PASSWORD "your_password"
```

## パフォーマンス

- **分類精度**: 79.72%（検証データ）
- **推論時間**: リアルタイム処理対応
- **メモリ使用量**: M5Stack Fire上で動作可能

## トラブルシューティング

### よくある問題

1. **コンパイルエラー**
   - PlatformIOとライブラリが最新版か確認
   - 依存ライブラリが正しくインストールされているか確認

2. **音が検出されない**
   - マイクの接続を確認
   - シリアルモニタでデバッグ情報を確認

3. **分類精度が低い**
   - 環境ノイズの影響を確認
   - 音源とマイクの距離を調整

## 開発・改良

詳細な変更履歴と技術仕様は `UPGRADE_34CLASS.md` を参照してください。

## ライセンス

このプロジェクトは研究・教育目的で開発されています。