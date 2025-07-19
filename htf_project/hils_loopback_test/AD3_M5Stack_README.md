# AD3 Ch2 to M5Stack Test System

AnalogDiscovery3のCh2出力からM5Stackのアナログ入力への信号伝送をテストするシステム。3.3V制限とシリアル確認機能付き。

## ハードウェア接続

```
AnalogDiscovery3 接続:
- W2 (ch2出力) → M5Stack GPIO36 (アナログ入力)
- GND → M5Stack GND
- USB → コンピュータ

M5Stack 接続:
- GPIO36 をアナログ入力として設定
- USB → コンピュータ (シリアル監視用)
```

## 使用方法

### 1. M5Stackコードのアップロード

PlatformIOを使用してM5Stackにコードをアップロード:

```bash
# inferenceフォルダに移動してアップロード
cd /home/sota/dev/2025/hils/noise-detector/inference
pio run -t upload
```

または、個別のArduinoファイルを使用する場合:
```bash
# Arduino IDEで m5stack_analog_reader.ino を開く
# ボード設定: M5Stack-Core-ESP32
# 正しいCOMポートを選択してアップロード
```

### 2. テスト実行

```bash
cd /home/sota/dev/2025/hils/noise-detector
source htf_env/bin/activate
cd htf_project/hils_loopback_test

# AD3制御のみ (自動電圧掃引)
python ad3_m5stack_test.py

# M5Stack監視付き協調テスト
python run_ad3_m5stack_test.py
```

## テストモード

### AD3制御のみ (`ad3_m5stack_test.py`)

**自動実行**: 0V～3.3Vの11段階電圧掃引テスト
- 各段階3秒間出力
- M5StackのLCD画面とシリアル出力で確認

**手動モード** (コード変更で選択可能):
1. **DC電圧掃引**: 0V～3.3Vをステップ実行
2. **正弦波テスト**: 周波数・振幅設定可能
3. **手動DC電圧**: 特定電圧を設定

### 協調テスト (`run_ad3_m5stack_test.py`)

1. **DC電圧検証**: 特定電圧をテストしM5Stack読み値を確認
2. **正弦波検出**: 正弦波を生成しM5Stackの応答を監視

## M5Stack機能

### LCD画面表示
- **リアルタイム電圧表示** (大きな文字)
- **最小/最大値追跡**
- **電圧範囲バー** (視覚的表示)
- **ボタン操作ガイド**

### シリアル出力
- **CSV形式**: `timestamp,voltage,raw_adc`
- **115200ボーレート**
- **連続出力** (100msサンプリング)

### ボタン操作
- **Aボタン**: 最小/最大値リセット
- **Bボタン**: 詳細情報表示
- **Cボタン**: 生ADC値表示

## 期待される結果

### DC電圧テスト
```
AD3出力 0.00V  → M5Stack読み取り ~0.00V
AD3出力 1.65V  → M5Stack読み取り ~1.65V  
AD3出力 3.30V  → M5Stack読み取り ~3.30V
```

### 正弦波テスト
```
0.5Hz, 1V振幅, 1.65Vオフセット
範囲: 0.65V ～ 2.65V
M5Stackで変動する電圧読み値を確認
```

## 安全機能

- **3.3V最大制限**: M5Stack安全範囲内に制限
- **電圧検証**: 制限値超過を防止
- **エラーハンドリング**: 例外時の安全停止

## トラブルシューティング

### M5Stackシリアル出力が表示されない
- USB接続を確認
- 正しいCOMポートか確認
- analog_reader.inoがアップロード済みか確認
- ボーレート115200に設定

### 電圧読み値が不正確
- AD3とM5Stack間のGND接続を確認
- W2からGPIO36への接続を確認
- M5Stackの電源確認（起動画面表示）

### AD3接続問題
- AD3のUSB接続を確認
- `python setup_test.py`でデバイス検出確認
- 必要に応じてWaveformsソフト再起動

## 信号確認方法

### M5Stack LCD画面で確認
- 電圧値がリアルタイム更新される
- 最小/最大値が表示される
- 電圧バーが変化する

### シリアル出力で確認
```
M5Stack Analog Input Monitor
Monitoring AD3 Ch2 input on GPIO36
Format: timestamp,voltage,raw_adc
=====================================
1000,0.023,28
2000,0.330,403
3000,0.660,806
4000,0.990,1210
```

フォーマット: `タイムスタンプ_ms,電圧_V,生ADC値`

## ファイル構成

```
htf_project/hils_loopback_test/
├── AD3_M5Stack_README.md          # このファイル
├── ad3_m5stack_test.py             # AD3制御スクリプト
├── run_ad3_m5stack_test.py         # 協調テストスクリプト
├── analog_discovery_plug.py        # AD3デバイス制御クラス
├── m5stack_analog_reader.ino       # M5Stack用Arduinoコード
├── requirements.txt                # Python依存関係
└── test_config.json                # テスト設定ファイル
```

## 実行例

1. **ハードウェア接続**: AD3 W2 → M5Stack GPIO36, GND接続
2. **M5Stackアップロード**: `pio run -t upload`
3. **テスト実行**: `python ad3_m5stack_test.py`
4. **確認**: M5Stack画面とシリアル出力で電圧変化を確認