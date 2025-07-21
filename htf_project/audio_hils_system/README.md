# 音声識別評価システム (Audio Identification Evaluation System)

既存のaudio_hils_systemを活用したAnalog Discovery 3 (AD3)による音声識別デバイスの評価システムです。

## 概要

このシステムは以下の機能を提供します：

1. **音声ファイル選択** - ESC-50データセットから自動選択またはマニュアル指定
2. **AD3アナログ出力** - Analog Discovery3経由で音声をアナログ信号として出力
3. **音声識別デバイステスト** - 音声識別デバイスの応答を記録・解析
4. **テストレポート作成** - 結果をCSVとグラフで保存、評価レポート生成

## システム構成

```
音声識別評価システム
├── 音声ファイル管理 (AudioProcessor + ESC-50データセット)
├── AD3制御 (AD3Controller + pydwf)
├── データ処理・保存 (DataHandler)
├── テスト実行エンジン (TestEngine)
└── レポート生成 (ReportGenerator)
```

## 必要な環境

### ハードウェア
- Analog Discovery 3 (AD3)デバイス
- 音声識別対象デバイス

### ソフトウェア
- Python 3.8以上
- WaveFormsソフトウェア（ADI社提供）
- 必要なPythonライブラリ（`requirements.txt`参照）

### データセット
- ESC-50音響イベント分類データセット（オプション）

## インストール

1. **リポジトリクローン**
```bash
git clone <repository-url>
cd htf_project/audio_hils_system
```

2. **依存関係インストール**
```bash
pip install -r waveform/requirements.txt
```

3. **WaveFormsソフトウェアインストール**
- [Digilent WaveFormsダウンロードページ](https://digilent.com/reference/software/waveforms/waveforms-3/start)からダウンロード・インストール

4. **ESC-50データセット配置（オプション）**
```bash
# ESC-50データセットをダウンロードして配置
wget https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip master.zip
mv ESC-50-master/audio waveform/ESC-50/
```

## 使用方法

### 基本的な使用方法

```bash
cd waveform/

# 利用可能ファイル確認
python audio_evaluation_system.py --list-files

# 利用可能カテゴリ確認
python audio_evaluation_system.py --list-categories

# 単一ファイルテスト（5秒間）
python audio_evaluation_system.py --file input/sample.wav --duration 5

# カテゴリテスト（dogカテゴリから最大3ファイル、各3秒）
python audio_evaluation_system.py --category dog --max-files 3 --duration 3

# ランダムファイルテスト（5ファイル、各3秒）
python audio_evaluation_system.py --random 5 --duration 3
```

### 詳細オプション

| オプション | 説明 | 例 |
|-----------|------|-----|
| `--file` | 単一音声ファイルのテスト | `--file input/test.wav` |
| `--category` | カテゴリ別テスト | `--category dog` |
| `--random N` | ランダムN個ファイルテスト | `--random 5` |
| `--duration N` | テスト時間（秒） | `--duration 3` |
| `--max-files N` | カテゴリテスト最大ファイル数 | `--max-files 3` |
| `--no-report` | レポート生成無効化 | `--no-report` |
| `--list-files` | 利用可能ファイル一覧 | `--list-files` |
| `--list-categories` | 利用可能カテゴリ一覧 | `--list-categories` |

## 出力ファイル

### データファイル
- `output/csv/` - 測定データCSV
- `output/graphs/` - グラフPNG
- `output/reports/` - テストレポートMarkdown

### ファイル命名規則
```
[ファイル名]_[カテゴリ]_[テスト名]_[タイムスタンプ]
例: 1-137-A-32_dog_test_20250121_143022
```

## システム設定

### config.py設定項目

| 設定項目 | 説明 | デフォルト値 |
|---------|------|------------|
| `SAMPLE_RATE` | サンプリングレート | 10000 Hz |
| `OUTPUT_AMP` | AD3出力振幅 | ±1.65V |
| `AD3_SCOPE_RANGE` | スコープ範囲 | ±5V |
| `EXTRACT_DURATION` | チャンク時間 | 1.0秒 |

## トラブルシューティング

### AD3接続エラー
```
エラー: Analog Discovery 3 デバイスが検出されませんでした
```
**解決方法:**
1. AD3デバイスがUSB接続されているか確認
2. WaveFormsソフトウェアがインストールされているか確認
3. デバイスが他のアプリケーションで使用されていないか確認

### 音声ファイルエラー
```
音声ファイル検証失敗: モノラルである必要があります
```
**解決方法:**
1. 音声ファイルがモノラル（1チャンネル）であることを確認
2. 16bit WAVファイルであることを確認
3. 1秒以上の長さがあることを確認

### pydwfインポートエラー
```
警告: pydwfライブラリが利用できません
```
**解決方法:**
```bash
pip install pydwf
```

## API参考

### TestEngine主要メソッド

```python
from test_engine import TestEngine

engine = TestEngine()
engine.initialize()

# 単一ファイルテスト
result = engine.run_single_file_test("sample.wav", duration=5)

# カテゴリテスト
result = engine.run_category_test("dog", max_files=3, duration_per_file=3)

# ランダムテスト
result = engine.run_random_test(file_count=5, duration_per_file=3)

engine.shutdown()
```

### AudioSelector主要メソッド

```python
from audio_selector import AudioSelector

selector = AudioSelector()

# 利用可能ファイル取得
files = selector.get_available_files()

# カテゴリ別ファイル取得
dog_files = selector.get_files_by_category("dog")

# ランダム選択
random_files = selector.select_random_files(count=5)
```

## ライセンス

このプロジェクトは[適切なライセンス]の下で公開されています。

## 開発者情報

- 開発: [開発者名]
- 連絡先: [連絡先情報]
- 最終更新: 2025年7月21日