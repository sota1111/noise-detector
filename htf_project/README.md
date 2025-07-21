# HTF Project - Hardware-in-the-Loop Testing Framework

音声識別デバイス向けのハードウェア・イン・ザ・ループ（HILS）テストフレームワークです。Analog Discovery 3（AD3）を使用して音声識別デバイスの性能評価を行います。

## プロジェクト概要

このプロジェクトは、音声識別デバイスの自動テストと評価を目的としたシステムです。ESC-50データセットを使用して、AD3デバイス経由でアナログ音声信号を出力し、音声識別デバイスの応答を測定・解析します。

## システム構成

```
htf_project/
├── audio_hils_system/          # 音声HILS評価システム
│   └── waveform/              # 波形処理・テストシステム
│       ├── core/              # コアモジュール（AD3制御、音声処理）
│       ├── dataset/           # データセット管理（ESC-50）
│       ├── identification/    # 音声識別コンポーネント
│       ├── testing/           # テスト実行・レポート生成
│       ├── interface/         # ユーザーインターフェース
│       └── config/            # システム設定
├── output/                    # テスト結果出力
│   ├── csv/                   # 測定データCSV
│   ├── graphs/                # 波形グラフPNG
│   └── reports/               # テストレポートMarkdown
└── manual/                    # ドキュメント・マニュアル
```

## 主な機能

### 🎵 音声処理・出力
- ESC-50データセットからの音声ファイル選択
- AD3デバイス経由でのアナログ音声信号出力
- リアルタイム音声波形記録

### 🧪 テスト実行
- 単一ファイルテスト
- カテゴリ別テスト（動物、自然音、人工音など）
- ランダムファイルテスト
- バッチテスト実行

### 📊 評価・レポート
- 音声識別精度測定
- 応答時間計測
- 統計データ生成（CSV）
- 詳細レポート作成（Markdown）
- 波形グラフ生成

### 🔌 ハードウェア統合
- Analog Discovery 3（AD3）制御
- 高精度アナログ信号生成
- 同期波形記録

## 必要な環境

### ハードウェア
- **Analog Discovery 3（AD3）** - テスト信号生成・測定用
- **音声識別対象デバイス** - 評価対象

### ソフトウェア
- **Python 3.8以上**
- **WaveFormsソフトウェア**（Digilent提供）
- **依存ライブラリ**（requirements.txt参照）

### データセット
- **ESC-50**（Environmental Sound Classification）- オプション

## クイックスタート

### 1. インストール

```bash
# リポジトリクローン
git clone <repository-url>
cd htf_project

# 依存関係インストール
pip install -r audio_hils_system/waveform/requirements.txt

# WaveFormsソフトウェアインストール
# https://digilent.com/reference/software/waveforms/waveforms-3/start
```

### 2. 基本テスト実行

```bash
# htf_projectディレクトリから実行
cd htf_project

# ランダム3ファイルテスト
python3 -m audio_hils_system.waveform.interface.audio_evaluation_system --random 3

# カテゴリ別テスト（犬の音3ファイル）
python3 -m audio_hils_system.waveform.interface.audio_evaluation_system --category dog --max-files 3

# 利用可能なオプション確認
python3 -m audio_hils_system.waveform.interface.audio_evaluation_system --help
```

### 3. テスト結果確認

```bash
# 最新のテストレポート確認
ls -la output/reports/
cat output/reports/session_*.md
```

## 使用例

### ランダム音声テスト
```bash
# 5つのランダム音声ファイルで各5秒間テスト
python3 -m audio_hils_system.waveform.interface.audio_evaluation_system --random 5 --duration 5
```

### カテゴリ別精度テスト
```bash
# サイレン音カテゴリから最大10ファイルテスト
python3 -m audio_hils_system.waveform.interface.audio_evaluation_system --category siren --max-files 10
```

### 単一ファイルテスト
```bash
# 特定音声ファイルの詳細解析
python3 -m audio_hils_system.waveform.interface.audio_evaluation_system --file audio_hils_system/waveform/data/input/sample.wav
```

## テスト結果

### 出力ファイル
- **測定データ**: `output/csv/` - 詳細な波形・識別データ
- **グラフ**: `output/graphs/` - 可視化された波形
- **レポート**: `output/reports/` - 統計・評価結果

## システム設定

### 主要設定項目（config/config.py）
```python
SAMPLE_RATE = 10000      # サンプリングレート（Hz）
OUTPUT_AMP = 1.65        # AD3出力振幅（V）
EXTRACT_DURATION = 1.0   # チャンク処理時間（秒）
AD3_SCOPE_RANGE = 5.0    # スコープ測定範囲（V）
```

## トラブルシューティング

### AD3接続エラー
```
エラー: Analog Discovery 3 デバイスが検出されませんでした
```
**解決方法:**
1. USBケーブル接続確認
2. WaveFormsソフトウェアインストール確認  
3. デバイスドライバー更新

### 音声ファイルエラー
```
音声ファイル検証失敗: モノラルである必要があります
```
**解決方法:**
1. WAVファイル形式確認（16bit モノラル）
2. ファイル長さ確認（1秒以上）
3. サンプリングレート確認

## 開発情報

### アーキテクチャ
- **モジュラー設計**: 機能別フォルダ構成
- **テスト駆動**: 自動テスト・評価機能
- **拡張可能**: 新しい識別器やデータセット対応

### 使用技術
- **Python** - メイン開発言語
- **NumPy/SciPy** - 数値計算・信号処理
- **Matplotlib** - グラフ可視化
- **pydwf** - AD3デバイス制御
- **ESC-50** - 音響データセット

