"""
設定ファイル
音声処理とAD3制御に関する設定パラメータ
"""

# フォルダ設定
INPUT_FOLDER = "input"
OUTPUT_CSV_FOLDER = "output/csv"
OUTPUT_GRAPH_FOLDER = "output/graphs"

# 音声処理設定
SAMPLE_RATE = 10000         # 10kHz
EXTRACT_DURATION = 1.0      # 抽出する秒数（1秒）

# AD3設定
OUTPUT_AMP = 1.65           # ±1.65V (3.3V p-p)
AD3_SCOPE_RANGE = 5.0       # スコープ範囲 ±5V
AD3_OFFSET_VOLTAGE = 1.65   # オフセット電圧

# デフォルト音声ファイル候補
DEFAULT_AUDIO_FILES = [
    f"{INPUT_FOLDER}/1-137-A-32.wav",
    "1-137-A-32.wav",
    "ESC-50/audio/1-137-A-32.wav"
]

# グラフ設定
GRAPH_DPI = 300
GRAPH_FIGSIZE = (12, 8)
GRAPH_LINEWIDTH = 0.5
GRAPH_ALPHA = 0.3