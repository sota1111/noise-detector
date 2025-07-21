#!/usr/bin/env python3
"""
音声識別評価システム メインエントリーポイント
"""

import sys
import os
from pathlib import Path

# パッケージパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from audio_hils_system.waveform.interface.audio_evaluation_system import main

if __name__ == "__main__":
    main()