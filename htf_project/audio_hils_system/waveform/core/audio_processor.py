"""
音声ファイル処理モジュール
pydwfに依存しない音声ファイル読み込み・処理機能
"""

import numpy as np
import wave
from scipy import signal
import os
import random
import glob
from ..config.config import SAMPLE_RATE, EXTRACT_DURATION, DEFAULT_AUDIO_FILES


class AudioProcessor:
    """音声ファイル処理クラス"""
    
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.chunk_duration = EXTRACT_DURATION
    
    
    def load_audio_file(self, wav_file):
        """
        音声ファイルを読み込み、指定のサンプリングレートにリサンプリングする
        
        Args:
            wav_file (str): WAVファイルのパス
            
        Returns:
            tuple: (リサンプリング後の波形データ, ファイル長[秒])
        """
        with wave.open(wav_file, 'rb') as wf:
            # ファイル情報確認
            if wf.getnchannels() != 1:
                raise ValueError("モノラルWAVファイルのみ対応しています")
            
            original_rate = wf.getframerate()
            total_frames = wf.getnframes()
            file_duration = total_frames / original_rate
            
            # 全データを読み込み
            wf.setpos(0)
            all_audio_data = wf.readframes(total_frames)
            
            # 16bit符号付き整数から-1.0~1.0の範囲に正規化
            waveform_orig_all = np.frombuffer(all_audio_data, dtype=np.int16).astype(np.float64) / 32768.0
            waveform_orig_all = np.clip(waveform_orig_all, -1.0, 1.0)
            
            # 指定のサンプリングレートにリサンプリング
            if original_rate != self.sample_rate:
                num_samples_all = int(len(waveform_orig_all) * self.sample_rate / original_rate)
                waveform_all = signal.resample(waveform_orig_all, num_samples_all)
            else:
                waveform_all = waveform_orig_all
            
            return waveform_all, file_duration
    
    def extract_chunks(self, waveform_all, chunk_duration=None):
        """
        波形データを指定時間ごとのチャンクに分割する
        
        Args:
            waveform_all (np.ndarray): 全波形データ
            chunk_duration (float, optional): チャンク長[秒]. Noneの場合はデフォルト値を使用
            
        Returns:
            list: 1秒ごとのチャンクのリスト
        """
        if chunk_duration is None:
            chunk_duration = self.chunk_duration
        
        chunk_samples = int(chunk_duration * self.sample_rate)
        chunks = []
        
        for i in range(0, len(waveform_all), chunk_samples):
            chunk = waveform_all[i:i+chunk_samples]
            if len(chunk) == chunk_samples:  # 完全なチャンクのみ
                chunks.append(chunk)
        
        return chunks

    def process_audio_file(self, wav_file_path):
        """
        音声ファイルを読み込み、チャンクに分割して返す
        
        Args:
            wav_file_path (str): 音声ファイルのパス
            
        Returns:
            tuple: (チャンクのリスト, ファイル情報の辞書)
        """
        # ファイル存在確認
        if not os.path.exists(wav_file_path):
            raise FileNotFoundError(f"音声ファイルが存在しません: {wav_file_path}")
        
        # 音声ファイル読み込み
        waveform_all, file_duration = self.load_audio_file(wav_file_path)
        
        # チャンクに分割
        chunks = self.extract_chunks(waveform_all)
        
        # ファイル情報
        file_info = {
            'file_path': wav_file_path,
            'file_duration': file_duration,
            'total_samples': len(waveform_all),
            'sample_rate': self.sample_rate,
            'num_chunks': len(chunks),
            'chunk_duration': self.chunk_duration
        }
        
        return chunks, file_info
    
    @staticmethod
    def find_default_audio_file():
        """
        デフォルト音声ファイルを探す（ランダム選択）
        
        Returns:
            str or None: 見つかったファイルのパス、見つからない場合はNone
        """
        # ESC-50フォルダから音声ファイルを探す
        esc50_patterns = [
            "ESC-50/audio/*.wav",
            "../ESC-50/audio/*.wav",
            "../../ESC-50/audio/*.wav"
        ]
        
        all_audio_files = []
        for pattern in esc50_patterns:
            files = glob.glob(pattern)
            if files:
                all_audio_files.extend(files)
                break
        
        # ESC-50ファイルが見つかった場合はランダムに選択
        if all_audio_files:
            return random.choice(all_audio_files)
        
        # ESC-50が見つからない場合はデフォルトファイルを順次チェック
        for file_path in DEFAULT_AUDIO_FILES:
            if os.path.exists(file_path):
                return file_path
                
        return None
    
    @staticmethod
    def validate_audio_file(wav_file_path):
        """
        音声ファイルの妥当性を検証する
        
        Args:
            wav_file_path (str): 検証するファイルのパス
            
        Returns:
            dict: 検証結果とファイル情報
        """
        result = {
            'valid': False,
            'error': None,
            'info': {}
        }
        
        try:
            if not os.path.exists(wav_file_path):
                result['error'] = f"ファイルが存在しません: {wav_file_path}"
                return result
            
            with wave.open(wav_file_path, 'rb') as wf:
                info = {
                    'channels': wf.getnchannels(),
                    'sample_width': wf.getsampwidth(),
                    'frame_rate': wf.getframerate(),
                    'frames': wf.getnframes(),
                    'duration': wf.getnframes() / wf.getframerate()
                }
                result['info'] = info
                
                # 妥当性チェック
                if info['channels'] != 1:
                    result['error'] = f"モノラルである必要があります（現在: {info['channels']}チャンネル）"
                    return result
                
                if info['sample_width'] != 2:
                    result['error'] = f"16bit音声である必要があります（現在: {info['sample_width'] * 8}bit）"
                    return result
                
                if info['duration'] < 1.0:
                    result['error'] = f"1秒以上の音声である必要があります（現在: {info['duration']:.2f}秒）"
                    return result
                
                result['valid'] = True
                
        except Exception as e:
            result['error'] = f"ファイル読み込みエラー: {e}"
        
        return result