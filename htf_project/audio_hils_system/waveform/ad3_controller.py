"""
AD3 (Analog Discovery 3) コントローラー
pydwf依存の部分を分離
"""

import numpy as np
import time
import ctypes
from config import OUTPUT_AMP, AD3_SCOPE_RANGE, AD3_OFFSET_VOLTAGE, SAMPLE_RATE, EXTRACT_DURATION

try:
    import pydwf
    PYDWF_AVAILABLE = True
except ImportError:
    PYDWF_AVAILABLE = False
    print("警告: pydwfライブラリが利用できません。ダミーモードで動作します。")


class AD3Controller:
    """Analog Discovery 3 制御クラス"""
    
    def __init__(self):
        self.dwf = None
        self.hdwf = None
        self.is_connected = False
        # 5秒間全体での正規化パラメータ
        self.global_min = None
        self.global_max = None
        # ダミーモードを削除：実機動作のみ
        if not PYDWF_AVAILABLE:
            raise ImportError("pydwfライブラリが必要です。'pip install pydwf'でインストールしてください。")
        
    def connect(self):
        """AD3デバイスに接続"""
        try:
            print("Analog Discovery 3 デバイスに接続試行中...")
            self.dwf = pydwf.DwfLibrary()
            
            # ライブラリのバージョン情報を取得
            try:
                version = ctypes.create_string_buffer(32)
                self.dwf._lib.FDwfGetVersion(version)
                print(f"DWF ライブラリバージョン: {version.value.decode()}")
            except:
                pass
            
            device_count = self.dwf.deviceEnum.enumerateDevices()
            print(f"検出されたデバイス数: {device_count}")
            
            if device_count == 0:
                print("エラー: Analog Discovery 3 デバイスが検出されませんでした。")
                print("確認事項:")
                print("  - デバイスがUSBで接続されているか")
                print("  - WaveFormsソフトウェアがインストールされているか")
                print("  - デバイスが他のアプリケーションで使用されていないか")
                raise RuntimeError("AD3デバイスが必要です。デバイスを接続してください。")
            
            # デバイス情報を表示
            for i in range(device_count):
                try:
                    device_name = ctypes.create_string_buffer(32)
                    device_id = ctypes.create_string_buffer(32)
                    self.dwf._lib.FDwfEnumDeviceName(ctypes.c_int(i), device_name)
                    self.dwf._lib.FDwfEnumSN(ctypes.c_int(i), device_id)
                    print(f"デバイス {i}: {device_name.value.decode()} (S/N: {device_id.value.decode()})")
                except:
                    print(f"デバイス {i}: 情報取得失敗")
            
            # デバイスを開く - 複数の方法を試行
            print("デバイス接続を試行中...")
            
            # 方法1: 高レベルAPI
            try:
                self.hdwf = self.dwf.deviceControl.open(0)
                print("✓ デバイス接続成功 (高レベルAPI)")
                self.is_connected = True
                return True
            except Exception as e1:
                print(f"× 高レベルAPI接続失敗: {e1}")
                
                # 方法2: 低レベルAPI
                try:
                    print("低レベルAPIで再試行中...")
                    hdwf_c = ctypes.c_int()
                    result = self.dwf._lib.FDwfDeviceOpen(ctypes.c_int(0), ctypes.byref(hdwf_c))
                    if result == 1:
                        self.hdwf = hdwf_c.value
                        print("✓ デバイス接続成功 (低レベルAPI)")
                        self.is_connected = True
                        return True
                    else:
                        # エラーコードを取得
                        error_code = ctypes.c_int()
                        self.dwf._lib.FDwfGetLastError(ctypes.byref(error_code))
                        print(f"× 低レベルAPI接続失敗 (エラーコード: {error_code.value})")
                        
                        error_messages = {
                            1: "デバイスが既に使用中です",
                            2: "デバイスのオープンに失敗しました",
                            3: "USB通信エラー",
                            4: "デバイスが見つかりません"
                        }
                        error_msg = error_messages.get(error_code.value, "不明なエラー")
                        print(f"エラー詳細: {error_msg}")
                        
                        raise RuntimeError(f"AD3デバイス接続失敗: {error_msg}")
                        
                except Exception as e2:
                    print(f"× 低レベルAPI接続エラー: {e2}")
                    raise RuntimeError(f"AD3デバイス接続失敗: {e2}")
                    
        except Exception as e:
            print(f"× 予期しないデバイス接続エラー: {e}")
            raise RuntimeError(f"AD3デバイス接続失敗: {e}")
    
    def set_global_normalization_params(self, all_chunks):
        """5秒間全体での正規化パラメータを設定"""
        if not all_chunks:
            self.global_min = 0.0
            self.global_max = 1.0
            return
        
        # 全チャンクの最小値・最大値を計算
        all_data = np.concatenate(all_chunks)
        self.global_min = np.min(all_data)
        self.global_max = np.max(all_data)
        
        print(f"5秒間全体の正規化パラメータ設定:")
        print(f"- 最小値: {self.global_min:.6f}")
        print(f"- 最大値: {self.global_max:.6f}")
    
    def _normalize_waveform(self, waveform_chunk):
        """波形を0~3.3Vの範囲に正規化（5秒間全体の基準で1.65Vを中心）"""
        if self.global_min is None or self.global_max is None:
            # グローバルパラメータが未設定の場合は1.65Vに設定
            return np.ones_like(waveform_chunk) * AD3_OFFSET_VOLTAGE
        
        if self.global_max > self.global_min:
            # 5秒間全体の最小値・最大値を使用して0~3.3Vに正規化
            waveform_normalized = (waveform_chunk - self.global_min) / (self.global_max - self.global_min) * 3.3
        else:
            # 無音の場合は1.65Vに設定
            waveform_normalized = np.ones_like(waveform_chunk) * AD3_OFFSET_VOLTAGE
        
        return waveform_normalized
    
    def _output_and_record_real(self, waveform_chunk, chunk_index):
        """実際のAD3デバイスで出力・記録（高レベルAPI使用）"""
        try:
            waveform_normalized = self._normalize_waveform(waveform_chunk)
            
            # AWG用に-1~1の範囲に変換
            awg_waveform = (waveform_normalized - AD3_OFFSET_VOLTAGE) / OUTPUT_AMP
            
            print(f"チャンク {chunk_index + 1}: 高レベルAPIで実機処理開始")
            
            # 高レベルAPIを使用してAWG設定
            device = self.hdwf
            channel = 0
            
            # AWGリセット
            device.analogOut.reset(channel)
            
            # AWG設定（高レベルAPI） - 正しいEnum値を使用
            import pydwf
            node_carrier = pydwf.DwfAnalogOutNode.Carrier
            func_play = pydwf.DwfAnalogOutFunction.Play
            
            device.analogOut.nodeEnableSet(channel, node_carrier, True)
            device.analogOut.nodeFrequencySet(channel, node_carrier, SAMPLE_RATE)
            device.analogOut.nodeAmplitudeSet(channel, node_carrier, OUTPUT_AMP)
            device.analogOut.nodeOffsetSet(channel, node_carrier, AD3_OFFSET_VOLTAGE)
            
            # カスタム波形データを設定
            device.analogOut.nodeFunctionSet(channel, node_carrier, func_play)
            device.analogOut.nodeDataSet(channel, node_carrier, awg_waveform)
            
            # スコープ設定（高レベルAPI）
            scope_channel = 0
            device.analogIn.channelEnableSet(scope_channel, True)
            device.analogIn.channelRangeSet(scope_channel, AD3_SCOPE_RANGE)
            device.analogIn.frequencySet(SAMPLE_RATE)
            device.analogIn.bufferSizeSet(len(waveform_chunk))
            
            # 同期開始
            device.analogIn.configure(False, True)  # スコープ開始
            device.analogOut.configure(channel, True)  # AWG開始
            
            # 記録完了まで待機（タイムアウト付き）
            timeout = 5.0  # 5秒タイムアウト
            start_time = time.time()
            
            while True:
                if time.time() - start_time > timeout:
                    print(f"タイムアウト: {timeout}秒経過、強制終了")
                    break
                    
                status = device.analogIn.status(True)
                if status == 2:  # DwfStateDone
                    print("記録完了")
                    break
                elif status == 3:  # DwfStateRunning
                    elapsed = time.time() - start_time
                    print(f"\r記録中: {elapsed:.1f}s", end="", flush=True)
                
                time.sleep(0.1)
            
            # データ取得
            captured_data = device.analogIn.statusData(scope_channel, len(waveform_chunk))
            
            print(f"チャンク {chunk_index + 1}: 実際のAD3で出力・記録完了")
            
            return {
                'output_voltage': waveform_normalized,
                'input_voltage': captured_data,
                'time_start': chunk_index * EXTRACT_DURATION
            }
            
        except Exception as e:
            print(f"実際のAD3での出力・記録エラー: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def output_waveform_and_record(self, waveform_chunk, chunk_index):
        """1秒分の波形を出力し、同時に入力を記録"""
        if not self.is_connected:
            raise RuntimeError("デバイスが接続されていません")
        
        result = self._output_and_record_real(waveform_chunk, chunk_index)
        if result is None:
            raise RuntimeError("実機での波形出力・記録に失敗しました")
        return result
    
    
    def disconnect(self):
        """デバイス接続を切断"""
        if not self.is_connected or self.hdwf is None:
            return False
        try:
            # まず高レベルAPIによるクローズを試行
            try:
                self.dwf.deviceControl.close(self.hdwf)
            except Exception:
                # フォールバックで低レベルAPIを利用
                # hdwfハンドルを正しくc_int化
                if isinstance(self.hdwf, int):
                    hdwf_c = ctypes.c_int(self.hdwf)
                elif hasattr(self.hdwf, 'value'):
                    hdwf_c = ctypes.c_int(self.hdwf.value)
                elif hasattr(self.hdwf, '_hdwf'):
                    hdwf_c = ctypes.c_int(self.hdwf._hdwf)
                else:
                    hdwf_c = None

                if hdwf_c is not None and self.dwf and hasattr(self.dwf, '_lib'):
                    self.dwf._lib.FDwfDeviceClose(hdwf_c)

            print("デバイス接続を終了しました")
            return True

        except Exception as e:
            print(f"デバイスクローズエラー: {e}")
            return False

        finally:
            # 状態をリセット
            self.is_connected = False
            self.hdwf = None
