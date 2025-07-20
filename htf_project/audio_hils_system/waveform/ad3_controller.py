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
        self.dummy_mode = not PYDWF_AVAILABLE
        
    def connect(self):
        """AD3デバイスに接続"""
        if self.dummy_mode:
            print("ダミーモード: AD3デバイス接続をシミュレート")
            self.is_connected = True
            return True
            
        try:
            self.dwf = pydwf.DwfLibrary()
            device_count = self.dwf.deviceEnum.enumerateDevices()
            print(f"接続されているデバイス数: {device_count}")
            
            if device_count == 0:
                print("Analog Discovery3デバイスが検出されませんでした。")
                print("ダミーモードに切り替えます。")
                self.dummy_mode = True
                self.is_connected = True
                return True
            
            # デバイスを開く - 複数の方法を試行
            try:
                self.hdwf = self.dwf.deviceControl.open(0)
                print("デバイス接続成功")
                self.is_connected = True
                return True
            except Exception as e1:
                print(f"デバイス接続失敗: {e1}")
                # 低レベルAPI使用を試行
                try:
                    hdwf_c = ctypes.c_int()
                    result = self.dwf._lib.FDwfDeviceOpen(ctypes.c_int(0), ctypes.byref(hdwf_c))
                    if result == 1:
                        self.hdwf = hdwf_c.value
                        print("デバイス接続成功 (低レベルAPI)")
                        self.is_connected = True
                        return True
                    else:
                        print("低レベルAPI接続も失敗")
                        print("ダミーモードに切り替えます。")
                        self.dummy_mode = True
                        self.is_connected = True
                        return True
                except Exception as e2:
                    print(f"低レベルAPI接続エラー: {e2}")
                    print("ダミーモードに切り替えます。")
                    self.dummy_mode = True
                    self.is_connected = True
                    return True
                    
        except Exception as e:
            print(f"デバイス接続エラー: {e}")
            print("ダミーモードに切り替えます。")
            self.dummy_mode = True
            self.is_connected = True
            return True
    
    def _normalize_waveform(self, waveform_chunk):
        """波形を0~3.3Vの範囲に正規化"""
        min_val = np.min(waveform_chunk)
        max_val = np.max(waveform_chunk)
        if max_val > min_val:
            waveform_normalized = (waveform_chunk - min_val) / (max_val - min_val) * 3.3
        else:
            waveform_normalized = np.ones_like(waveform_chunk) * AD3_OFFSET_VOLTAGE
        return waveform_normalized
    
    def _output_and_record_real(self, waveform_chunk, chunk_index):
        """実際のAD3デバイスで出力・記録"""
        try:
            waveform_normalized = self._normalize_waveform(waveform_chunk)
            
            # AWG用に-1~1の範囲に変換
            awg_waveform = (waveform_normalized - AD3_OFFSET_VOLTAGE) / OUTPUT_AMP
            
            # 低レベルAPIでAWG設定
            lib = self.dwf._lib
            channel = 0
            hdwf_c = ctypes.c_int(self.hdwf) if isinstance(self.hdwf, int) else ctypes.c_int(self.hdwf.value)
            
            # AWGリセット
            lib.FDwfAnalogOutReset(hdwf_c, ctypes.c_int(channel))
            
            # AWG設定
            lib.FDwfAnalogOutNodeEnableSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_bool(True))
            lib.FDwfAnalogOutNodeFunctionSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_int(31))  # funcPlay
            lib.FDwfAnalogOutNodeFrequencySet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(SAMPLE_RATE))
            lib.FDwfAnalogOutNodeAmplitudeSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(OUTPUT_AMP))
            lib.FDwfAnalogOutNodeOffsetSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(AD3_OFFSET_VOLTAGE))
            
            # データ設定
            rgdData = (ctypes.c_double * len(awg_waveform))(*awg_waveform)
            lib.FDwfAnalogOutNodeDataSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), rgdData, ctypes.c_int(len(awg_waveform)))
            
            # スコープ設定（入力記録用）
            scope_channel = 0
            lib.FDwfAnalogInChannelEnableSet(hdwf_c, ctypes.c_int(scope_channel), ctypes.c_bool(True))
            lib.FDwfAnalogInChannelRangeSet(hdwf_c, ctypes.c_int(scope_channel), ctypes.c_double(AD3_SCOPE_RANGE))
            lib.FDwfAnalogInFrequencySet(hdwf_c, ctypes.c_double(SAMPLE_RATE))
            lib.FDwfAnalogInBufferSizeSet(hdwf_c, ctypes.c_int(len(waveform_chunk)))
            
            # 同期開始
            lib.FDwfAnalogInConfigure(hdwf_c, ctypes.c_bool(False), ctypes.c_bool(True))  # スコープ開始
            lib.FDwfAnalogOutConfigure(hdwf_c, ctypes.c_int(channel), ctypes.c_bool(True))  # AWG開始
            
            # 記録完了まで待機
            sts = ctypes.c_byte()
            while True:
                lib.FDwfAnalogInStatus(hdwf_c, ctypes.c_bool(True), ctypes.byref(sts))
                if sts.value == 2:  # DwfStateDone
                    break
                time.sleep(0.01)
            
            # データ取得
            rgdSamples = (ctypes.c_double * len(waveform_chunk))()
            lib.FDwfAnalogInStatusData(hdwf_c, ctypes.c_int(scope_channel), rgdSamples, ctypes.c_int(len(waveform_chunk)))
            captured_data = np.array(rgdSamples)
            
            print(f"チャンク {chunk_index + 1}: 実際のAD3で出力・記録完了")
            
            return {
                'output_voltage': waveform_normalized,
                'input_voltage': captured_data,
                'time_start': chunk_index * EXTRACT_DURATION
            }
            
        except Exception as e:
            print(f"実際のAD3での出力・記録エラー: {e}")
            return None
    
    def _output_and_record_dummy(self, waveform_chunk, chunk_index):
        """ダミーモードでの出力・記録シミュレーション"""
        waveform_normalized = self._normalize_waveform(waveform_chunk)
        
        # ダミーデータ生成（少しノイズを追加）
        captured_data = waveform_normalized + np.random.normal(0, 0.01, len(waveform_normalized))
        
        # 処理時間をシミュレート
        time.sleep(0.1)
        
        print(f"チャンク {chunk_index + 1}: ダミーモードで出力・記録完了")
        
        return {
            'output_voltage': waveform_normalized,
            'input_voltage': captured_data,
            'time_start': chunk_index * EXTRACT_DURATION
        }
    
    def output_waveform_and_record(self, waveform_chunk, chunk_index):
        """1秒分の波形を出力し、同時に入力を記録"""
        if not self.is_connected:
            print("デバイスが接続されていません")
            return None
        
        if self.dummy_mode:
            return self._output_and_record_dummy(waveform_chunk, chunk_index)
        else:
            result = self._output_and_record_real(waveform_chunk, chunk_index)
            if result is None:
                # 実デバイスで失敗した場合はダミーモードにフォールバック
                print("実デバイスでの処理に失敗しました。ダミーモードにフォールバック。")
                self.dummy_mode = True
                return self._output_and_record_dummy(waveform_chunk, chunk_index)
            return result
    
    def output_continuous_and_record(self, waveform_chunks):
        """連続した複数チャンクの波形を出力し、全体を連続記録"""
        if not self.is_connected:
            print("デバイスが接続されていません")
            return None
        
        if self.dummy_mode:
            return self._output_continuous_dummy(waveform_chunks)
        else:
            result = self._output_continuous_real(waveform_chunks)
            if result is None:
                print("実デバイスでの連続処理に失敗しました。ダミーモードにフォールバック。")
                self.dummy_mode = True
                return self._output_continuous_dummy(waveform_chunks)
            return result
    
    def _output_continuous_real(self, waveform_chunks):
        """実際のAD3デバイスで連続出力・記録"""
        try:
            # 全チャンクを結合
            all_waveform = np.concatenate(waveform_chunks)
            total_samples = len(all_waveform)
            total_duration = len(waveform_chunks) * EXTRACT_DURATION
            
            # 正規化
            waveform_normalized = self._normalize_waveform(all_waveform)
            awg_waveform = (waveform_normalized - AD3_OFFSET_VOLTAGE) / OUTPUT_AMP
            
            # 低レベルAPIでAWG設定
            lib = self.dwf._lib
            channel = 0
            hdwf_c = ctypes.c_int(self.hdwf) if isinstance(self.hdwf, int) else ctypes.c_int(self.hdwf.value)
            
            # AWGリセット
            lib.FDwfAnalogOutReset(hdwf_c, ctypes.c_int(channel))
            
            # AWG設定
            lib.FDwfAnalogOutNodeEnableSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_bool(True))
            lib.FDwfAnalogOutNodeFunctionSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_int(31))  # funcPlay
            lib.FDwfAnalogOutNodeFrequencySet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(SAMPLE_RATE))
            lib.FDwfAnalogOutNodeAmplitudeSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(OUTPUT_AMP))
            lib.FDwfAnalogOutNodeOffsetSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(AD3_OFFSET_VOLTAGE))
            
            # データ設定
            rgdData = (ctypes.c_double * total_samples)(*awg_waveform)
            lib.FDwfAnalogOutNodeDataSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), rgdData, ctypes.c_int(total_samples))
            
            # スコープ設定（連続記録用）
            scope_channel = 0
            lib.FDwfAnalogInChannelEnableSet(hdwf_c, ctypes.c_int(scope_channel), ctypes.c_bool(True))
            lib.FDwfAnalogInChannelRangeSet(hdwf_c, ctypes.c_int(scope_channel), ctypes.c_double(AD3_SCOPE_RANGE))
            lib.FDwfAnalogInFrequencySet(hdwf_c, ctypes.c_double(SAMPLE_RATE))
            lib.FDwfAnalogInBufferSizeSet(hdwf_c, ctypes.c_int(total_samples))
            
            print(f"連続出力開始: {total_duration}秒間, {total_samples}サンプル")
            
            # 同期開始
            lib.FDwfAnalogInConfigure(hdwf_c, ctypes.c_bool(False), ctypes.c_bool(True))  # スコープ開始
            lib.FDwfAnalogOutConfigure(hdwf_c, ctypes.c_int(channel), ctypes.c_bool(True))  # AWG開始
            
            # 記録完了まで待機
            sts = ctypes.c_byte()
            start_time = time.time()
            while True:
                lib.FDwfAnalogInStatus(hdwf_c, ctypes.c_bool(True), ctypes.byref(sts))
                if sts.value == 2:  # DwfStateDone
                    break
                
                elapsed = time.time() - start_time
                print(f"\r進行状況: {elapsed:.1f}s / {total_duration:.1f}s", end="", flush=True)
                time.sleep(0.1)
            
            print(f"\n連続記録完了")
            
            # データ取得
            rgdSamples = (ctypes.c_double * total_samples)()
            lib.FDwfAnalogInStatusData(hdwf_c, ctypes.c_int(scope_channel), rgdSamples, ctypes.c_int(total_samples))
            captured_data = np.array(rgdSamples)
            
            return {
                'output_voltage': waveform_normalized,
                'input_voltage': captured_data,
                'time_start': 0.0,
                'total_duration': total_duration,
                'num_chunks': len(waveform_chunks)
            }
            
        except Exception as e:
            print(f"実際のAD3での連続出力・記録エラー: {e}")
            return None
    
    def _output_continuous_dummy(self, waveform_chunks):
        """ダミーモードでの連続出力・記録シミュレーション"""
        # 全チャンクを結合
        all_waveform = np.concatenate(waveform_chunks)
        total_duration = len(waveform_chunks) * EXTRACT_DURATION
        
        # 正規化
        waveform_normalized = self._normalize_waveform(all_waveform)
        
        # ダミーデータ生成（少しノイズを追加）
        captured_data = waveform_normalized + np.random.normal(0, 0.01, len(waveform_normalized))
        
        # 処理時間をシミュレート
        print(f"ダミーモード: 連続出力開始 {total_duration}秒間")
        for i in range(int(total_duration * 2)):  # 0.5秒ごとに進行状況表示
            time.sleep(0.5)
            progress = (i + 1) * 0.5
            print(f"\r進行状況: {progress:.1f}s / {total_duration:.1f}s", end="", flush=True)
        
        print(f"\nダミーモード: 連続記録完了")
        
        return {
            'output_voltage': waveform_normalized,
            'input_voltage': captured_data,
            'time_start': 0.0,
            'total_duration': total_duration,
            'num_chunks': len(waveform_chunks)
        }
    
    def disconnect(self):
        """デバイス接続を切断"""
        if self.dummy_mode:
            print("ダミーモード: デバイス切断をシミュレート")
            self.is_connected = False
            return
            
        if self.is_connected and self.hdwf is not None:
            try:
                if isinstance(self.hdwf, int):
                    if self.dwf and hasattr(self.dwf, '_lib'):
                        self.dwf._lib.FDwfDeviceClose(ctypes.c_int(self.hdwf))
                else:
                    # 高レベルAPIでのクローズを試行
                    try:
                        self.dwf.deviceControl.close(self.hdwf)
                    except AttributeError:
                        # 属性が存在しない場合は低レベルAPIを使用
                        if hasattr(self.hdwf, 'value'):
                            self.dwf._lib.FDwfDeviceClose(ctypes.c_int(self.hdwf.value))
                        else:
                            self.dwf._lib.FDwfDeviceClose(ctypes.c_int(int(self.hdwf)))
                print("デバイス接続を終了しました")
            except Exception as e:
                print(f"デバイスクローズエラー: {e}")
            finally:
                self.is_connected = False