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
        """実際のAD3デバイスで出力・記録（高レベルAPI使用）"""
        try:
            waveform_normalized = self._normalize_waveform(waveform_chunk)
            
            # AWG用に-1~1の範囲に変換
            awg_waveform = (waveform_normalized - AD3_OFFSET_VOLTAGE) / OUTPUT_AMP
            awg_waveform = np.clip(awg_waveform, -1.0, 1.0)
            
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
    
    def output_continuous_and_record(self, waveform_chunks):
        """連続した複数チャンクの波形を出力し、全体を連続記録"""
        if not self.is_connected:
            raise RuntimeError("デバイスが接続されていません")
        
        result = self._output_continuous_real(waveform_chunks)
        if result is None:
            raise RuntimeError("実機での連続処理に失敗しました")
        return result
    
    def _output_continuous_real(self, waveform_chunks):
        """実際のAD3デバイスで連続出力・記録"""
        try:
            # 全チャンクを結合
            all_waveform = np.concatenate(waveform_chunks)
            total_samples = len(all_waveform)
            total_duration = len(waveform_chunks) * EXTRACT_DURATION
            
            # バッファサイズの制限チェック（AD3の制限を考慮）
            max_buffer_size = 100000  # AD3の最大バッファサイズ
            if total_samples > max_buffer_size:
                print(f"警告: サンプル数が多すぎます ({total_samples} > {max_buffer_size})")
                print("チャンクベースの処理に切り替えます")
                return self._output_chunks_real(waveform_chunks)
            
            # 正規化
            waveform_normalized = self._normalize_waveform(all_waveform)
            awg_waveform = (waveform_normalized - AD3_OFFSET_VOLTAGE) / OUTPUT_AMP
            
            # クリッピング防止
            awg_waveform = np.clip(awg_waveform, -1.0, 1.0)
            
            # 低レベルAPIでAWG設定
            lib = self.dwf._lib
            channel = 0
            # hdwfハンドルの適切な取得
            if isinstance(self.hdwf, int):
                hdwf_c = ctypes.c_int(self.hdwf)
            elif hasattr(self.hdwf, 'value'):
                hdwf_c = ctypes.c_int(self.hdwf.value)
            elif hasattr(self.hdwf, 'hdwf'):
                hdwf_c = ctypes.c_int(self.hdwf.hdwf)
            else:
                # DwfDeviceオブジェクトから実際のハンドルを取得
                try:
                    # pydwfのDwfDeviceオブジェクトの内部ハンドルを取得
                    hdwf_c = ctypes.c_int(self.hdwf._hdwf)
                except AttributeError:
                    print(f"警告: hdwfハンドルの取得に失敗しました。オブジェクト型: {type(self.hdwf)}")
                    raise Exception("hdwfハンドルを取得できません")
            
            # デバイスリセット
            lib.FDwfDeviceAutoConfigureSet(hdwf_c, ctypes.c_int(0))  # 自動設定無効
            
            # AWGリセット
            lib.FDwfAnalogOutReset(hdwf_c, ctypes.c_int(channel))
            
            # AWG設定（c_boolをc_intに変更）
            lib.FDwfAnalogOutNodeEnableSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_int(1))  # True=1
            lib.FDwfAnalogOutNodeFunctionSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_int(31))  # funcPlay
            lib.FDwfAnalogOutNodeFrequencySet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(SAMPLE_RATE))
            lib.FDwfAnalogOutNodeAmplitudeSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(OUTPUT_AMP))
            lib.FDwfAnalogOutNodeOffsetSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), ctypes.c_double(AD3_OFFSET_VOLTAGE))
            lib.FDwfAnalogOutRepeatSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(1))  # 1回のみ再生
            
            # データ設定
            rgdData = (ctypes.c_double * total_samples)(*awg_waveform)
            lib.FDwfAnalogOutNodeDataSet(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0), rgdData, ctypes.c_int(total_samples))
            
            # スコープ設定（連続記録用）
            scope_channel = 0
            lib.FDwfAnalogInReset(hdwf_c)
            lib.FDwfAnalogInChannelEnableSet(hdwf_c, ctypes.c_int(scope_channel), ctypes.c_int(1))
            lib.FDwfAnalogInChannelRangeSet(hdwf_c, ctypes.c_int(scope_channel), ctypes.c_double(AD3_SCOPE_RANGE))
            lib.FDwfAnalogInFrequencySet(hdwf_c, ctypes.c_double(SAMPLE_RATE))
            lib.FDwfAnalogInBufferSizeSet(hdwf_c, ctypes.c_int(total_samples))
            lib.FDwfAnalogInTriggerSourceSet(hdwf_c, ctypes.c_int(0))  # triggerNone
            
            print(f"連続出力開始: {total_duration}秒間, {total_samples}サンプル")
            
            # 同期開始
            lib.FDwfAnalogInConfigure(hdwf_c, ctypes.c_int(0), ctypes.c_int(1))  # スコープ開始
            lib.FDwfAnalogOutConfigure(hdwf_c, ctypes.c_int(channel), ctypes.c_int(1))  # AWG開始
            
            # 記録完了まで待機（タイムアウト付き）
            sts = ctypes.c_byte()
            start_time = time.time()
            timeout = total_duration + 10.0  # 余裕を持ったタイムアウト
            
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                
                if elapsed > timeout:
                    print(f"\nタイムアウトが発生しました（{timeout}秒）")
                    lib.FDwfAnalogInConfigure(hdwf_c, ctypes.c_int(0), ctypes.c_int(0))  # 停止
                    lib.FDwfAnalogOutConfigure(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0))  # 停止
                    return None
                
                lib.FDwfAnalogInStatus(hdwf_c, ctypes.c_int(1), ctypes.byref(sts))
                if sts.value == 2:  # DwfStateDone
                    break
                elif sts.value == 1:  # DwfStateArmed
                    print(f"\r待機中: {elapsed:.1f}s", end="", flush=True)
                elif sts.value == 3:  # DwfStateRunning
                    print(f"\r記録中: {elapsed:.1f}s / {total_duration:.1f}s", end="", flush=True)
                
                time.sleep(0.1)
            
            print(f"\n連続記録完了")
            
            # データ取得
            rgdSamples = (ctypes.c_double * total_samples)()
            lib.FDwfAnalogInStatusData(hdwf_c, ctypes.c_int(scope_channel), rgdSamples, ctypes.c_int(total_samples))
            captured_data = np.array(rgdSamples)
            
            # 停止
            lib.FDwfAnalogOutConfigure(hdwf_c, ctypes.c_int(channel), ctypes.c_int(0))
            
            return {
                'output_voltage': waveform_normalized,
                'input_voltage': captured_data,
                'time_start': 0.0,
                'total_duration': total_duration,
                'num_chunks': len(waveform_chunks)
            }
            
        except Exception as e:
            print(f"実際のAD3での連続出力・記録エラー: {e}")
            # エラー時は停止処理
            try:
                if 'lib' in locals() and 'hdwf_c' in locals():
                    lib.FDwfAnalogInConfigure(hdwf_c, ctypes.c_int(0), ctypes.c_int(0))
                    lib.FDwfAnalogOutConfigure(hdwf_c, ctypes.c_int(0), ctypes.c_int(0))
            except:
                pass
            return None
    
    def _output_chunks_real(self, waveform_chunks):
        """チャンクベースの実機処理（大きなデータ用）"""
        try:
            all_output_data = []
            all_input_data = []
            
            print(f"チャンクベース処理開始: {len(waveform_chunks)}チャンク")
            
            for i, chunk in enumerate(waveform_chunks):
                print(f"チャンク {i+1}/{len(waveform_chunks)} 処理中...")
                result = self._output_and_record_real(chunk, i)
                if result is None:
                    print(f"チャンク {i+1} の処理に失敗しました")
                    return None
                
                all_output_data.append(result['output_voltage'])
                all_input_data.append(result['input_voltage'])
            
            # 全データを結合
            combined_output = np.concatenate(all_output_data)
            combined_input = np.concatenate(all_input_data)
            total_duration = len(waveform_chunks) * EXTRACT_DURATION
            
            print("チャンクベース処理完了")
            
            return {
                'output_voltage': combined_output,
                'input_voltage': combined_input,
                'time_start': 0.0,
                'total_duration': total_duration,
                'num_chunks': len(waveform_chunks)
            }
            
        except Exception as e:
            print(f"チャンクベース処理エラー: {e}")
            return None
    
    def disconnect(self):
        """デバイス接続を切断"""
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