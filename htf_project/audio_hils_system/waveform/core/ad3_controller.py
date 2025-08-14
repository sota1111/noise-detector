"""
AD3 (Analog Discovery 3) コントローラー
pydwf依存の部分を分離
"""

import numpy as np
import time
import ctypes
from ..config.config import OUTPUT_AMP, AD3_SCOPE_RANGE, AD3_OFFSET_VOLTAGE, SAMPLE_RATE, EXTRACT_DURATION

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
            
            # 方法1: 高レベルAPI（タイムアウト付き）
            try:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("デバイス接続タイムアウト")
                
                # 5秒タイムアウトを設定
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                
                try:
                    self.hdwf = self.dwf.deviceControl.open(0)
                    signal.alarm(0)  # タイムアウト解除
                    print("✓ デバイス接続成功 (高レベルAPI)")
                    self.is_connected = True
                    return True
                finally:
                    signal.alarm(0)  # 確実にタイムアウト解除
                    
            except (Exception, TimeoutError) as e1:
                print(f"× 高レベルAPI接続失敗: {e1}")
                
                # 方法2: 低レベルAPI（タイムアウト付き）
                try:
                    print("低レベルAPIで再試行中...")
                    
                    # タイムアウト設定
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(5)
                    
                    try:
                        hdwf_c = ctypes.c_int()
                        result = self.dwf._lib.FDwfDeviceOpen(ctypes.c_int(0), ctypes.byref(hdwf_c))
                        signal.alarm(0)  # タイムアウト解除
                        
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
                    finally:
                        signal.alarm(0)  # 確実にタイムアウト解除
                        
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
        f"""波形を{AD3_OFFSET_VOLTAGE - OUTPUT_AMP}~{AD3_OFFSET_VOLTAGE + OUTPUT_AMP}Vの範囲に正規化（5秒間全体の基準で{AD3_OFFSET_VOLTAGE}Vを中心）"""
        if self.global_min is None or self.global_max is None:
            # グローバルパラメータが未設定の場合は1.65Vに設定
            return np.ones_like(waveform_chunk) * AD3_OFFSET_VOLTAGE
        
        if self.global_max > self.global_min:
            # 5秒間全体の最小値・最大値を使用して電圧範囲に正規化
            voltage_range = 2 * OUTPUT_AMP  # 電圧幅
            min_voltage = AD3_OFFSET_VOLTAGE - OUTPUT_AMP  # 最小電圧
            waveform_normalized = (waveform_chunk - self.global_min) / (self.global_max - self.global_min) * voltage_range + min_voltage
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
            
            print("AWG設定中...")
            try:
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
                
                # 設定確認
                actual_freq = device.analogOut.nodeFrequencyGet(channel, node_carrier)
                actual_amp = device.analogOut.nodeAmplitudeGet(channel, node_carrier)
                print(f"AWG設定確認: {actual_freq:.0f}Hz, {actual_amp:.3f}V振幅")
                
            except Exception as awg_error:
                print(f"× AWG設定エラー: {awg_error}")
                return None
            
            # スコープ設定（高レベルAPI）
            scope_channel = 0
            print("スコープ設定中...")
            try:
                device.analogIn.reset()  # スコープリセット
                device.analogIn.channelEnableSet(scope_channel, True)
                device.analogIn.channelRangeSet(scope_channel, AD3_SCOPE_RANGE)
                device.analogIn.frequencySet(SAMPLE_RATE)
                device.analogIn.bufferSizeSet(len(waveform_chunk))
                
                # 設定が正しく適用されたか確認
                actual_freq = device.analogIn.frequencyGet()
                actual_buffer = device.analogIn.bufferSizeGet()
                print(f"スコープ設定確認: {actual_freq:.0f}Hz, {actual_buffer}サンプル")
                
            except Exception as scope_error:
                print(f"× スコープ設定エラー: {scope_error}")
                return None
            
            # スコープのトリガー設定を追加
            print("トリガー設定中...")
            try:
                # 自動トリガーに設定
                import pydwf
                device.analogIn.triggerAutoTimeoutSet(0)  # 自動トリガー
                device.analogIn.triggerSourceSet(pydwf.DwfTriggerSource.None_)  # 自動トリガー
                print("✓ トリガー設定完了")
            except Exception as trigger_error:
                print(f"× トリガー設定エラー: {trigger_error}")
                # トリガー設定に失敗しても継続（デフォルト設定を使用）
                print("デフォルトトリガー設定で継続")
            
            # 同期開始（段階的に実行）
            print("デバイス同期開始...")
            try:
                # まずスコープを単発記録モードに設定
                device.analogIn.configure(False, True)  # 単発記録開始
                time.sleep(0.1)  # 設定安定化待機
                
                # AWGを開始（これで記録が開始される）
                device.analogOut.configure(channel, True)
                time.sleep(0.1)  # 開始安定化待機
                
                print("✓ デバイス同期開始完了")
                
            except Exception as sync_error:
                print(f"× デバイス同期開始エラー: {sync_error}")
                return None
            
            # 記録完了まで待機（タイムアウト付き）
            timeout = 10.0  # タイムアウトを10秒に延長
            start_time = time.time()
            recording_completed = False
            last_status = None
            
            print("記録開始待機...")
            
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                
                if elapsed > timeout:
                    print(f"\n× タイムアウト: {timeout}秒経過、処理を中断")
                    print(f"最終ステータス: {last_status}")
                    # デバイスを停止
                    try:
                        device.analogIn.configure(False, False)  # スコープ停止
                        device.analogOut.configure(channel, False)  # AWG停止
                        print("✓ デバイス停止完了")
                    except Exception as stop_error:
                        print(f"× デバイス停止エラー: {stop_error}")
                    return None
                    
                try:
                    status = device.analogIn.status(True)
                    last_status = status
                    
                    # pydwf.DwfStateのEnum値を適切に処理
                    import pydwf
                    
                    if status == pydwf.DwfState.Done:  # 完了
                        # Done状態が初回の場合のみ記録完了とする（連続Done防止）
                        if elapsed > 0.8:  # 0.8秒以上経過してからのDoneのみ有効
                            print(f"\n✓ 記録完了 ({elapsed:.1f}秒)")
                            recording_completed = True
                            break
                        else:
                            print(f"\r準備中: {elapsed:.1f}s (初期Done)", end="", flush=True)
                    elif status == pydwf.DwfState.Running:  # 実行中
                        print(f"\r記録中: {elapsed:.1f}s (実行中)", end="", flush=True)
                    elif status == pydwf.DwfState.Armed:  # 準備完了
                        print(f"\r待機中: {elapsed:.1f}s (準備完了)", end="", flush=True)
                    elif status == pydwf.DwfState.Ready:  # 準備中
                        print(f"\r初期化中: {elapsed:.1f}s (準備中)", end="", flush=True)
                    elif status == pydwf.DwfState.Triggered:  # トリガー待機
                        print(f"\r記録中: {elapsed:.1f}s (トリガー中)", end="", flush=True)
                    elif status == pydwf.DwfState.Prefill:  # プリフィル
                        print(f"\r準備中: {elapsed:.1f}s (プリフィル)", end="", flush=True)
                    else:
                        print(f"\r不明状態: {elapsed:.1f}s (status={status})", end="", flush=True)
                    
                except Exception as status_error:
                    print(f"\n× ステータス取得エラー: {status_error}")
                    # ハードウェアエラー時も停止
                    try:
                        device.analogIn.configure(False, False)
                        device.analogOut.configure(channel, False)
                    except:
                        pass
                    return None
                
                time.sleep(0.2)  # ポーリング間隔を少し長めに
            
            # 記録が正常に完了した場合のみデータ取得
            if not recording_completed:
                print("× 記録が正常に完了しませんでした")
                return None
                
            try:
                # データ取得
                captured_data = device.analogIn.statusData(scope_channel, len(waveform_chunk))
            except Exception as data_error:
                print(f"× データ取得エラー: {data_error}")
                return None
            
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
            print("デバイスは既に切断されています")
            return True
            
        disconnect_success = False
        
        try:
            # デバイスを停止してからクローズ
            try:
                # 進行中の処理を停止
                if hasattr(self.hdwf, 'analogIn'):
                    self.hdwf.analogIn.configure(False, False)  # スコープ停止
                if hasattr(self.hdwf, 'analogOut'):
                    for channel in range(2):  # チャンネル0,1を停止
                        try:
                            self.hdwf.analogOut.configure(channel, False)
                        except:
                            pass
            except Exception as stop_error:
                print(f"× デバイス停止エラー（継続）: {stop_error}")
            
            # まず高レベルAPIによるクローズを試行
            try:
                self.dwf.deviceControl.close(self.hdwf)
                disconnect_success = True
                print("✓ 高レベルAPIでデバイス切断完了")
            except Exception as high_level_error:
                print(f"× 高レベルAPI切断失敗: {high_level_error}")
                
                # フォールバックで低レベルAPIを利用
                try:
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
                        disconnect_success = True
                        print("✓ 低レベルAPIでデバイス切断完了")
                    else:
                        print("× 低レベルAPI切断: ハンドル取得失敗")
                        
                except Exception as low_level_error:
                    print(f"× 低レベルAPI切断失敗: {low_level_error}")

            return disconnect_success

        except Exception as e:
            print(f"× デバイス切断エラー: {e}")
            return False

        finally:
            # エラーが発生してもリソースを強制的にリセット
            self.is_connected = False
            self.hdwf = None
            if disconnect_success:
                print("✓ デバイスリソース解放完了")
            else:
                print("× デバイス切断に問題が発生しましたが、リソースはリセットしました")
