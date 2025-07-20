# AnalogDiscovery3 DAC API Reference

## 概要
AnalogDiscovery3のDAC（Analog Out）を使用したカスタム波形出力のためのAPI仕様

## 重要なAPI関数

### 1. デバイス制御

#### FDwfDeviceOpen
```c
FDwfDeviceOpen(int idxDevice, HDWF *phdwf)
```
- デバイスを開いてハンドルを取得
- `idxDevice`: 列挙されたデバイスのインデックス（-1で自動選択）
- `phdwf`: デバイスハンドルを受け取るポインタ

#### FDwfDeviceAutoConfigureSet
```c
FDwfDeviceAutoConfigureSet(HDWF hdwf, int fAutoConfigure)
```
- 自動設定の有効/無効を制御
- `fAutoConfigure`: 0=無効, 1=有効, 3=動的設定
- 高速化のため0（無効）に設定することを推奨

### 2. AnalogOut（DAC）制御

#### ノードの有効化
```c
FDwfAnalogOutNodeEnableSet(HDWF hdwf, int idxChannel, AnalogOutNode node, int fEnable)
```
- チャンネルのノードを有効化
- `idxChannel`: チャンネル番号（0 or 1）
- `node`: `DwfAnalogOutNode.Carrier` (キャリア信号)
- `fEnable`: 1=有効, 0=無効

#### 関数タイプの設定
```c
FDwfAnalogOutNodeFunctionSet(HDWF hdwf, int idxChannel, AnalogOutNode node, DwfAnalogOutFunction func)
```
- 出力関数タイプを設定
- `func`: `DwfAnalogOutFunction.Custom` (カスタム波形用)

#### 周波数設定
```c
FDwfAnalogOutNodeFrequencySet(HDWF hdwf, int idxChannel, AnalogOutNode node, double hzFrequency)
```
- 出力周波数を設定（Hz）
- カスタム波形の場合、サンプルレートを指定

#### 振幅設定
```c
FDwfAnalogOutNodeAmplitudeSet(HDWF hdwf, int idxChannel, AnalogOutNode node, double vAmplitude)
```
- 出力振幅を設定（V）
- **3.3V最大出力の場合**: `vAmplitude = 1.65` (±1.65V)

#### オフセット設定
```c
FDwfAnalogOutNodeOffsetSet(HDWF hdwf, int idxChannel, AnalogOutNode node, double vOffset)
```
- 出力オフセットを設定（V）
- **3.3V最大出力の場合**: `vOffset = 1.65` (1.65V中心)

#### カスタムデータ設定
```c
FDwfAnalogOutNodeDataSet(HDWF hdwf, int idxChannel, AnalogOutNode node, double *rgdData, int cdData)
```
- カスタム波形データを設定
- `rgdData`: **±1に正規化されたdouble配列**
- `cdData`: サンプル数
- **重要**: 出力値 = `Offset + Sample × Amplitude`
  - Sample = 0 → Offset
  - Sample = +1 → Offset + Amplitude  
  - Sample = -1 → Offset - Amplitude

#### 出力開始
```c
FDwfAnalogOutConfigure(HDWF hdwf, int idxChannel, int fStart)
```
- 出力を開始/停止
- `fStart`: 1=開始, 0=停止

## ⚠️ 重要：AD3の実際の最大出力は2.57V

**注意**: AD3の実際の最大出力電圧は2.57Vです（3.3Vではありません）

### 2.57V最大出力の正しい設定

```python
# AD3実測値：2.57V最大出力設定
amplitude = 1.285  # ±1.285V (2.57V / 2)
offset = 1.285     # 1.285V中心 (2.57V / 2)

# 音声データを±1に正規化
normalized_audio = audio_data / max(abs(audio_data))

# 設定
device.analogOut.nodeAmplitudeSet(0, DwfAnalogOutNode.Carrier, amplitude)
device.analogOut.nodeOffsetSet(0, DwfAnalogOutNode.Carrier, offset)
device.analogOut.nodeDataSet(0, DwfAnalogOutNode.Carrier, normalized_audio)
```

### 出力電圧計算
- 最大出力 (+1): offset + amplitude = 1.285 + 1.285 = 2.57V
- 最小出力 (-1): offset - amplitude = 1.285 - 1.285 = 0V  
- 中央値 (0): offset = 1.285V

## 音声ファイル出力のワークフロー

1. **音声ファイル読み込み**
   ```python
   import librosa
   audio, sr = librosa.load('audio.wav', sr=None)
   ```

2. **データ正規化（±1範囲）**
   ```python
   normalized_audio = audio / max(abs(audio))
   ```

3. **DAC設定**
   - Amplitude: 1.65V
   - Offset: 1.65V  
   - Function: Custom
   - Frequency: サンプルレート

4. **データ設定と出力開始**
   ```python
   device.analogOut.nodeDataSet(0, DwfAnalogOutNode.Carrier, normalized_audio, len(normalized_audio))
   device.analogOut.configure(0, True)
   ```

## デバイス仕様

### Analog Discovery 3 (AD3)
- チャンネル数: 2チャンネル
- 最大出力電圧: ±5V (設定可能)
- **3.3V制限の場合**: Amplitude=1.65V, Offset=1.65V
- サンプルレート: 最大100MHz（デバイス依存）

## エラーハンドリング

```python
def check_dwf_error():
    error_code = dwf.FDwfGetLastError()
    if error_code != 0:
        error_msg = dwf.FDwfGetLastErrorMsg()
        print(f"DWF Error {error_code}: {error_msg}")
        return False
    return True
```

## 注意事項

1. **データ正規化**: 音声データは必ず±1範囲に正規化
2. **電圧設定**: 3.3V最大の場合、Amplitude=1.65V, Offset=1.65V
3. **サンプルレート**: 音声ファイルのサンプルレートに合わせて設定
4. **AutoConfigure**: 高速化のため無効化を推奨
5. **リソース解放**: 使用後は必ずデバイスを閉じる

## 参考
- WaveForms SDK Reference Manual v3.21.3
- Section 6: Analog Out (Arbitrary Waveform Generator)