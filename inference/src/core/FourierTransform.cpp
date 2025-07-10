#include "FourierTransform.h"
#include <M5Stack.h>
#include "Free_Fonts.h" 
#include "arduinoFFT.h"

FourierTransform fourierTransform;

volatile int16_t micFFT = 0;
hw_timer_t *timer_fft = NULL;

FourierTransform::FourierTransform() {
    adc_buf = nullptr;
    vReal = nullptr;
    vImag = nullptr;
}

FourierTransform::~FourierTransform() {
    freeBuffer();
}

void FourierTransform::freeBuffer() {
    if (adc_buf) {
        delete[] adc_buf;
        adc_buf = nullptr;

    }
    if (vReal) {
        delete[] vReal;
        vReal = nullptr;
    }
    if (vImag) {
        delete[] vImag;
        vImag = nullptr;
    }
}

void FourierTransform::initFourierTransform() {
    freeBuffer();
    adc_buf = new int16_t[FFT_SAMPLES];
    if (adc_buf == nullptr) {
        M5.Lcd.println("Failed to allocate buffer");
        return;
    }
    vReal = new double[FFT_SAMPLES];
    if (vReal == nullptr) {
        M5.Lcd.println("Failed to allocate buffer");
        return;
    }
    vImag = new double[FFT_SAMPLES];
    if (vImag == nullptr) {
        M5.Lcd.println("Failed to allocate buffer");
        return;
    }
    for (int i = 0; i < FFT_SAMPLES; i++) {
        adc_buf[i] = 0;
    }
    for (int i = 0; i < FFT_SAMPLES; i++) {
        vReal[i] = 0.0;
        vImag[i] = 0.0;
    }
}

void IRAM_ATTR onTimerFFT() {
    micFFT = analogRead(36);
    fourierTransform.getMic();
}

void FourierTransform::startTimer() {
    M5.Lcd.fillScreen(BLACK);
    timer_fft = timerBegin(1, 80, true);
    timerAttachInterrupt(timer_fft, &onTimerFFT, true);
    timerAlarmWrite(timer_fft, TIME_FFT, true);
    timerAlarmEnable(timer_fft);
}

void FourierTransform::restartTimer() {
    memset(adc_buf, 0, sizeof(adc_buf));
    timerRestart(timer_fft);
    timerStart(timer_fft);
}

void FourierTransform::DCRemoval(double *vData, unsigned int samples) {
    double mean = 0;
    for (uint16_t i = 0; i < samples; i++) {
        mean += vData[i];
    }
    mean /= samples;
    for (uint16_t i = 0; i < samples; i++) {
        vData[i] -= mean;
    }
}

void FourierTransform::drawChart(int nsamples) {
    // グラフ原点や描画可能領域の定義
    int X0 = 10;
    int Y0 = 20;
    int _height = 220 - Y0;   // 実際に使える縦サイズ
    int _width = 320;         // 実際に使える横サイズ
    float dmax = 5.0;         // グラフ化する際の最大値（振幅）

    //==============================================
    // 1. スキップ係数を計算: サンプル数が描画幅を超える場合は間引く
    //   例: nsamples = 500, _width = 320 のとき
    //       -> skipFactor = ceil(500 / 320) = ceil(1.5625) = 2
    //       -> 2 サンプルに 1 回描画する
    //==============================================
    int skipFactor = 1;
    if (nsamples > _width) {
        skipFactor = ceil((float)nsamples / _width);
    }

    // 実際に描画を行う回数(横方向に何本描画するか)
    int drawCount = nsamples / skipFactor;
    if (drawCount < 1) {
        // 安全策として 1 本だけでも描画させる
        drawCount = 1;
    }

    //==============================================
    // 2. 横幅あたりの描画バンド幅を計算し、0 にならないようにする
    //==============================================
    int band_width = floor((float)_width / drawCount);
    if (band_width < 1) {
        band_width = 1;  // 幅が 0 にならないように最低 1 を確保
    }

    // fillRect に与えるパッド幅(余白)を計算
    int band_pad = band_width - 1;
    if (band_pad < 1) {
        band_pad = 1;  // 負数にならないように最低 1 を確保
    }

    //==============================================
    // 3. グラフを描画するループ (スキップ分を加味)
    //==============================================
    for (int i = 0; i < nsamples; i += skipFactor) {
        // band (描画バンドのインデックス) は i / skipFactor で計算
        int band = i / skipFactor;
        int hpos = band * band_width + X0;  // 横方向の描画位置

        // 実際のサンプル値
        float d = vReal[i];
        if (d > dmax) d = dmax;  // オーバー分はクリップ
        if (d < 0.0)  d = 0.0;   // 負値が来る場合はとりあえず 0 にクリップ

        // 高さを計算 (d/dmax) を描画可能領域にスケーリング
        int h = (int)((d / dmax) * _height);

        // fillRect で縦バー描画
        //  (注) y 座標は上から下へ増える前提なので、塗りつぶしは
        //       下から上へ行うように座標を調整
        M5.Lcd.fillRect(hpos, (_height - h), band_pad, h, WHITE);

        //==============================================
        // 4. ラベル(周波数表示など)を描画 (任意)
        //    こちらもサンプル数が大きいと文字が重なるため
        //    ある程度の区切りで表示するようにする
        //==============================================
        if (drawCount >= 4) {
            // drawCount(実際に描画する本数)の1/4おきにラベルを描画
            int label_interval = drawCount / 4;
            if (label_interval <= 0) {
                label_interval = 1;
            }
            if ( (band % label_interval) == 0 ) {
                M5.Lcd.setCursor(hpos, _height+5);
                M5.Lcd.printf("%.1fkHz",
                    ((i * 1.0 * SAMPLING_FREQUENCY) / FFT_SAMPLES / 1000)
                );
            }
        }
    }
}

void FourierTransform::doFFT() {
    // adc_bufをvRealに代入（ADCデータのコピー）
    for (int i = 0; i < FFT_SAMPLES; i++) {
        vReal[i] = static_cast<double>(adc_buf[i]);
    }
    DCRemoval(vReal, FFT_SAMPLES);
    // FFT処理
    ArduinoFFT<double> FFT = ArduinoFFT<double>();
    FFT.windowing(vReal, FFT_SAMPLES, FFT_WIN_TYP_HAMMING, FFT_FORWARD);  // ハミング窓
    FFT.compute(vReal, vImag, FFT_SAMPLES, FFT_FORWARD);  // FFT計算
    FFT.complexToMagnitude(vReal, vImag, FFT_SAMPLES);  // 実数信号の絶対値計算
    for (int i = 0; i < FFT_SAMPLES; i++) {
        vReal[i] = vReal[i] / (FFT_SAMPLES / 2);  // 最大値をサンプル数で割る
    }
    drawChart(FFT_SAMPLES / 2);
    // FFTでピークの周波数を表示
    double peak = FFT.majorPeak(vReal, FFT_SAMPLES, SAMPLING_FREQUENCY);
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.printf("Peak: %.1fHz\n", peak);
}

void FourierTransform::startFFT() {
    timerStop(timer_fft);
    M5.Lcd.fillScreen(BLACK);
    doFFT();
    M5.Lcd.drawString("UPDATE", 140, 230);
    freeBuffer();
    initFourierTransform();
    restartTimer();
}

void FourierTransform::getMic() {
    static int pt = FFT_SAMPLES - 1;
    adc_buf[pt] = micFFT;

    if (--pt < 0) {
        pt = FFT_SAMPLES - 1;
    }
}
