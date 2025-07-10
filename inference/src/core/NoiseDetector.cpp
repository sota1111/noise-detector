#include <WiFi.h>
#include <HTTPClient.h>
#include "arduinoFFT.h"
#include "common.h"
#include "NoiseDetector.h"

NoiseDetector noiseDetector;
hw_timer_t *timer = NULL;

// A特性補正用の表
struct AWeightTable {
    double frequency;
    double response;
};

const AWeightTable aWeightTable[] = {
    {16, -55},
    {32, -40},
    {63, -25},
    {125, -16},
    {250, -8},
    {500, -4},
    {1000, 0},
    {2000, 2},
    {4000, 0},
    {8000, -2},
    {16000, -6},
};

// 初期化
NoiseDetector::NoiseDetector() {
    noiseDetector.micValue = 0;
    val_buf = nullptr;
    vReal = nullptr;
    vImag = nullptr;
    isRequestSpeaker = false;
    isDataStored = false;
    isTimerStopped = false;
    write_index = 0;
    detect_index = 0;
    integralValue = 0;
    sampleIntegralCount = 0;
    adcAverageDetect = 0;
    for (int i = 0; i < MAX_NOISE_EVENTS; i++) {
        noiseEventTimes_A[i] = 0;
        noiseEventTimes_B[i] = 0;
    }
}

NoiseDetector::~NoiseDetector() {
    freeBuffer();
}

void NoiseDetector::freeBuffer() {
    if (val_buf) {
        delete[] val_buf;
        val_buf = nullptr;
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

// 1秒間のAD変換値の平均を取得する関数
void NoiseDetector::getADCAverage() {
    const unsigned long duration = 1000; // 1秒間 (ミリ秒単位)
    unsigned long startTime = millis();
    long sum = 0;
    int count = 0;

    while (millis() - startTime < duration) {
        int adcValue = analogRead(36);
        sum += adcValue;
        count++;
    }
    adcAverageDetect = (int16_t)(sum / count);
}

void NoiseDetector::initBuf() {
    for (int i = 0; i < RECORD_MAX_LEN; i++) {
        val_buf[i] = 0;
    }
    for (int i = 0; i < FFT_SAMPLES_APROP; i++) {
        vReal[i] = 0.0;
        vImag[i] = 0.0;
    }
    for (int i = 0; i < MAX_NOISE_EVENTS; i++) {
        noiseEventTimes_A[i] = 0;
        noiseEventTimes_B[i] = 0;
    }
    integralValue = 0;
    sampleIntegralCount = 0;
}

void NoiseDetector::initNoiseDetector() {
    freeBuffer();  // 以前のバッファがあれば解放
    val_buf = new int16_t[RECORD_MAX_LEN];  // バッファを動的に確保
    if (val_buf == nullptr) {
        M5.Lcd.println("Failed to allocate buffer");
        return;
    }
    vReal = new double[FFT_SAMPLES_APROP];
    if (vReal == nullptr) {
        M5.Lcd.println("Failed to allocate buffer");
        return;
    }
    vImag = new double[FFT_SAMPLES_APROP];
    if (vImag == nullptr) {
        M5.Lcd.println("Failed to allocate buffer");
        return;
    }
    initBuf();
    sdcardHandler.initSDCard(APARTMENT_NAME, ROOM_NAME);
    wifiHandler.connectWiFi(WIFI_SSID, WIFI_PASSWORD);
    wifiHandler.synchronizeTime();

    // hello worldをGet
    // Lambdaからデータ取得
    HTTPClient http;
    String url = String(BASE_URL) + "hello/";
    http.begin(url);
    int httpCode = http.GET();

    if (httpCode > 0) {
        String payload = http.getString();
        M5.Lcd.println("Response:");
        M5.Lcd.println(payload);
    } else {
        M5.Lcd.println("HTTP GET failed, error: " + String(httpCode));
    }
    http.end();
    delay(1000);
    M5.Lcd.fillScreen(TFT_BLACK);
}

// デシベル変換
int NoiseDetector::calculateDbValue(int avgIntegral) {
    if (avgIntegral < 100) {
        return 40 + (2 * avgIntegral) / 10;
    } else {
        return 60 + (avgIntegral - 100) / 20;
    }
}

// 複数回検出
bool NoiseDetector::detectNoise_A(int avgIntegral) {
    static int noiseEventIndex_A = 0;
    int dBValue = calculateDbValue(avgIntegral);
    unsigned long currentTime = millis();

    if (dBValue >= INSTANT_NOISE_THRESHOLD_DB_A) {
        for (int i = 0; i < MAX_NOISE_EVENTS; ++i) {
            long difTime = currentTime - noiseEventTimes_A[i];
            if ((noiseEventTimes_A[i] != 0) && (difTime < TIME_IGNORE_NOISE_A)){
                return false;
            }   
        }
        noiseEventTimes_A[noiseEventIndex_A] = currentTime;
        noiseEventIndex_A = (noiseEventIndex_A + 1) % MAX_NOISE_EVENTS;

        unsigned long observationTimeMillis = OBSERVATION_DURATION_SECOND_A * 1000;
        int eventCount = 1;

        for (int i = 0; i < MAX_NOISE_EVENTS; ++i) {
            if (noiseEventTimes_A[i] != 0) {
                long difTime = currentTime - noiseEventTimes_A[i];
                if (difTime <= observationTimeMillis) {
                    eventCount++;
                }
            }
        }
        if (eventCount >= NOISE_EVENT_COUNT_THRESHOLD_A) {
            dBValue_A = dBValue;
            return true;
        }
    }

    return false;
}

bool NoiseDetector::detectNoise_B(int avgIntegral) {
    static int noiseEventIndex_B = 0;
    int dBValue = calculateDbValue(avgIntegral);
    unsigned long currentTime = millis();

    if (dBValue >= INSTANT_NOISE_THRESHOLD_DB_B) {
        for (int i = 0; i < MAX_NOISE_EVENTS; ++i) {
            long difTime = currentTime - noiseEventTimes_B[i];
            if ((noiseEventTimes_B[i] != 0) && (difTime < TIME_IGNORE_NOISE_B)){
                return false;
            }   
        }
        noiseEventTimes_B[noiseEventIndex_B] = currentTime;
        noiseEventIndex_B = (noiseEventIndex_B + 1) % MAX_NOISE_EVENTS;

        unsigned long observationTimeMillis = OBSERVATION_DURATION_SECOND_B * 1000;
        int eventCount = 1;

        for (int i = 0; i < MAX_NOISE_EVENTS; ++i) {
            if (noiseEventTimes_B[i] != 0) {
                long difTime = currentTime - noiseEventTimes_B[i];
                if (difTime <= observationTimeMillis) {
                    eventCount++;
                }
            }
        }
        if (eventCount >= NOISE_EVENT_COUNT_THRESHOLD_B) {
            dBValue_B = dBValue;
            return true;
        }
    }

    return false;
}

// ============================================================
//  移動積分を計算する関数
// ============================================================
int NoiseDetector::calculateMovingIntegral(int currentMicValue, int writeIndex)
{
    int16_t adcVal = abs(currentMicValue - adcAverageDetect);
    integralValue += adcVal;

    if (sampleIntegralCount < INTEGRAL_SAMPLES_DETECT) {
        sampleIntegralCount++;
    } else {
        int prevIndex       = writeIndex - INTEGRAL_SAMPLES_DETECT;
        unsigned int oldPos = (prevIndex + RECORD_MAX_LEN) % RECORD_MAX_LEN;
        integralValue    -= abs(val_buf[oldPos] - adcAverageDetect);
    }

    int avgIntegral = integralValue / INTEGRAL_SAMPLES_DETECT;
    return avgIntegral;
}

void NoiseDetector::DCRemoval(double *vData, unsigned int samples) {
    double mean = 0;
    for (uint16_t i = 0; i < samples; i++) {
        mean += vData[i];
    }
    mean /= samples;
    for (uint16_t i = 0; i < samples; i++) {
        vData[i] -= mean;
    }
}

double NoiseDetector::doFFT(int detect_count) {
    for (int i = 0; i < FFT_SAMPLES_APROP; i++) {   
        int j = i + detect_count - FFT_SAMPLES_APROP;
        if (j < 0) {
            vReal[i] = static_cast<double>(val_buf[(j + RECORD_MAX_LEN ) % RECORD_MAX_LEN]);
        } else {
            vReal[i] = static_cast<double>(val_buf[j % RECORD_MAX_LEN]);
        }
    }
    DCRemoval(vReal, FFT_SAMPLES_APROP);
    // FFT処理
    ArduinoFFT<double> FFT = ArduinoFFT<double>();
    FFT.windowing(vReal, FFT_SAMPLES_APROP, FFT_WIN_TYP_HAMMING, FFT_FORWARD);  // ハミング窓
    FFT.compute(vReal, vImag, FFT_SAMPLES_APROP, FFT_FORWARD);  // FFT計算
    FFT.complexToMagnitude(vReal, vImag, FFT_SAMPLES_APROP);  // 実数信号の絶対値計算
    // ピーク周波数を計算
    double maxValue = 0.0;
    int peakIndex = 0;
    for (int i = 1; i < FFT_SAMPLES_APROP / 2; i++) { // Nyquist周波数まで
        if (vReal[i] > maxValue) {
            maxValue = vReal[i];
            peakIndex = i;
        }
    }
    double peak = peakIndex * SAMPLING_FREQUENCY_APROP / FFT_SAMPLES_APROP;
    return peak;
}

// 線形補間を用いて相対レスポンスを計算する関数
double interpolateAWeight(double freq) {
    int size = sizeof(aWeightTable) / sizeof(AWeightTable);

    // 範囲外チェック
    if (freq <= aWeightTable[0].frequency) {
        return aWeightTable[0].response;
    }
    if (freq >= aWeightTable[size - 1].frequency) {
        return aWeightTable[size - 1].response;
    }

    // 表を参照して線形補間
    for (int i = 0; i < size - 1; i++) {
        if (freq >= aWeightTable[i].frequency && freq <= aWeightTable[i + 1].frequency) {
            double f1 = aWeightTable[i].frequency;
            double f2 = aWeightTable[i + 1].frequency;
            double r1 = aWeightTable[i].response;
            double r2 = aWeightTable[i + 1].response;
            return r1 + (r2 - r1) * (freq - f1) / (f2 - f1);
        }
    }
    return 0; // ここには到達しない
}

int calc_Aprop(double peak_freq, int dB_base) {
    double aResponse = interpolateAWeight(peak_freq);
    return dB_base + aResponse;
}

// ============================================================
//  updateBuffer から移動積分の計算処理を関数呼び出しに置き換え
// ============================================================
void NoiseDetector::updateBuffer(int micValue) {
    static unsigned int detect_count = 0;
    static bool isNoiseDetected      = false;
    static unsigned int startTime    = 0;
    static char noiseSource;

    if (!isDataStored) {
        // ノイズ検出前後のデータを記録        
        write_index = (write_index + 1) % RECORD_MAX_LEN;
        val_buf[write_index] = micValue;

        // 移動積分の計算
        int avgIntegral = calculateMovingIntegral(micValue, write_index);

        // ノイズ検出
        bool isDetectNoise_A = (detectNoise_A(avgIntegral));
        bool isDetectNoise_B = (detectNoise_B(avgIntegral));

        if ( ( isDetectNoise_A || isDetectNoise_B) && (isNoiseDetected == false) ) {
            if (isDetectNoise_A) {
                noiseSource = 'A';
            } else if (isDetectNoise_B) {
                noiseSource = 'B';
            }
            isNoiseDetected = true;
            detect_index = write_index;
            M5.Lcd.println("NOISE DETECTED");
            startTime = millis();
        }
        if(isNoiseDetected){
            detect_count++;
        }
        // ノイズ検出後、データを貯め続ける。
        if(detect_count > RECORD_AFTER_LEN){
            timerStop(timer);
            int dBValue = 0;
            if (noiseSource == 'A') {
                dBValue = dBValue_A;
            } else if (noiseSource == 'B') {
                dBValue = dBValue_B;
            }
            
            // A特性補正を行う
            M5.Lcd.printf("dB: %d\n", dBValue);
            double peak_freq = doFFT(detect_count);
            M5.Lcd.printf("Peak: %.1fHz\n", peak_freq);
            int dB_Aprop = calc_Aprop(peak_freq, dBValue);
            M5.Lcd.printf("dB_Aprop: %d\n", dB_Aprop);
            if (noiseSource == 'A') {
                if (dB_Aprop >= INSTANT_NOISE_THRESHOLD_DB_A) {
                    isRequestSpeaker = true;
                }
            } else if (noiseSource == 'B') {
                if (dB_Aprop >= INSTANT_NOISE_THRESHOLD_DB_B) {
                    isRequestSpeaker = true;
                }
            }
            int stopTime = millis() - startTime;
            M5.Lcd.printf("Time: %d\n", stopTime);

            write_index = 0;
            detect_index = 0;
            dBValue_A = 0;
            dBValue_B = 0;
            detect_count = 0;
            avgIntegral = 0;
            isDataStored = true;
            isNoiseDetected = false;
        }
    }
}

void NoiseDetector::logNoiseTimestamp() {
    struct tm timeInfo;
    String csvData = "";
    // detect_index+RECORD_AFTER_LENがcsvDataの先頭になるようにする
    int startIndex = detect_index + RECORD_AFTER_LEN + 1;
    int endIndex = startIndex + RECORD_MAX_LEN;

    for (int i = startIndex; i < endIndex; i++) {
        csvData += String(val_buf[i % RECORD_MAX_LEN]) + ",";
    }

    if (getLocalTime(&timeInfo)) {
        char timestamp[64];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", &timeInfo);

        // CSVファイル名を生成
        char fileName[128];
        String dirName = "/" + String(APARTMENT_NAME) + "/" + String(ROOM_NAME);
        snprintf(fileName, sizeof(fileName), "%s/log_%s.csv", dirName.c_str(), timestamp);
        File csvFile = SD.open(fileName, FILE_APPEND);
        csvFile.print(csvData.c_str());
        csvFile.close();
        M5.Lcd.println("STORE DATA");

        // POST処理を追加
        postCSVtoServer(fileName);
    } else {
        // 現在時刻が取得できなかった場合のエラーメッセージ
        M5.Lcd.setTextColor(TFT_RED, TFT_BLACK);
        M5.Lcd.println("Current time has not been obtained.");
        delay(1000);
    }
}

void NoiseDetector::postCSVtoServer(const char* fileName) {
    HTTPClient http;
    String url = String(BASE_URL) + "detect-sound";
    File csvFile = SD.open(fileName, FILE_READ);

    if (!csvFile) {
        M5.Lcd.setTextColor(TFT_RED, TFT_BLACK);
        M5.Lcd.println("Failed to open CSV file.");
        return;
    }

    size_t fileSize = csvFile.size();
    uint8_t* buffer = (uint8_t*)malloc(fileSize);
    if (buffer == NULL) {
        M5.Lcd.setTextColor(TFT_RED, TFT_BLACK);
        M5.Lcd.println("Failed to allocate memory.");
        csvFile.close();
        return;
    }

    csvFile.read(buffer, fileSize);
    csvFile.close();

    String sanitizedFileName = fileName;
    if (sanitizedFileName.startsWith("/")) {
        sanitizedFileName = sanitizedFileName.substring(1);
    }

    http.begin(url);
    http.addHeader("Content-Type", "text/csv");
    http.addHeader("X-File-Name", sanitizedFileName);

    int httpResponseCode = http.POST(buffer, fileSize);
    free(buffer);

    if (httpResponseCode > 0) {
        String response = http.getString();
        M5.Lcd.println("Response: " + response);
    } else {
        M5.Lcd.setTextColor(TFT_RED, TFT_BLACK);
        M5.Lcd.printf("POST failed, error: %d\n", httpResponseCode);
    }

    http.end();
}

void NoiseDetector::notificationAWS() {
    HTTPClient http;
    String url = String(BASE_URL) + "notification";
    http.begin(url);
    // POSTリクエストを送信
    int httpCode = http.POST("notification");
    if (httpCode > 0) {
        String response = http.getString();
        M5.Lcd.println("Notification:" + response);
    } else {
        M5.Lcd.println("HTTP POST failed, error: " + String(httpCode));
    }
}

void NoiseDetector::storeNoise() {
    if (isRequestSpeaker) {
        speakerHandler.playTone(440, 100);
        isRequestSpeaker = false;
    }

    if (isDataStored) {
        //notificationAWS();
        logNoiseTimestamp();
        isDataStored = false;
        M5.Lcd.println("NOISE STORED");
        restartTimer();
    }
}

void IRAM_ATTR onTimer() {
    noiseDetector.micValue = analogRead(36);
    noiseDetector.updateBuffer(noiseDetector.micValue);
}

void NoiseDetector::startTimer() {
    M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.println("NOISE DETECTING");
    timer = timerBegin(0, 80, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, TIME_IRQ, true);
    timerAlarmEnable(timer);
}

void NoiseDetector::restartTimer() {
    M5.Lcd.println("Restart Timer");
    delay(1000);
    M5.Lcd.fillScreen(TFT_BLACK);
    delay(100);
    M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.println("NOISE DETECTING");
    initBuf();
    timerRestart(timer);
    timerStart(timer);
    delay(100);
}

