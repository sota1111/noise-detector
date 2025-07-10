#include <WiFi.h>
#include <HTTPClient.h>
#include "common.h"
#include "AnnotationData.h"

AnnotationData annotationData;
hw_timer_t *timer_annotation = NULL;

// 初期化
AnnotationData::AnnotationData() {
    annotationData.micValue = 0;
    val_buf = nullptr;
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
    }
}

AnnotationData::~AnnotationData() {
    freeBuffer();
}

void AnnotationData::freeBuffer() {
    if (val_buf) {
        delete[] val_buf;
        val_buf = nullptr;
    }
}

// 1秒間のAD変換値の平均を取得する関数
void AnnotationData::getADCAverage() {
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

void AnnotationData::initBuf() {
    for (int i = 0; i < RECORD_MAX_LEN; i++) {
        val_buf[i] = 0;
    }
    for (int i = 0; i < MAX_NOISE_EVENTS; i++) {
        noiseEventTimes_A[i] = 0;
    }
    write_index = 0;
    detect_index = 0;
    dBValue_A = 0;
    isDataStored = false;
    integralValue = 0;
    sampleIntegralCount = 0;
}

void AnnotationData::initAnnotationData() {
    freeBuffer();  // 以前のバッファがあれば解放
    val_buf = new int16_t[RECORD_MAX_LEN];  // バッファを動的に確保
    if (val_buf == nullptr) {
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
int AnnotationData::calculateDbValue(int avgIntegral) {
    int dBValue = 0;
    if (avgIntegral < 100) {
        dBValue = 40 + (2 * avgIntegral) / 10;
    } else {
        dBValue = 60 + (avgIntegral - 100) / 20;
    }
    return dBValue;
}

// 複数回検出
bool AnnotationData::detectNoise_A(int avgIntegral) {
    static int noiseEventIndex_A = 0;
    int dBValue = calculateDbValue(avgIntegral);
    unsigned long currentTime = millis();

    if (dBValue >= INSTANT_NOISE_THRESHOLD_DB) {
        for (int i = 0; i < MAX_NOISE_EVENTS; ++i) {
            long difTime = currentTime - noiseEventTimes_A[i];
            if ((noiseEventTimes_A[i] != 0) && (difTime < TIME_IGNORE_NOISE)){
                return false;
            }   
        }
        noiseEventTimes_A[noiseEventIndex_A] = currentTime;
        noiseEventIndex_A = (noiseEventIndex_A + 1) % MAX_NOISE_EVENTS;

        unsigned long observationTimeMillis = OBSERVATION_DURATION_SECOND * 1000;
        int eventCount = 1;

        for (int i = 0; i < MAX_NOISE_EVENTS; ++i) {
            if (noiseEventTimes_A[i] != 0) {
                long difTime = currentTime - noiseEventTimes_A[i];
                if (difTime <= observationTimeMillis) {
                    eventCount++;
                }
            }
        }
        if (eventCount >= NOISE_EVENT_COUNT_THRESHOLD) {
            dBValue_A = dBValue;
            return true;
        }
    }

    return false;
}

// ============================================================
//  移動積分を計算する関数
// ============================================================
int AnnotationData::calculateMovingIntegral(int currentMicValue, int writeIndex)
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

// ============================================================
//  updateBuffer から移動積分の計算処理を関数呼び出しに置き換え
// ============================================================
void AnnotationData::updateBuffer(int micValue) {
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

        if ( ( isDetectNoise_A) && (isNoiseDetected == false) ) {
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
            isDataStored = true;
            timerStop(timer_annotation);
            int dBValue = dBValue_A;
            
            M5.Lcd.printf("dB: %d\n", dBValue);
            isRequestSpeaker = true;
            int stopTime = millis() - startTime;
            M5.Lcd.printf("Time: %d\n", stopTime);

            // 関数スコープの変数初期化
            detect_count = 0;
            isNoiseDetected      = false;
            startTime    = 0;
        }
    }
}

void AnnotationData::logNoiseTimestamp() {
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
    } else {
        // 現在時刻が取得できなかった場合のエラーメッセージ
        M5.Lcd.setTextColor(TFT_RED, TFT_BLACK);
        M5.Lcd.println("Current time has not been obtained.");
        delay(1000);
    }
}

void AnnotationData::storeNoise() {
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

void IRAM_ATTR onTimerAnnototion() {
    annotationData.micValue = analogRead(36);
    annotationData.updateBuffer(annotationData.micValue);
}

void AnnotationData::startTimer() {
    M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.println("ANNOTATION MODE");
    timer_annotation = timerBegin(0, 80, true);
    timerAttachInterrupt(timer_annotation, &onTimerAnnototion, true);
    timerAlarmWrite(timer_annotation, TIME_IRQ, true);
    timerAlarmEnable(timer_annotation);
}

void AnnotationData::restartTimer() {
    M5.Lcd.println("Restart Timer");
    delay(1000);
    M5.Lcd.fillScreen(TFT_BLACK);
    delay(100);
    M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.println("NOISE DETECTING");
    initBuf();
    timerRestart(timer_annotation);
    timerStart(timer_annotation);
    delay(100);
}

