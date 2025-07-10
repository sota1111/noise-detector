#include <WiFi.h>
#include <HTTPClient.h>
#include "common.h"
#include "NeuralNetwork.h"
#include "Validation/Validation_inference.h"
#include "Validation/Validation_parameters.h"

NeuralNetwork neuralNetwork;
hw_timer_t *timer_nn = NULL;
void *_context = NULL;
float *nn_input_buffer;

// 初期化
NeuralNetwork::NeuralNetwork() {
    neuralNetwork.micValue = 0;
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
    _context = nnablart_validation_allocate_context(Validation_parameters);
    nn_input_buffer = nnablart_validation_input_buffer(_context, 0);
}

NeuralNetwork::~NeuralNetwork() {
    freeBuffer();
}

void NeuralNetwork::freeBuffer() {
    if (val_buf) {
        delete[] val_buf;
        val_buf = nullptr;
    }
}

void NeuralNetwork::softmax(const float* logits, float* probs, int size) {
    float max_logit = logits[0];
    for (int i = 1; i < size; ++i) {
      if (logits[i] > max_logit) {
        max_logit = logits[i];
      }
    }
  
    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i) {
      probs[i] = expf(logits[i] - max_logit);  // 数値安定化のためにmaxを引く
      sum_exp += probs[i];
    }
  
    for (int i = 0; i < size; ++i) {
      probs[i] /= sum_exp;
    }
  }

// 1秒間のAD変換値の平均を取得する関数
void NeuralNetwork::getADCAverage() {
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

void NeuralNetwork::initBuf() {
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

void NeuralNetwork::initNeuralNetworkData() {
    freeBuffer();  // 以前のバッファがあれば解放
    val_buf = new int16_t[RECORD_MAX_LEN];  // バッファを動的に確保
    if (val_buf == nullptr) {
        M5.Lcd.println("Failed to allocate buffer");
        return;
    }
    initBuf();
#if 0
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
#endif
    M5.Lcd.fillScreen(TFT_BLACK);
}

// デシベル変換
int NeuralNetwork::calculateDbValue(int avgIntegral) {
    int dBValue = 0;
    if (avgIntegral < 100) {
        dBValue = 40 + (2 * avgIntegral) / 10;
    } else {
        dBValue = 60 + (avgIntegral - 100) / 20;
    }
    return dBValue;
}

// 複数回検出
bool NeuralNetwork::detectNoise_A(int avgIntegral) {
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
int NeuralNetwork::calculateMovingIntegral(int currentMicValue, int writeIndex)
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
void NeuralNetwork::updateBuffer(int micValue) {
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
            //M5.Lcd.println("NOISE DETECTED");
            startTime = millis();
        }
        if(isNoiseDetected){
            detect_count++;
        }
        // ノイズ検出後、データを貯め続ける。
        if(detect_count > RECORD_AFTER_LEN){
            isDataStored = true;
            timerStop(timer_nn);
            int dBValue = dBValue_A;
            
            //M5.Lcd.printf("dB: %d\n", dBValue);
            isRequestSpeaker = true;
            int stopTime = millis() - startTime;
            //M5.Lcd.printf("Time: %d\n", stopTime);

            // 関数スコープの変数初期化
            detect_count = 0;
            isNoiseDetected      = false;
            startTime    = 0;
        }
    }
}

void NeuralNetwork::logNoiseTimestamp() {
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

int NeuralNetwork::exeNeuralNetwork() {
    if (!_context) {
        M5.Lcd.println("NN context allocation failed");
        return -1;
    }
    int startIndex = detect_index - NN_BEFORE_LEN;
    int endIndex = startIndex + NNABLART_VALIDATION_INPUT0_SIZE;

    int bufferIndex = 0;
    for (int index = startIndex; index < endIndex; index++) {
        nn_input_buffer[bufferIndex] = ((float)(val_buf[index % RECORD_MAX_LEN] - 2048)) / 2048.0;
        bufferIndex++;
    }

    // 推論の実行（ミリ秒単位で計測）
    int64_t start_time = millis();
    nnablart_validation_inference(_context);
    int64_t elapsed_time = millis() - start_time;

    // 推論結果の取得
    float* logits = nnablart_validation_output_buffer(_context, 0);
    float probs[NNABLART_VALIDATION_OUTPUT0_SIZE];

    softmax(logits, probs, NNABLART_VALIDATION_OUTPUT0_SIZE);
    int top_class = 0;
    float top_probability = 0.0f;
    for (int classNo = 0; classNo < NNABLART_VALIDATION_OUTPUT0_SIZE; classNo++) {
      M5.Lcd.printf("class %d: %f\n", classNo, probs[classNo]);
      if (probs[classNo] > top_probability) {
        top_probability = probs[classNo];
        top_class = classNo;
      }
    }

    // 推論結果の表示
    M5.Lcd.printf("result: %d, ", top_class);
    M5.Lcd.printf("time:  %lld ms\n\n", elapsed_time);
    return top_class;
}


void NeuralNetwork::storeNoise() {
    if (isRequestSpeaker) {
        speakerHandler.playTone(440, 100);
        isRequestSpeaker = false;
    }

    if (isDataStored) {
        //notificationAWS();
        //logNoiseTimestamp();
        int soundClass = exeNeuralNetwork();

        M5.Lcd.setTextSize(FONT_SIZE_LARGE);
        M5.Lcd.setCursor(0, 60);

        if (soundClass == 0) {
            M5.Lcd.println("VOICE");
        } else if (soundClass == 1) {
            M5.Lcd.println("CLAP");
        }
        M5.Lcd.setTextSize(1);
        delay(1000);
        isDataStored = false;
        //M5.Lcd.println("NOISE STORED");
        restartTimer();
    }
}

void IRAM_ATTR onTimerNN() {
    neuralNetwork.micValue = analogRead(36);
    neuralNetwork.updateBuffer(neuralNetwork.micValue);
}

void NeuralNetwork::startTimer() {
    M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.println("Neural Network MODE");
    timer_nn = timerBegin(0, 80, true);
    timerAttachInterrupt(timer_nn, &onTimerNN, true);
    timerAlarmWrite(timer_nn, TIME_IRQ, true);
    timerAlarmEnable(timer_nn);
}

void NeuralNetwork::restartTimer() {
    M5.Lcd.println("Restart Timer");
    delay(1000);
    M5.Lcd.fillScreen(TFT_BLACK);
    delay(100);
    M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.println("NOISE DETECTING");
    initBuf();
    timerRestart(timer_nn);
    timerStart(timer_nn);
    delay(100);
}
