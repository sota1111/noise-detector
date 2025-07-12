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
    isTimerStopped = false;
    write_index = 0;
    adcAverageDetect = 0;
    lastNNExecution = 0;
    nnInterval = 1000; // 1秒間隔
    integralValue = 0;
    sampleIntegralCount = 0;
    dBValues = nullptr;
    dBValueIndex = 0;
    maxDbIndex = 0;
    maxDbValue = 0;
    isDataStored = false;
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
    if (dBValues) {
        delete[] dBValues;
        dBValues = nullptr;
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
    write_index = 0;
    lastNNExecution = 0;
    integralValue = 0;
    sampleIntegralCount = 0;
    dBValueIndex = 0;
    maxDbIndex = 0;
    maxDbValue = 0;
    isDataStored = false;
}

void NeuralNetwork::initNeuralNetworkData() {
    freeBuffer();  // 以前のバッファがあれば解放
    val_buf = new int16_t[RECORD_MAX_LEN];  // バッファを動的に確保
    if (val_buf == nullptr) {
        M5.Lcd.println("Failed to allocate buffer");
        return;
    }
    
    // 1秒間のサンプル数でdBValuesバッファを確保
    int samplesPerSecond = 1000000 / TIME_IRQ; // 1秒間のサンプル数 (10000サンプル)
    dBValues = new int[samplesPerSecond];
    if (dBValues == nullptr) {
        Serial.println("Failed to allocate dB buffer");
        return;
    }
    Serial.println("dB buffer allocated successfully");
    
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
    static int debugCount = 0;
    
    if (!isDataStored) {
        // 連続的にバッファに音声データを保存
        write_index = (write_index + 1) % RECORD_MAX_LEN;
        val_buf[write_index] = micValue;

        // 最新の音声データでデシベル値を計算
        int avgIntegral = calculateMovingIntegral(micValue, write_index);
        int dBValue = calculateDbValue(avgIntegral);

        // デシベル値を保存
        int samplesPerSecond = 1000000 / TIME_IRQ;
        if (dBValueIndex < samplesPerSecond) {
            dBValues[dBValueIndex] = dBValue;
            
            // 最大dB値とそのインデックスを更新
            if (dBValue > maxDbValue) {
                maxDbValue = dBValue;
                maxDbIndex = write_index;
            }
            
            dBValueIndex++;
            
            // 1秒分のデータが溜まったらタイマーを止める
            if (dBValueIndex >= samplesPerSecond) {
                isDataStored = true;
                timerStop(timer_nn);
                Serial.printf("1sec data collected: samples=%d, maxDb=%d\n", dBValueIndex, maxDbValue);
            }
        }
    }
}

void NeuralNetwork::logNoiseTimestamp() {
    struct tm timeInfo;
    String csvData = "";
    // write_index+RECORD_AFTER_LENがcsvDataの先頭になるようにする
    int startIndex = write_index + RECORD_AFTER_LEN + 1;
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
    
    // 最大dB値の前NN_BEFORE_LENのデータを使用
    int startIndex = maxDbIndex - NN_BEFORE_LEN;
    int endIndex = startIndex + NNABLART_VALIDATION_INPUT0_SIZE;

    int bufferIndex = 0;
    for (int index = startIndex; index < endIndex; index++) {
        int bufIndex = (index + RECORD_MAX_LEN) % RECORD_MAX_LEN;
        nn_input_buffer[bufferIndex] = ((float)(val_buf[bufIndex] - 2048)) / 2048.0;
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

    // 詳細情報表示エリアをクリアして表示
    M5.Lcd.fillRect(0, 40, 320, 90, TFT_BLACK);
    M5.Lcd.setCursor(0, 40);
    M5.Lcd.setTextSize(1);
    M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    
    for (int classNo = 0; classNo < NNABLART_VALIDATION_OUTPUT0_SIZE; classNo++) {
      M5.Lcd.printf("class %d: %.3f\n", classNo, probs[classNo]);
      if (probs[classNo] > top_probability) {
        top_probability = probs[classNo];
        top_class = classNo;
      }
    }

    // 推論結果の表示
    M5.Lcd.printf("result: %d, time: %lld ms\n", top_class, elapsed_time);
    return top_class;
}


void NeuralNetwork::processAudio() {
    // 1秒分のデータが溜まったらニューラルネットワークを実行
    if(isDataStored){
        int soundClass = exeNeuralNetwork();
        
        // 結果が変わった場合のみ画面を更新
        static int lastSoundClass = -1;
        static int lastDbValue = -1;
        if (soundClass != lastSoundClass || abs(maxDbValue - lastDbValue) > 2) {
            // 結果表示エリアのみクリア
            M5.Lcd.fillRect(0, 130, 320, 110, TFT_BLACK);
            M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
            M5.Lcd.setTextSize(2);
            M5.Lcd.setCursor(0, 130);
            
            // デシベル値を表示
            M5.Lcd.printf("dB: %d\n", maxDbValue);
            
            // クラス名を表示
            M5.Lcd.setTextSize(FONT_SIZE_LARGE);
            M5.Lcd.setCursor(0, 160);
            if (soundClass == 0) {
                M5.Lcd.println("VOICE");
            } else if (soundClass == 1) {
                M5.Lcd.println("CLAP");
            } else {
                M5.Lcd.println("UNKNOWN");
            }
            M5.Lcd.setTextSize(1);
            lastSoundClass = soundClass;
            lastDbValue = maxDbValue;
        }
        
        // シリアル通信で検出クラスを送信
        Serial.print("DETECTED_CLASS:");
        Serial.println(soundClass);
        Serial.print("DB_VALUE:");
        Serial.println(maxDbValue);
        
        // クラス名もシリアル送信
        if (soundClass == 0) {
            Serial.println("CLASS_NAME:VOICE");
        } else if (soundClass == 1) {
            Serial.println("CLASS_NAME:CLAP");
        } else {
            Serial.println("CLASS_NAME:UNKNOWN");
        }
        
        // NN処理完了後、次の1秒間のデータ収集を開始
        isDataStored = false;
        dBValueIndex = 0;
        maxDbValue = 0;
        maxDbIndex = 0;
        
        // タイマーを再開
        Serial.println("Restarting timer for next cycle");
        timerRestart(timer_nn);
        timerStart(timer_nn);
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
    Serial.println("Starting timer...");
    
    timer_nn = timerBegin(0, 80, true);
    if (timer_nn == NULL) {
        M5.Lcd.println("Timer begin failed");
        Serial.println("Timer begin failed");
        return;
    }
    Serial.println("Timer created");
    
    timerAttachInterrupt(timer_nn, &onTimerNN, true);
    Serial.println("Interrupt attached");
    
    timerAlarmWrite(timer_nn, TIME_IRQ, true);
    Serial.println("Alarm written");
    
    timerAlarmEnable(timer_nn);
    Serial.println("Timer started");
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
