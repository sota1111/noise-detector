#include <WiFi.h>
#include <HTTPClient.h>
#include "common.h"
#include "NeuralNetwork.h"
#include "config/config.h"
#include "Validation/Validation_inference.h"
#include "Validation/Validation_parameters.h"

// ESC-50の34クラス名定義
const char* ESC50_CLASS_NAMES[34] = {
    "dog", "cat", "rain", "sea_waves", "crackling_fire", "crickets", 
    "wind", "water_drops", "thunderstorm", "crying_baby", "sneezing", 
    "clapping", "breathing", "coughing", "footsteps", "laughing", 
    "brushing_teeth", "snoring", "door_wood_knock", "mouse_click", 
    "keyboard_typing", "can_opening", "washing_machine", "vacuum_cleaner", 
    "toilet_flush", "clock_alarm", "clock_tick", "glass_breaking", 
    "door_wood_creaks", "siren", "car_horn", "engine", "train", "silent"
};

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
    detect_index = 0;
    adcAverageDetect = 2048;
    integralValue = 0;
    sampleIntegralCount = 0;
    maxDbIndex = 0;
    maxDbValue = 0;
    isDataStored = false;
    triggerMode = NN_DEFAULT_TRIGGER_MODE;
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
    write_index = 0;
    detect_index = 0;
    integralValue = 0;
    sampleIntegralCount = 0;
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
    static unsigned int detect_count = 0;
    static bool isNoiseDetected      = false;
    static unsigned int startTime    = 0;

    if (!isDataStored) {
        // ノイズ検出前後のデータを記録        
        write_index = (write_index + 1) % RECORD_MAX_LEN;
        val_buf[write_index] = micValue;

        // 移動積分の計算
        int avgIntegral = calculateMovingIntegral(micValue, write_index);
        int dBValue = calculateDbValue(avgIntegral);

        // トリガーモードに応じて閾値判定
        bool isTriggered = false;
        if (triggerMode == NN_TRIGGER_MODE_DB) {
            // デシベル閾値モード
            if (dBValue >= NN_TRIGGER_THRESHOLD_DB && !isNoiseDetected) {
                isTriggered = true;
                Serial.printf("High volume detected: %ddB\n", dBValue);
            }
        } else if (triggerMode == NN_TRIGGER_MODE_VOLTAGE) {
            // 電圧閾値モード
            if (avgIntegral >= NN_TRIGGER_THRESHOLD_VOLTAGE && !isNoiseDetected) {
                isTriggered = true;
                Serial.printf("High voltage detected: %d\n", avgIntegral);
            }
        }
        
        if (isTriggered) {
            isNoiseDetected = true;
            detect_index = write_index;
            maxDbValue = dBValue;
            maxDbIndex = write_index;
            startTime = millis();
        }
        
        // ノイズ検出中は音声データの最大dB値を更新
        if (isNoiseDetected) {
            detect_count++;
            if (dBValue > maxDbValue) {
                maxDbValue = dBValue;
                maxDbIndex = write_index;
            }
        }
        
        // ノイズ検出後、十分なデータを貯め続ける
        if (detect_count > RECORD_AFTER_LEN) {
            isDataStored = true;
            timerStop(timer_nn);
            int stopTime = millis() - startTime;
            Serial.printf("Data collection complete: time=%dms, maxDb=%d\n", stopTime, maxDbValue);

            // 関数スコープの変数初期化（次の検出サイクルのため）
            detect_count = 0;
            isNoiseDetected = false;
            startTime = 0;
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
    
    // 音を検出したポイント（detect_index）の前後のデータを使用
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

    // 上位5クラスを見つける
    int top_indices[5] = {0, 1, 2, 3, 4};
    float top_probs[5] = {probs[0], probs[1], probs[2], probs[3], probs[4]};
    
    // 全クラスの中から最大確率を見つける
    for (int classNo = 0; classNo < NNABLART_VALIDATION_OUTPUT0_SIZE; classNo++) {
      if (probs[classNo] > top_probability) {
        top_probability = probs[classNo];
        top_class = classNo;
      }
      
      // 上位5位に入るかチェック
      for (int i = 0; i < 5; i++) {
        if (probs[classNo] > top_probs[i]) {
          // より低い順位のものを後ろにシフト
          for (int j = 4; j > i; j--) {
            top_probs[j] = top_probs[j-1];
            top_indices[j] = top_indices[j-1];
          }
          top_probs[i] = probs[classNo];
          top_indices[i] = classNo;
          break;
        }
      }
    }
    
    // 詳細情報表示エリアをクリアして上位5クラスのみ表示
    M5.Lcd.fillRect(0, 40, 320, 90, TFT_BLACK);
    M5.Lcd.setCursor(0, 40);
    M5.Lcd.setTextSize(1);
    M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    
    for (int i = 0; i < 5; i++) {
      M5.Lcd.printf("%2d %-15s %.3f\n", top_indices[i], ESC50_CLASS_NAMES[top_indices[i]], top_probs[i]);
    }

    // 推論結果の表示
    M5.Lcd.printf("result: %d, time: %lld ms\n", top_class, elapsed_time);
    return top_class;
}


void NeuralNetwork::processAudio() {
    // NN_TRIGGER_THRESHOLD_DBdB以上の音を検出してデータが溜まったらニューラルネットワークを実行
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
            
            // トリガーモード表示
            M5.Lcd.setTextSize(1);
            if (triggerMode == NN_TRIGGER_MODE_DB) {
                M5.Lcd.printf("Trigger: dB >= %d\n", NN_TRIGGER_THRESHOLD_DB);
            } else {
                M5.Lcd.printf("Trigger: Volt >= %d\n", NN_TRIGGER_THRESHOLD_VOLTAGE);
            }
            
            // クラス名を表示（ESC-50の34クラス対応）
            M5.Lcd.setTextSize(2);
            M5.Lcd.setCursor(0, 160);
            
            if (soundClass >= 0 && soundClass < 34) {
                M5.Lcd.println(ESC50_CLASS_NAMES[soundClass]);
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
        
        // クラス名もシリアル送信（ESC-50の34クラス対応）
        if (soundClass >= 0 && soundClass < 34) {
            Serial.print("CLASS_NAME:");
            Serial.println(ESC50_CLASS_NAMES[soundClass]);
        } else {
            Serial.println("CLASS_NAME:UNKNOWN");
        }
        
        // 待機
        delay(1000);
        
        // 表示を検出中に戻す
        M5.Lcd.fillScreen(TFT_BLACK);
        M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
        M5.Lcd.setCursor(0, 0);
        M5.Lcd.setTextSize(2);
        M5.Lcd.println("Neural Network MODE");
        
        // トリガーモード表示
        M5.Lcd.setCursor(0, 30);
        M5.Lcd.setTextSize(1);
        if (triggerMode == NN_TRIGGER_MODE_DB) {
            M5.Lcd.printf("Trigger: dB >= %d", NN_TRIGGER_THRESHOLD_DB);
        } else {
            M5.Lcd.printf("Trigger: Voltage >= %d", NN_TRIGGER_THRESHOLD_VOLTAGE);
        }
        
        M5.Lcd.setCursor(0, 50);
        M5.Lcd.setTextSize(2);
        M5.Lcd.println("DETECTING...");
        
        // NN処理完了後、次の検出サイクルを開始
        isDataStored = false;
        maxDbValue = 0;
        maxDbIndex = 0;
        
        // バッファを初期化して次の検出サイクルの準備
        initBuf();
        
        // タイマーを再開
        Serial.println("Restarting timer for next detection cycle");
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
    
    // トリガーモード表示
    M5.Lcd.setCursor(0, 10);
    M5.Lcd.setTextSize(1);
    if (triggerMode == NN_TRIGGER_MODE_DB) {
        M5.Lcd.printf("Trigger: dB >= %d", NN_TRIGGER_THRESHOLD_DB);
    } else {
        M5.Lcd.printf("Trigger: Voltage >= %d", NN_TRIGGER_THRESHOLD_VOLTAGE);
    }
    
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
