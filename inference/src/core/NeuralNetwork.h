#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <M5Stack.h>
#include <SD.h>
#include <time.h>
#include "config/secret.h"

// 騒音検出
#define MAX_NOISE_EVENTS  10  // 騒音イベント記録の最大数

// 基準値A: X_A dB以上の瞬間的な騒音がY_A分間にZ_A回以上観測された場合通知
#define INSTANT_NOISE_THRESHOLD_DB (70) //X_A
#define OBSERVATION_DURATION_SECOND (20*1000) //Y_A(ms)
#define TIME_IGNORE_NOISE (1000) //ms
#define NOISE_EVENT_COUNT_THRESHOLD (1) //Z_A MAX_NOISE_EVENTSを超えない値にする

//#define RECORD_1s_LEN 1000*1000/TIME_IRQ
// 10000を超えるデータはDRAMが足りない。PSRAMを使う必要がある。
#define RECORD_BEFORE_LEN (5000)
#define RECORD_AFTER_LEN (5000)
#define RECORD_MAX_LEN (RECORD_BEFORE_LEN + RECORD_AFTER_LEN)

#define TIME_IRQ (100) //us
// デシベル計算用の積分時間　
#define TIME_INTEGRAL_DETECT (200*1000) //us
// RECORD_MAX_LENを超えない値にする
#define INTEGRAL_SAMPLES_DETECT (TIME_INTEGRAL_DETECT / TIME_IRQ)
#define FFT_SAMPLES_APROP (2048) // 2のべき乗 これ以上大きいとリセットする
#define SAMPLING_FREQUENCY_APROP (1/(TIME_IRQ*1e-6)) 

// 表示関連
#define FONT_SIZE 2
#define FONT_SIZE_LARGE 8

// NN関連
#define NN_BEFORE_LEN 200

class NeuralNetwork {
private:
    bool isRequestSpeaker;
    bool isDataStored;
    bool isTimerStopped;
    int dBValue_A;
    unsigned int write_index;
    unsigned int detect_index;
    int16_t* val_buf=nullptr;

    // 騒音検出
    unsigned long noiseEventTimes_A[MAX_NOISE_EVENTS];
    long integralValue;
    int  sampleIntegralCount;
    int16_t adcAverageDetect;

    void initBuf();
    bool detectNoise_A(int avgIntegral);
    int calculateDbValue(int avgIntegral);
    int calculateMovingIntegral(int currentMicValue, int writeIndex);
    void logNoiseTimestamp();
    void notificationAWS();
    void freeBuffer();
    void softmax(const float* logits, float* probs, int size);
    int exeNeuralNetwork();

public:
    NeuralNetwork();
    ~NeuralNetwork();
    int micValue;
    void initNeuralNetworkData();
    void getADCAverage();
    void updateBuffer(int micValue);
    void startTimer();
    void restartTimer();
    bool judgeRestartTimer();
    void storeNoise();
};

extern NeuralNetwork neuralNetwork;

#endif