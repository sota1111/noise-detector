#ifndef NOISEDETECTOR_H
#define NOISEDETECTOR_H

#include <M5Stack.h>
#include <SD.h>
#include <time.h>
#include "../config/config.h"
#include "../config/secret.h"

class NoiseDetector {
private:
    bool isRequestSpeaker;
    bool isDataStored;
    bool isTimerStopped;
    int dBValue_A;
    int dBValue_B;
    unsigned int write_index;
    unsigned int detect_index;
    int16_t* val_buf=nullptr;
    double* vReal=nullptr;
    double* vImag=nullptr;
    // 騒音検出
    unsigned long noiseEventTimes_A[MAX_NOISE_EVENTS];
    unsigned long noiseEventTimes_B[MAX_NOISE_EVENTS];
    long integralValue;
    int  sampleIntegralCount;
    int16_t adcAverageDetect;

    void initBuf();
    bool detectNoise_A(int avgIntegral);
    bool detectNoise_B(int avgIntegral);
    int calculateDbValue(int avgIntegral);
    int calculateMovingIntegral(int currentMicValue, int writeIndex);
    void logNoiseTimestamp();
    void notificationAWS();
    void postCSVtoServer(const char* fileName);
    void freeBuffer();
    void DCRemoval(double *vData, unsigned int samples);
    double doFFT(int detect_count);

public:
    NoiseDetector();
    ~NoiseDetector();
    int micValue;
    void initNoiseDetector();
    void getADCAverage();
    void updateBuffer(int micValue);
    void startTimer();
    void restartTimer();
    bool judgeRestartTimer();
    void storeNoise();
};

extern NoiseDetector noiseDetector;

#endif