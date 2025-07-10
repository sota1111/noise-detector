#ifndef FOURIERTRANSFORM_H
#define FOURIERTRANSFORM_H

#include <M5Stack.h>

#define TIME_FFT (200) //us
#define FFT_SAMPLES (4096) // 2のべき乗
#define SAMPLING_FREQUENCY (1/(TIME_FFT*1e-6)) 
#define X_OFFSET 0
#define Y_OFFSET 100
#define MIC_Unit 36

class FourierTransform {
private:
    int16_t* adc_buf=nullptr;
    double* vReal=nullptr;
    double* vImag=nullptr;
    void freeBuffer();
    void doFFT();
    void DCRemoval(double *vData, unsigned int samples);
    void drawChart(int samples);

public:
    FourierTransform();
    ~FourierTransform();
    void startTimer();
    void restartTimer();
    void getMic();
    void initFourierTransform();
    void startFFT();
};

extern FourierTransform fourierTransform;

#endif