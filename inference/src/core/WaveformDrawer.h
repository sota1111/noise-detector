#ifndef WAVEFORMDRAWER_H
#define WAVEFORMDRAWER_H

#include <M5Stack.h>

#define TIME_WAVE (30*1000) //us
#define TIME_INTEGRAL (300*1000) //us
#define INTEGRAL_SAMPLES (TIME_INTEGRAL / TIME_WAVE)
#define GRAPH_MAX_LEN 320
#define X_OFFSET 0
#define Y_OFFSET 100
#define X_SCALE 1
#define MIC_Unit 36

class WaveformDrawer {
private:
    int16_t* adc_buf=nullptr;
    int16_t* val_buf=nullptr;
    int16_t* integ_buf=nullptr;
    void drawStringWithFormat(const char* label, int value, int x, int y);
    void freeBuffer();

public:
    WaveformDrawer();
    ~WaveformDrawer();
    void getADCAverage();
    void startTimer();
    void restartTimer();
    void drawWaveform();
    void initWaveformDrawer();
};

extern WaveformDrawer waveformDrawer;

#endif