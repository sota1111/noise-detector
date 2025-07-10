#ifndef SPEAKERHANDLER_H
#define SPEAKERHANDLER_H

#include <M5Stack.h>

class SpeakerHandler {
public:
    void playTone(int frequency, int duration);
};

#endif