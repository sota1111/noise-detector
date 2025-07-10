#include "SpeakerHandler.h"
SpeakerHandler speakerHandler;

void SpeakerHandler::playTone(int frequency, int duration) {
    M5.Speaker.tone(frequency, duration);
    delay(100);
    M5.Speaker.end();
    delay(100);
    M5.Speaker.tone(frequency, duration);
    delay(100);
    M5.Speaker.end();
}