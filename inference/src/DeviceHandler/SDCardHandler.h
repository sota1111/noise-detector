#ifndef SDCARDHANDLER_H
#define SDCARDHANDLER_H

#include <M5Stack.h>
#include <SD.h>
#include <time.h>

//#define APARTMENT_NAME "APARTMENT_A"
//#define ROOM_NAME "ROOM_A"
#define APARTMENT_NAME "ANNOTATION"
#define ROOM_NAME "VOICE"
class SDCardHandler {
public:
    void initSDCard(const char* apartmentName, const char* roomName);
    bool writeSDCard(const char* filename, const char* data);
};

#endif