#include "SDCardHandler.h"
SDCardHandler sdcardHandler;

void SDCardHandler::initSDCard(const char* apartmentName, const char* roomName) {
    if (!SD.begin(TFCARD_CS_PIN, SPI, 40000000)) {
        M5.Lcd.println("SD Card Mount Failed");
        delay(1000);
        return;
    }
    M5.Lcd.println("SD Card Mount Success");
    delay(1000);

    String firstDir = "/" + String(apartmentName);
    if (!SD.exists(firstDir.c_str())) {
        SD.mkdir(firstDir.c_str());
        M5.Lcd.println("make directory " + firstDir);
    }

    String fullDir = "/" + String(apartmentName) + "/" + String(roomName);
    if (!SD.exists(fullDir.c_str())) {
        SD.mkdir(fullDir.c_str());
        M5.Lcd.println("make directory " + fullDir);
    }
}

// SDカード書き込み
bool SDCardHandler::writeSDCard(const char* filename, const char* data) {
    File file = SD.open(filename, FILE_APPEND);
    if (!file) {
        M5.Lcd.println("Failed to open file for writing");
        delay(1000);
        return false;  // 書き込み失敗
    }
    file.println(data);
    file.close();
    return true;  // 書き込み成功
}