#include "WiFiHandler.h"
WiFiHandler wifiHandler;

void WiFiHandler::connectWiFi(const char* ssid, const char* password) {
    M5.Lcd.printf("Connecting to %s\n", ssid);
    WiFi.begin(ssid, password);

    unsigned long startAttemptTime = millis();
    const unsigned long timeout = TIMEOUT_WIFI_CONNECT;

    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - startAttemptTime >= timeout) {
            M5.Lcd.println("\nWiFi Connection Timeout!");
            return;
        }
        delay(500);
        M5.Lcd.print('.');
    }
    M5.Lcd.println("\nWiFi Connected!");
}

void WiFiHandler::synchronizeTime() {
    configTime(JST, 0, "ntp.nict.jp", "time.google.com", "ntp.jst.mfeed.ad.jp");
}