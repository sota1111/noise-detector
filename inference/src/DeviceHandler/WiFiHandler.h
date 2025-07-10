#ifndef WIFIHANDLER_H
#define WIFIHANDLER_H

#include <WiFi.h>
#include <M5Stack.h>

#define BASE_URL "https://6ansren87i.execute-api.ap-northeast-1.amazonaws.com/Prod/"
#define TIMEOUT_WIFI_CONNECT (30000)
#define JST (3600L * 9)

class WiFiHandler {
public:
    void connectWiFi(const char* ssid, const char* password);
    void synchronizeTime();
};

#endif