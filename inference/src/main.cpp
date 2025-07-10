#include <M5Stack.h>
#include "core/NoiseDetector.h"
#include "core/WaveformDrawer.h"
#include "core/FourierTransform.h"
#include "core/AnnotationData.h"
#include "core/NeuralNetwork.h"
#include "config/secret.h"

// 現在のモードを記憶する変数
enum Mode { NONE, NOISE_DETECTOR, FOUNRIER_TRANSFORM, WAVEFORM_DRAWER, ANNOTATION, NEURAL_NETWORK };
Mode currentMode = NONE;

// メニュー選択用の変数
int selectedItem = 0;
const int NUM_MENU_ITEMS = 5;
const char* menuItems[] = {"Noise Detect", "FFT", "Wave Drawer", "Annotation", "Neural Network"};
const int MENU_START_Y = 30;  // メニューの開始Y座標
const int MENU_ITEM_HEIGHT = 30;  // メニュー項目の高さ

void showMenu() {
    M5.Lcd.clear();
    M5.Lcd.setTextSize(2);
    
    // タイトルを表示
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.printf("Select Mode");

    // メニュー項目の表示（固定位置）
    for(int i = 0; i < NUM_MENU_ITEMS; i++) {
        M5.Lcd.setCursor(20, MENU_START_Y + (i * MENU_ITEM_HEIGHT));
        if(i == selectedItem) {
            M5.Lcd.setTextColor(TFT_BLUE);
        } else {
            M5.Lcd.setTextColor(TFT_WHITE);
        }
        M5.Lcd.printf("%s", menuItems[i]);
    }
    
    // 操作説明の表示（固定位置）
    M5.Lcd.setTextColor(TFT_WHITE);
    M5.Lcd.setTextSize(2);
    M5.Lcd.drawString("Down", 35, 220);
    M5.Lcd.drawString("Select", 135, 220);
    M5.Lcd.drawString("Up", 250, 220);
}

void setup() {
    M5.begin();
    showMenu();
}

void loop() {
    if (currentMode == NONE) {
        M5.update();
        if (M5.BtnA.wasPressed()) {  // 左ボタン - 下移動
            selectedItem = (selectedItem + 1) % NUM_MENU_ITEMS;
            showMenu();
        } else if (M5.BtnC.wasPressed()) {  // 右ボタン - 上移動
            selectedItem = (selectedItem - 1 + NUM_MENU_ITEMS) % NUM_MENU_ITEMS;
            showMenu();
        } else if (M5.BtnB.wasPressed()) {  // 中央ボタン - 選択
            M5.Lcd.clear();
            switch(selectedItem) {
                case 0: // Noise Detector
                    M5.Lcd.setCursor(0, 0);
                    M5.Lcd.setTextSize(1);
                    M5.Lcd.printf("Starting Noise Detector...\n");
                    currentMode = NOISE_DETECTOR;
                    noiseDetector.initNoiseDetector();
                    noiseDetector.getADCAverage();
                    noiseDetector.startTimer();
                    break;
                case 1: // FFT
                    currentMode = FOUNRIER_TRANSFORM;
                    fourierTransform.initFourierTransform();
                    fourierTransform.startTimer();
                    M5.Lcd.setTextSize(2);
                    M5.Lcd.drawString("START FFT", 115, 220);
                    M5.Lcd.setTextSize(1);
                    break;
                case 2: // Waveform Drawer
                    M5.Lcd.clear();
                    M5.Lcd.setCursor(0, 0);
                    M5.Lcd.setTextSize(1);
                    M5.Lcd.printf("Starting Waveform Drawer...\n");
                    currentMode = WAVEFORM_DRAWER;
                    waveformDrawer.initWaveformDrawer();
                    M5.Lcd.clear();
                    waveformDrawer.getADCAverage();
                    waveformDrawer.startTimer();
                    break;
                case 3: // Annotation
                    M5.Lcd.clear();
                    M5.Lcd.setCursor(0, 0);
                    M5.Lcd.setTextSize(1);
                    M5.Lcd.printf("Starting Annotation...\n");
                    currentMode = ANNOTATION;
                    annotationData.initAnnotationData();
                    annotationData.getADCAverage();
                    annotationData.startTimer();
                    break;
                case 4: // NN
                    M5.Lcd.clear();
                    M5.Lcd.setCursor(0, 0);
                    M5.Lcd.setTextSize(1);
                    M5.Lcd.printf("Neural Network...\n");
                    currentMode = NEURAL_NETWORK;
                    neuralNetwork.initNeuralNetworkData();
                    neuralNetwork.getADCAverage();
                    neuralNetwork.startTimer();
                    break;
            }
        }
    } else {
        if (currentMode == NOISE_DETECTOR) {
            noiseDetector.storeNoise();
        } else if (currentMode == FOUNRIER_TRANSFORM) {
            M5.update();
            if (M5.BtnB.wasPressed()) {
                fourierTransform.startFFT();
            }
        } else if (currentMode == WAVEFORM_DRAWER) {
        } else if (currentMode == ANNOTATION) {
            annotationData.storeNoise();
        } else if (currentMode == NEURAL_NETWORK) {
            neuralNetwork.storeNoise();
        }
    }
}
