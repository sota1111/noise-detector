#ifndef CONFIG_H
#define CONFIG_H

// 騒音検出
#define MAX_NOISE_EVENTS  10  // 騒音イベント記録の最大数

// 基準値A: X_A dB以上の瞬間的な騒音がY_A分間にZ_A回以上観測された場合通知
#define INSTANT_NOISE_THRESHOLD_DB_A (70) //X_A
#define OBSERVATION_DURATION_SECOND_A (20*1000) //Y_A(ms)
#define TIME_IGNORE_NOISE_A (1000) //ms
#define NOISE_EVENT_COUNT_THRESHOLD_A (5) //Z_A MAX_NOISE_EVENTSを超えない値にする

// 基準値A: X_B dB以上の瞬間的な騒音がY_B分間にZ_B回以上観測された場合通知
#define INSTANT_NOISE_THRESHOLD_DB_B (90) //X_B
#define OBSERVATION_DURATION_SECOND_B (2000) //Y_B(ms)
#define TIME_IGNORE_NOISE_B (1000) //ms
#define NOISE_EVENT_COUNT_THRESHOLD_B (1) //Z_B MAX_NOISE_EVENTSを超えない値にする

//#define RECORD_1s_LEN 1000*1000/TIME_IRQ
// 10000を超えるデータはDRAMが足りない。PSRAMを使う必要がある。
#define RECORD_BEFORE_LEN (5000)
#define RECORD_AFTER_LEN (5000)
#define RECORD_MAX_LEN (RECORD_BEFORE_LEN + RECORD_AFTER_LEN)

#define TIME_IRQ (100) //us
// デシベル計算用の積分時間　
#define TIME_INTEGRAL_DETECT (200*1000) //us
// RECORD_MAX_LENを超えない値にする
#define INTEGRAL_SAMPLES_DETECT (TIME_INTEGRAL_DETECT / TIME_IRQ)

#define FFT_SAMPLES_APROP (2048) // 2のべき乗 これ以上大きいとリセットする
#define SAMPLING_FREQUENCY_APROP (1/(TIME_IRQ*1e-6)) 

// デバイス管理
#define TIMEOUT_WIFI_CONNECT (30000)
#define JST (3600L * 9)

// 表示関連
#define FONT_SIZE 2

// NN関連
#define NN_TRIGGER_THRESHOLD_DB (60)  // NN実行をトリガーするdB値
#define NN_TRIGGER_THRESHOLD_VOLTAGE (200)  // NN実行をトリガーする電圧値
#define NN_TRIGGER_MODE_DB (0)  // デシベル閾値モード
#define NN_TRIGGER_MODE_VOLTAGE (1)  // 電圧閾値モード
#define NN_DEFAULT_TRIGGER_MODE NN_TRIGGER_MODE_DB  // デフォルトのトリガーモード
#define NN_BEFORE_LEN (400)

#endif