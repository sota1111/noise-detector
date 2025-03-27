#include <M5Unified.h>
#include <SD.h>
#include "Validation/Validation_inference.h"
#include "Validation/Validation_parameters.h"

void *_context = NULL;
float *nn_input_buffer;

void softmax(const float* logits, float* probs, int size) {
  float max_logit = logits[0];
  for (int i = 1; i < size; ++i) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }

  float sum_exp = 0.0f;
  for (int i = 0; i < size; ++i) {
    probs[i] = expf(logits[i] - max_logit);  // 数値安定化のためにmaxを引く
    sum_exp += probs[i];
  }

  for (int i = 0; i < size; ++i) {
    probs[i] /= sum_exp;
  }
}

void setup() {
  // M5Stack の初期化
  M5.begin();

  if (!SD.begin(4)) {
    M5.Lcd.println("SD Card Mount Failed");
    return;
  }
  _context = nnablart_validation_allocate_context(Validation_parameters);
  nn_input_buffer = nnablart_validation_input_buffer(_context, 0);
  //M5.Lcd.println("SD Card Mounted");
}

void loop() {
  // SDカードのルートディレクトリをオープン
  File root = SD.open("/");
  if (!root) {
    M5.Lcd.println("Failed to open root directory");
    delay(2000);
    return;
  }

  bool found = false;
  // ループでルートディレクトリ内のファイルを列挙
  while (true) {
    File file = root.openNextFile();
    if (!file) {
      break;  // ファイルがなくなったらループ終了
    }

    // ディレクトリでない場合に処理
    if (!file.isDirectory()) {
      String filename = file.name();
      // 拡張子が ".csv" の場合に処理する
      if (filename.endsWith(".csv")) {
        found = true;
        M5.Lcd.printf("file: %s\n", filename.c_str());

        // CSVファイルの読み込み（1行だけを読み込む）
        String csvLine = file.readStringUntil('\n');
        file.close();

        if (!_context) {
          M5.Lcd.println("NN context allocation failed");
          continue;
        }

        // CSV行を解析し、各値を入力バッファに格納
        int index = 0;
        // strtok を使うために、String を char 配列に変換
        char csvBuffer[csvLine.length() + 1];
        csvLine.toCharArray(csvBuffer, csvLine.length() + 1);
        char *token = strtok(csvBuffer, ",");

        int csvSize = NNABLART_VALIDATION_INPUT0_SIZE;
        for(index = 0; index < csvSize; index++){
          nn_input_buffer[index] = ((float)atof(token)-(float)2048.0)/(float)2048.0;
          token = strtok(NULL, ",");
          index++;
          if(token == NULL){
            break;
          }
        }
        // M5.Lcd.printf("csvReadSize:%d\n", index);

        // 推論の実行（ミリ秒単位で計測）
        int64_t start_time = millis();
        nnablart_validation_inference(_context);
        int64_t elapsed_time = millis() - start_time;

        // 推論結果の取得
        float* logits = nnablart_validation_output_buffer(_context, 0);
        float probs[NNABLART_VALIDATION_OUTPUT0_SIZE];

        softmax(logits, probs, NNABLART_VALIDATION_OUTPUT0_SIZE);
        int top_class = 0;
        float top_probability = 0.0f;
        for (int classNo = 0; classNo < NNABLART_VALIDATION_OUTPUT0_SIZE; classNo++) {
          M5.Lcd.printf("class %d: %f\n", classNo, probs[classNo]);
          if (probs[classNo] > top_probability) {
            top_probability = probs[classNo];
            top_class = classNo;
          }
        }

        // 推論結果の表示
        M5.Lcd.printf("result: %d, ", top_class);
        M5.Lcd.printf("time:  %lld ms\n\n", elapsed_time);

        // NNコンテキストの解放
        // int free_ret = nnablart_validation_free_context(_context);
      } else {
        file.close();  // .csv でないファイルは閉じる
      }
    } else {
      file.close();  // ディレクトリの場合は閉じる
    }
  }
  root.close();

  if (!found) {
    M5.Lcd.println("No .csv files found in root directory");
  }

  // すべてのファイル処理後、ループを停止する場合は以下の無限ループで対応
  while (1) {
    delay(1000);
  }
}
