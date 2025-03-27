#include <M5Unified.h>
#include <SD.h>
#include "Validation/Validation_inference.h"
#include "Validation/Validation_parameters.h"

void setup() {
  // M5Stack の初期化
  M5.begin();

  if (!SD.begin(4)) {
    M5.Lcd.println("SD Card Mount Failed");
    return;
  }
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
      // 拡張子が ".raw" の場合に推論を実行
      if (filename.endsWith(".raw")) {
        found = true;
        M5.Lcd.printf("file: %s, ", filename.c_str());

        // 画像サイズ（例：28x28）を定義
        const int imageSize = 28 * 28;
        uint8_t image_data[imageSize];

        // ファイルから画像データを読み込む
        size_t bytesRead = file.read(image_data, imageSize);
        file.close();  // 各ファイルは処理後に閉じる

        if (bytesRead != imageSize) {
          M5.Lcd.printf("RAW file read error: %s, ", filename.c_str());
          continue;
        }

        //M5.Lcd.println("RAW image loaded into memory");

        // 推論処理の準備
        void *nn_context = nnablart_validation_allocate_context(Validation_parameters);
        if (!nn_context) {
          M5.Lcd.println("NN context allocation failed");
          continue;
        }

        // 推論ネットワークの入力バッファ（float 型配列）を取得
        float *nn_input_buffer = nnablart_validation_input_buffer(nn_context, 0);

        // 型変換: uint8_t → float（今回は正規化は行わずキャストする）
        for (int i = 0; i < imageSize; i++) {
          nn_input_buffer[i] = (float)image_data[i];
        }

        // 推論の実行 ms単位でカウント
        int64_t start_time = millis();
        nnablart_validation_inference(nn_context);
        int64_t elapsed_time = millis() - start_time;

        // 推論結果の取得
        float *probs = nnablart_validation_output_buffer(nn_context, 0);
        int top_class = 0;
        float top_probability = 0.0f;
        for (int classNo = 0; classNo < NNABLART_VALIDATION_OUTPUT0_SIZE; classNo++) {
          //M5.Lcd.printf("class %d: %f\n", classNo, probs[classNo]);
          if (probs[classNo] > top_probability) {
            top_probability = probs[classNo];
            top_class = classNo;
          }
        }

        // 推論結果の表示
        M5.Lcd.printf("result: %d, ", top_class);
        M5.Lcd.printf("time:  %lld ms\n", elapsed_time);

        // 必要に応じて nn_context の解放処理を追加する
      } else {
        file.close();  // .raw でないファイルは閉じる
      }
    } else {
      file.close();  // ディレクトリの場合は閉じる
    }
  }
  root.close();

  if (!found) {
    M5.Lcd.println("No .raw files found in root directory");
  }

  // すべてのファイルを処理後、以降のループ処理を停止する場合は以下の無限ループなどで対応
  while (1) {
    delay(1000);
  }
}
