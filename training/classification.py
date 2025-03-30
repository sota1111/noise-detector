from __future__ import absolute_import
from six.moves import range

import os
import glob
import datetime  # 追加：日時取得のため

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

from args import get_args  # コマンドライン引数を取得
from _checkpoint_nnp_util import save_checkpoint, load_checkpoint, save_nnp  # チェックポイントの保存・復元関数

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

input_size = 800

def load_audio_csv(file_path, normalize=True):
    """
    CSVファイルから音声データを読み込む関数。
    
    Args:
        file_path (str): 読み込むCSVファイルのパス。
        normalize (bool): 各サンプル内の最大値で0~1に正規化するかどうか。
    
    Returns:
        np.ndarray: 読み込んだ音声データの配列。（shape: (1,input_size) を想定）
    """
    try:
        df = pd.read_csv(file_path, header=None)
        data = df.values.flatten()[:input_size]
        if normalize:
            data = (data - 2048) / 2048.0
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def data_iterator_audio(batch_size, train, rng=None):
    """
    dataset/label0, dataset/label1 に格納されたCSVファイルから音声データを読み込み、
    バッチごとに返すイテレータ（常に batch_size 個のサンプルを返す）。
    """
    label0_files = glob.glob(os.path.join("dataset", "label0", "*.csv"))
    label1_files = glob.glob(os.path.join("dataset", "label1", "*.csv"))
    files = [(f, 0) for f in label0_files] + [(f, 1) for f in label1_files]
    rng = rng or np.random.RandomState(1234)

    while True:
        rng.shuffle(files)
        i = 0
        num_files = len(files)
        while i < num_files:
            batch_data = []
            batch_labels = []
            # 必ずバッチサイズ分のデータを集める（枯渇時はファイルリストの先頭に戻る）
            while len(batch_data) < batch_size:
                if i >= num_files:
                    i = 0
                    rng.shuffle(files)
                file_path, label = files[i]
                i += 1
                data = load_audio_csv(file_path)
                if data is None:
                    continue  # 読み込みエラーの場合はスキップ

                # データの形状を (1, input_size, 1) に変換
                data = data.reshape(1, input_size, 1)
                batch_data.append(data)
                batch_labels.append([label])
            batch_data = np.stack(batch_data, axis=0)
            batch_labels = np.array(batch_labels, dtype=np.int32)
            yield batch_data, batch_labels


def plot_training_progress(iteration_list, loss_list, error_list, val_iteration_list, val_error_list, val_accuracy_list):
    """
    訓練過程の損失とエラー率の推移をプロットする関数。
    """
    print("Validation Error List:", val_error_list)
    print("Validation Accuracy List:", val_accuracy_list)

    plt.figure(figsize=(12, 5))
    
    # 損失の推移
    plt.subplot(1, 2, 1)
    plt.plot(iteration_list, loss_list, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()
    
    # エラー率と精度の推移
    plt.subplot(1, 2, 2)
    plt.plot(iteration_list, error_list, label='Training Error', color='r')
    plt.plot(val_iteration_list, val_error_list, label='Validation Error', color='b')
    plt.plot(val_iteration_list, val_accuracy_list, label='Validation Accuracy', color='g')
    plt.xlabel('Iteration')
    plt.ylabel('Rate')
    plt.title('Training and Validation Error/Accuracy')
    plt.legend()
    
    plt.show()
    
def categorical_error(pred, label):
    """
    予測結果と正解ラベルを比較し、分類エラー率を計算する。
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()

def augmentation(h, test, aug):
    """
    音声データの拡張処理を実施する関数。
    """
    if aug is None:
        aug = not test
    if aug:
        # ランダムな音量スケーリング（0.8〜1.2倍）
        batch_size = h.shape[0]
        scale_factors = np.random.uniform(0.8, 1.2, size=(batch_size, 1, 1, 1)).astype(np.float32)
        scale = nn.Variable(scale_factors.shape, need_grad=False)
        scale.d = scale_factors
        h = h * scale
        
        # ガウスノイズの追加（平均0、標準偏差0.005）
        noise = F.randn(h.shape, 0, 0.005)
        h = h + noise
    return h


def audio_conv1d_prediction(x_in, test=False, aug=None):
    """
    batch_normalizationは、学習時とテスト時で挙動が異なるため、コメントアウトしておく。
    """
    # 1層目: 畳み込み → ReLU → MaxPooling
    #h = PF.batch_normalization(x_in, name='bn1')
    #h = PF.convolution(h, 3, (3, 1), name='conv1d_1')
    c1 = PF.convolution(x_in, 3, (3, 1), name='conv1d_1')
    c1 = F.relu(c1)
    c1 = F.max_pooling(c1, (2, 1))
    
    # 2層目: 畳み込み → ReLU → MaxPooling
    #h = PF.batch_normalization(h, name='bn2')
    c2 = PF.convolution(c1, 6, (3, 1), name='conv1d_2')
    c2 = F.relu(c2)
    c2 = F.max_pooling(c2, (2, 1))
    
    # 3層目: 畳み込み → ReLU
    #h = PF.batch_normalization(h, name='bn3')
    c3 = PF.convolution(c2, 12, (3, 1), name='conv1d_3')
    c3 = F.relu(c3)
    
    # グローバル平均プーリング
    c3 = F.global_average_pooling(c3)
    
    # 全結合層（24ユニット, ReLU活性化）
    #h = PF.batch_normalization(h, name='bn4')
    c4 = PF.affine(c3, 24, name='fc1')
    c4 = F.relu(c4)
    
    # 出力層（2クラス分類: 全結合層）
    #h = PF.batch_normalization(h, name='bn5')
    c5 = PF.affine(c4, 2, name='fc2')
    # F.softmax_cross_entropy は内部で softmax を計算するため、既に softmax 済みの出力を渡すと正しい損失が計算されず、学習にも悪影響が出る
    #y = F.softmax(logits)
    return c5

def train():
    # 実行時に新規ログファイルを作成するための処理
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    now = datetime.datetime.now()
    # 日付と時間（秒まで）を含むログファイル名を生成
    log_filename = os.path.join(log_dir, f"log_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    print(f"ログは {log_filename} に出力されます。")

    # 損失とエラー率を記録するリスト
    loss_list = []
    error_list = []
    accuracy_list = []
    val_error_list = []
    val_accuracy_list = []
    iteration_list = []
    val_iteration_list = []
    
    """
    audioデータを使用してCNNを訓練する。
    """
    args = get_args()  # コマンドライン引数の取得

    from numpy.random import seed, RandomState
    seed(0)

    # 計算コンテキストの設定
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context(args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # 使用するネットワークの選択
    if args.net == 'conv1d':
        audio_cnn_prediction = audio_conv1d_prediction
    else:
        raise ValueError("Unknown network type {}".format(args.net))

    # TRAIN
    audio = nn.Variable([args.batch_size, 1, input_size, 1])
    label = nn.Variable([args.batch_size, 1])
    pred = audio_cnn_prediction(audio, test=False, aug=args.augment_train)
    pred.persistent = True
    loss = F.mean(F.softmax_cross_entropy(pred, label))

    # TEST
    vaudio = nn.Variable([args.batch_size, 1, input_size, 1])
    vlabel = nn.Variable([args.batch_size, 1])
    vpred = audio_cnn_prediction(vaudio, test=True, aug=args.augment_test)

    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())
    start_point = 0

    if args.checkpoint is not None:
        start_point = load_checkpoint(args.checkpoint, solver)

    print(f"Current learning rate: {solver.learning_rate()}")

    from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=1)
    monitor_err = MonitorSeries("Training error", monitor, interval=1)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=10)
    monitor_verr = MonitorSeries("Test error", monitor, interval=1)

    contents = save_nnp({'x': vaudio}, {'y': vpred}, args.batch_size)
    save.save(os.path.join(args.model_save_path,
                           '{}_result_epoch0.nnp'.format(args.net)), contents)

    from numpy.random import RandomState
    train_data = iter(data_iterator_audio(args.batch_size, True, rng=RandomState(1223)))
    val_data = iter(data_iterator_audio(args.batch_size, False))
    
    # Training loop.
    for i in range(start_point, args.max_iter):
        if i % args.val_interval == 0:
            ve = 0.0
            va = 0.0
            for j in range(args.val_iter):
                vaudio.d, vlabel.d = next(val_data)
                vpred.forward(clear_buffer=True)
                ve += categorical_error(vpred.d, vlabel.d)
                va += 1 - categorical_error(vpred.d, vlabel.d)
            ve /= args.val_iter
            va /= args.val_iter
            print(f"Iteration {i}: Validation Error = {ve:.4f}, Validation Accuracy = {va:.4f}")
            # ログ出力時は生成したファイル名を使用
            with open(log_filename, 'a') as log_file:
                log_file.write(f"Iteration {i}: Validation Error = {ve:.4f}, Validation Accuracy = {va:.4f}\n")
            val_error_list.append(ve)
            val_accuracy_list.append(va)
            val_iteration_list.append(i)

        if i % args.model_save_interval == 0:
            save_checkpoint(args.model_save_path, i, solver)
            
        audio.d, label.d = next(train_data)
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()
        loss.data.cast(np.float32, ctx)
        pred.data.cast(np.float32, ctx)
        e = categorical_error(pred.d, label.d)
        monitor_loss.add(i, loss.d.copy())
        monitor_err.add(i, e)
        monitor_time.add(i)
        e = categorical_error(pred.d, label.d)
        a = 1 - e
        print(f"Iteration {i}: Loss = {loss.d.copy():.4f}, Training Error = {e:.4f}, Training Accuracy = {a:.4f}")
        with open(log_filename, 'a') as log_file:
            log_file.write(f"Iteration {i}: Loss = {loss.d.copy():.4f}, Training Error = {e:.4f}, Training Accuracy = {a:.4f}\n")
        
        loss_list.append(loss.d.copy())
        error_list.append(e)
        accuracy_list.append(a)
        iteration_list.append(i)

    ve = 0.0
    va = 0.0

    for j in range(args.val_iter):
        vaudio.d, vlabel.d = next(val_data)
        vpred.forward(clear_buffer=True)
        ve += categorical_error(vpred.d, vlabel.d)
        va += 1 - categorical_error(vpred.d, vlabel.d)
    ve /= args.val_iter
    va /= args.val_iter
    print(f"Final Validation Error: {ve:.4f}, Final Validation Accuracy: {va:.4f}")
    with open(log_filename, 'a') as log_file:
        log_file.write(f"Final Validation Error: {ve:.4f}, Final Validation Accuracy: {va:.4f}\n")
    val_error_list.append(ve)
    val_accuracy_list.append(va)
    val_iteration_list.append(i)
    
    parameter_file = os.path.join(
        args.model_save_path, '{}_params_{:06}.h5'.format(args.net, args.max_iter))
    nn.save_parameters(parameter_file)

    contents = save_nnp({'x': vaudio}, {'y': vpred}, args.batch_size)
    save.save(os.path.join(args.model_save_path, '{}_result.nnp'.format(args.net)), contents)    
    plot_training_progress(iteration_list, loss_list, error_list, val_iteration_list, val_error_list, val_accuracy_list)
    plot_confusion_matrix(vaudio, vlabel, vpred, args.batch_size, args.val_iter, log_filename)
    
    # 個別のサンプルに対する確率を表示
    validate_specific_samples(parameter_file)

def validate_specific_samples(model_path, num_samples=10):
    """
    label0とlabel1のデータそれぞれnum_samples個に対してvalidationを行い、
    各データに対する確率を表示する関数
    
    Args:
        model_path (str): 学習済みモデルのパス
        num_samples (int): 各ラベルから検証するサンプル数
    """
    # モデルのロード
    audio = nn.Variable([1, 1, input_size, 1])
    pred = audio_conv1d_prediction(audio, test=True)
    nn.load_parameters(model_path)
    
    # データの準備
    label0_files = glob.glob(os.path.join("dataset", "label0", "*.csv"))[:num_samples]
    label1_files = glob.glob(os.path.join("dataset", "label1", "*.csv"))[:num_samples]
    
    print("\n=== Label 0 のサンプルの検証 ===")
    for file_path in label0_files:
        data = load_audio_csv(file_path)
        if data is not None:
            audio.d = data.reshape(1, 1, input_size, 1)
            pred.forward(clear_buffer=True)
            pred_reshaped = F.reshape(pred, (-1, 2))
            pred_reshaped.forward(clear_buffer=True)
            softmax_out = F.softmax(pred_reshaped)
            softmax_out.forward(clear_buffer=True)
            probabilities = softmax_out.d[0]
            print(f"\nFile: {os.path.basename(file_path)}")
            print(f"Label 0の確率: {probabilities[0]:.4f}")
            print(f"Label 1の確率: {probabilities[1]:.4f}")
            print(f"予測: Label {np.argmax(probabilities)}")
    
    print("\n=== Label 1 のサンプルの検証 ===")
    for file_path in label1_files:
        data = load_audio_csv(file_path)
        if data is not None:
            audio.d = data.reshape(1, 1, input_size, 1)
            pred.forward(clear_buffer=True)
            pred_reshaped = F.reshape(pred, (-1, 2))
            pred_reshaped.forward(clear_buffer=True)
            softmax_out = F.softmax(pred_reshaped)
            softmax_out.forward(clear_buffer=True)
            probabilities = softmax_out.d[0]
            print(f"\nFile: {os.path.basename(file_path)}")
            print(f"Label 0の確率: {probabilities[0]:.4f}")
            print(f"Label 1の確率: {probabilities[1]:.4f}")
            print(f"予測: Label {np.argmax(probabilities)}")


def plot_confusion_matrix(vaudio, vlabel, vpred, batch_size, val_iter, log_filename):
    """
    検証データを使用して混合行列を計算し表示する関数
    
    Args:
        vaudio: 検証用の入力データ変数
        vlabel: 検証用のラベル変数
        vpred: 検証用の予測変数
        batch_size: バッチサイズ
        val_iter: 検証イテレーション数
        log_filename: ログを出力するファイルパス
    """
    print("\n最終的な検証データでの混合行列を計算中...")
    y_true = []
    y_pred = []
    
    # 検証データで予測を収集
    val_data = iter(data_iterator_audio(batch_size, False))
    for j in range(val_iter):
        vaudio.d, vlabel.d = next(val_data)
        vpred.forward(clear_buffer=True)
        pred_labels = vpred.d.argmax(axis=1)
        y_true.extend(vlabel.d.flatten())
        y_pred.extend(pred_labels)
    
    # 混合行列の計算と表示
    cm = confusion_matrix(y_true, y_pred)
    plt.close('all')  # すべての図をクリア
    fig = plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Label 0', 'Label 1'])
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title('Validation Confusion Matrix')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # 描画が完了するのを待つ
    plt.show(block=True)
    plt.close(fig)
    
    # 混合行列の詳細な結果をログに出力
    with open(log_filename, 'a') as log_file:
        log_file.write("\n混合行列:\n")
        log_file.write(str(cm))
        log_file.write("\n")
        
        # クラスごとの精度の計算
        for i in range(len(cm)):
            true_positive = cm[i][i]
            total = sum(cm[i])
            accuracy = true_positive / total if total > 0 else 0
            log_file.write(f"\nLabel {i} の精度: {accuracy:.4f}")

if __name__ == '__main__':
    train()
