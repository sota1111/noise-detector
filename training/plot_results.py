import matplotlib.pyplot as plt
import re
import glob

def parse_log_file(log_file):
    """ログファイルから訓練結果を解析"""
    iterations = []
    losses = []
    train_errors = []
    train_accuracies = []
    val_errors = []
    val_accuracies = []
    val_iterations = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # 訓練データの解析
            train_match = re.search(r'Iteration (\d+): Loss = ([\d.]+), Training Error = ([\d.]+), Training Accuracy = ([\d.]+)', line)
            if train_match:
                iter_num = int(train_match.group(1))
                loss = float(train_match.group(2))
                error = float(train_match.group(3))
                accuracy = float(train_match.group(4))
                
                iterations.append(iter_num)
                losses.append(loss)
                train_errors.append(error)
                train_accuracies.append(accuracy)
            
            # 検証データの解析
            val_match = re.search(r'Iteration (\d+): Validation Error = ([\d.]+), Validation Accuracy = ([\d.]+)', line)
            if val_match:
                iter_num = int(val_match.group(1))
                val_error = float(val_match.group(2))
                val_accuracy = float(val_match.group(3))
                
                val_iterations.append(iter_num)
                val_errors.append(val_error)
                val_accuracies.append(val_accuracy)
    
    return iterations, losses, train_errors, train_accuracies, val_iterations, val_errors, val_accuracies

def plot_training_results():
    """最新のログファイルから訓練結果をプロット"""
    # 最新のログファイルを取得
    log_files = glob.glob('log/log_*.txt')
    if not log_files:
        print("ログファイルが見つかりません")
        return
    
    latest_log = max(log_files)
    print(f"ログファイル: {latest_log}")
    
    # データを解析
    iterations, losses, train_errors, train_accuracies, val_iterations, val_errors, val_accuracies = parse_log_file(latest_log)
    
    if not iterations:
        print("データが見つかりません")
        return
    
    # グラフを作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 損失の推移
    ax1.plot(iterations, losses, 'b-', alpha=0.7, label='Training Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 訓練精度の推移
    ax2.plot(iterations, train_accuracies, 'g-', alpha=0.7, label='Training Accuracy')
    if val_iterations and val_accuracies:
        ax2.plot(val_iterations, val_accuracies, 'r-', alpha=0.7, label='Validation Accuracy')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # エラー率の推移
    ax3.plot(iterations, train_errors, 'orange', alpha=0.7, label='Training Error')
    if val_iterations and val_errors:
        ax3.plot(val_iterations, val_errors, 'red', alpha=0.7, label='Validation Error')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Error Rate')
    ax3.set_title('Error Rate')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # 最近の精度推移（詳細表示）
    if len(iterations) > 100:
        recent_start = len(iterations) - 100
        ax4.plot(iterations[recent_start:], train_accuracies[recent_start:], 'g-', alpha=0.7, label='Training Accuracy (Recent)')
        if val_iterations and val_accuracies:
            # 最近の検証データも表示
            recent_val_idx = [i for i, x in enumerate(val_iterations) if x >= iterations[recent_start]]
            if recent_val_idx:
                ax4.plot([val_iterations[i] for i in recent_val_idx], 
                        [val_accuracies[i] for i in recent_val_idx], 'r-', alpha=0.7, label='Validation Accuracy (Recent)')
    else:
        ax4.plot(iterations, train_accuracies, 'g-', alpha=0.7, label='Training Accuracy')
        if val_iterations and val_accuracies:
            ax4.plot(val_iterations, val_accuracies, 'r-', alpha=0.7, label='Validation Accuracy')
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Recent Accuracy Progress')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # 統計情報を表示
    print(f"\n=== 訓練統計 ===")
    print(f"総イテレーション数: {len(iterations)}")
    print(f"最終訓練精度: {train_accuracies[-1]:.4f}")
    print(f"最終訓練損失: {losses[-1]:.4f}")
    
    if val_accuracies:
        print(f"最終検証精度: {val_accuracies[-1]:.4f}")
        print(f"最高検証精度: {max(val_accuracies):.4f}")
    
    # 90%達成の予測
    if train_accuracies:
        current_acc = train_accuracies[-1]
        if current_acc >= 0.9:
            print("🎉 目標精度90%達成!")
        else:
            remaining = 0.9 - current_acc
            print(f"90%まで残り: {remaining:.4f} ({remaining*100:.2f}%)")
    
    plt.show()

if __name__ == "__main__":
    plot_training_results()