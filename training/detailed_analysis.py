import matplotlib.pyplot as plt
import re
import numpy as np

def detailed_analysis():
    """詳細な学習分析とグラフ表示"""
    log_file = 'log/log_2025-07-12_22-21-46.txt'
    
    iterations = []
    losses = []
    train_accuracies = []
    val_iterations = []
    val_accuracies = []
    
    # ログファイルを解析
    with open(log_file, 'r') as f:
        for line in f:
            # 訓練データの解析
            train_match = re.search(r'Iteration (\d+): Loss = ([\d.]+), Training Error = ([\d.]+), Training Accuracy = ([\d.]+)', line)
            if train_match:
                iter_num = int(train_match.group(1))
                loss = float(train_match.group(2))
                accuracy = float(train_match.group(4))
                
                iterations.append(iter_num)
                losses.append(loss)
                train_accuracies.append(accuracy)
            
            # 検証データの解析
            val_match = re.search(r'Iteration (\d+): Validation Error = ([\d.]+), Validation Accuracy = ([\d.]+)', line)
            if val_match:
                iter_num = int(val_match.group(1))
                val_accuracy = float(val_match.group(3))
                
                val_iterations.append(iter_num)
                val_accuracies.append(val_accuracy)
    
    # 4つのサブプロットを作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 損失の推移
    ax1.plot(iterations, losses, 'b-', alpha=0.8, linewidth=1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Progression')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(losses) * 1.1)
    
    # 学習率変更ポイントを表示
    lr_changes = [1000, 3000, 6000]
    for change_point in lr_changes:
        if change_point < max(iterations):
            ax1.axvline(x=change_point, color='red', linestyle='--', alpha=0.5, label=f'LR change at {change_point}')
    ax1.legend()
    
    # 2. 精度の推移
    ax2.plot(iterations, train_accuracies, 'g-', alpha=0.7, label='Training Accuracy', linewidth=1)
    ax2.plot(val_iterations, val_accuracies, 'r-', marker='o', markersize=4, label='Validation Accuracy', linewidth=2)
    ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.8, label='Target 90%')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Progression')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # 3. 最近1000イテレーションの詳細
    if len(iterations) > 1000:
        recent_start = len(iterations) - 1000
        ax3.plot(iterations[recent_start:], train_accuracies[recent_start:], 'g-', alpha=0.7, label='Training Accuracy')
        
        # 最近の検証データ
        recent_val_idx = [i for i, x in enumerate(val_iterations) if x >= iterations[recent_start]]
        if recent_val_idx:
            ax3.plot([val_iterations[i] for i in recent_val_idx], 
                    [val_accuracies[i] for i in recent_val_idx], 'r-', marker='o', markersize=4, label='Validation Accuracy')
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Recent 1000 Iterations Detail')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # 4. 学習率とパフォーマンスの関係
    # 学習率スケジュールに基づく分析
    phase_colors = ['blue', 'green', 'orange', 'red']
    phase_labels = ['LR=0.01 (0-1000)', 'LR=0.005 (1000-3000)', 'LR=0.002 (3000-6000)', 'LR=0.001 (6000+)']
    
    for i, (start, end) in enumerate([(0, 1000), (1000, 3000), (3000, 6000), (6000, len(iterations))]):
        mask = (np.array(iterations) >= start) & (np.array(iterations) < end)
        if np.any(mask):
            phase_iters = np.array(iterations)[mask]
            phase_accs = np.array(train_accuracies)[mask]
            ax4.scatter(phase_iters, phase_accs, c=phase_colors[i], alpha=0.6, s=1, label=phase_labels[i])
    
    # 検証精度も追加
    ax4.plot(val_iterations, val_accuracies, 'r-', marker='o', markersize=6, linewidth=3, label='Validation Accuracy', alpha=0.8)
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Learning Rate Schedule vs Performance')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # 統計情報を出力
    print(f"\n🎯 詳細分析結果")
    print(f"=" * 50)
    print(f"総イテレーション数: {len(iterations):,}")
    print(f"最終訓練精度: {train_accuracies[-1]:.4f} ({train_accuracies[-1]*100:.2f}%)")
    print(f"最終訓練損失: {losses[-1]:.4f}")
    print(f"最終検証精度: {val_accuracies[-1]:.4f} ({val_accuracies[-1]*100:.2f}%)")
    print(f"最高検証精度: {max(val_accuracies):.4f} ({max(val_accuracies)*100:.2f}%)")
    
    # 学習フェーズごとの分析
    print(f"\n📊 学習フェーズ分析")
    print(f"-" * 30)
    
    phases = [
        (0, 1000, "初期学習 (LR=0.01)"),
        (1000, 3000, "中期学習 (LR=0.005)"),
        (3000, 6000, "後期学習 (LR=0.002)"),
        (6000, len(iterations), "微調整 (LR=0.001)")
    ]
    
    for start, end, phase_name in phases:
        if end <= len(iterations):
            phase_mask = (np.array(iterations) >= start) & (np.array(iterations) < end)
            if np.any(phase_mask):
                phase_accs = np.array(train_accuracies)[phase_mask]
                avg_acc = np.mean(phase_accs)
                final_acc = phase_accs[-1] if len(phase_accs) > 0 else 0
                print(f"{phase_name}: 平均精度 {avg_acc:.3f}, 最終精度 {final_acc:.3f}")
    
    # 改善推定
    if len(val_accuracies) >= 2:
        recent_trend = val_accuracies[-1] - val_accuracies[-2] if len(val_accuracies) >= 2 else 0
        print(f"\n🔮 予測")
        print(f"-" * 20)
        print(f"直近トレンド: {recent_trend:+.4f}")
        
        current_acc = val_accuracies[-1]
        if current_acc >= 0.9:
            print("🎉 目標90%達成済み！")
        else:
            remaining = 0.9 - current_acc
            if recent_trend > 0:
                est_iters = remaining / recent_trend * 400  # 400は検証間隔
                print(f"90%到達予想: 約{est_iters:.0f}イテレーション後")
            else:
                print(f"90%まで残り: {remaining:.4f} ({remaining*100:.2f}%)")
    
    plt.show()

if __name__ == "__main__":
    detailed_analysis()