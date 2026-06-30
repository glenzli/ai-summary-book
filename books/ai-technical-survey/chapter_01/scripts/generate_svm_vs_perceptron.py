import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def plot_svm_vs_perceptron():
    # 设置随机种子保证可复现
    np.random.seed(42)

    # 生成简单的线性可分数据
    # 类 1
    X1 = np.random.randn(20, 2) * 0.5 + [2, 2]
    # 类 2
    X2 = np.random.randn(20, 2) * 0.5 + [4, 4]

    # 为了演示效果，手动添加几个关键的支持向量点，使 Margin 更明显
    X1 = np.vstack([X1, [2.8, 2.8]]) 
    X2 = np.vstack([X2, [3.2, 3.2]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 左图：感知机 (Perceptron) ---
    ax = axes[0]
    ax.scatter(X1[:, 0], X1[:, 1], c='#DAE8FC', edgecolors='#6C8EBF', s=80, label='Class 0')
    ax.scatter(X2[:, 0], X2[:, 1], c='#F8CECC', edgecolors='#B85450', s=80, marker='s', label='Class 1')
    
    # 绘制多条可能的分割线
    x_vals = np.array([0, 6])
    
    # 线 1 (不太好)
    y_vals_1 = -1.2 * x_vals + 6.5
    ax.plot(x_vals, y_vals_1, '--', color='#B85450', alpha=0.6, linewidth=2, label='Solution A')
    
    # 线 2 (不太好)
    y_vals_2 = -0.8 * x_vals + 5.5
    ax.plot(x_vals, y_vals_2, '--', color='#82B366', alpha=0.6, linewidth=2, label='Solution B')
    
    # 线 3 (还可以)
    y_vals_3 = -1.0 * x_vals + 6.0
    ax.plot(x_vals, y_vals_3, '--', color='gray', alpha=0.6, linewidth=2, label='Solution C')

    ax.set_title('Perceptron: Many Possible Solutions\n(Random Walk)', fontsize=14)
    ax.set_xlim(1, 5)
    ax.set_ylim(1, 5)
    ax.grid(True, linestyle=':', alpha=0.6)
    # ax.legend()

    # --- 右图：SVM ---
    ax = axes[1]
    ax.scatter(X1[:, 0], X1[:, 1], c='#DAE8FC', edgecolors='#6C8EBF', s=80, alpha=0.4) # 淡化非支持向量
    ax.scatter(X2[:, 0], X2[:, 1], c='#F8CECC', edgecolors='#B85450', s=80, marker='s', alpha=0.4)

    # 强调支持向量 (Support Vectors)
    sv1 = [2.8, 2.8]
    sv2 = [3.2, 3.2]
    ax.scatter([sv1[0]], [sv1[1]], c='#DAE8FC', edgecolors='#6C8EBF', s=150, linewidth=3)
    ax.scatter([sv2[0]], [sv2[1]], c='#F8CECC', edgecolors='#B85450', s=150, marker='s', linewidth=3)

    # 最佳分割线
    slope = -1
    intercept = 6.0
    y_vals_svm = slope * x_vals + intercept
    ax.plot(x_vals, y_vals_svm, '-', color='#000000', linewidth=3, label='Max Margin Hyperplane')

    # Margin 边界
    margin = 0.4 * np.sqrt(2) # 垂直距离
    # 实际上这里简化处理，直接画平行的线穿过支持向量
    # y = -x + 5.6 (过 2.8, 2.8)
    # y = -x + 6.4 (过 3.2, 3.2)
    ax.plot(x_vals, slope * x_vals + 5.6, ':', color='#000000', linewidth=1.5)
    ax.plot(x_vals, slope * x_vals + 6.4, ':', color='#000000', linewidth=1.5)

    # 绘制 Margin 区域
    ax.fill_between(x_vals, slope * x_vals + 5.6, slope * x_vals + 6.4, color='#FFF2CC', alpha=0.3, label='Max Margin')
    
    # 标注箭头
    ax.annotate('', xy=(3.0, 3.0), xytext=(3.15, 3.15), arrowprops=dict(arrowstyle='<->', linewidth=1.5))
    ax.text(3.1, 3.0, 'Margin', fontsize=10, rotation=0)

    ax.set_title('SVM: Unique Optimal Solution\n(Max Margin)', fontsize=14)
    ax.set_xlim(1, 5)
    ax.set_ylim(1, 5)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('chapter_01/images/svm_vs_perceptron.png', dpi=300, bbox_inches='tight')
    print("Image saved to chapter_01/images/svm_vs_perceptron.png")

if __name__ == "__main__":
    plot_svm_vs_perceptron()
