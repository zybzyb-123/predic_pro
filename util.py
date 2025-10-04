import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def handle_file():
    # 存放所有CSV文件的文件夹路径（替换成你的实际路径，注意路径中的斜杠是 / 或 \\）
    csv_folder = r"E:\\cache_code\\predic\\wind\\12"  
    # 合并后要保存的新文件路径（可自定义，如保存在桌面）
    output_file = r"E:\\cache_code\\predic\\wind\\trace0-11h.csv"

    # 存储所有CSV数据的列表
    all_data = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(csv_folder):
        # 只处理后缀为 .csv 的文件
        if filename.endswith(".csv"):
            # 拼接完整的文件路径
            file_path = os.path.join(csv_folder, filename)
            df = pd.read_csv(file_path, usecols=[0, 1], header=0, encoding="utf-8")
            # 将当前文件的数据加入列表
            all_data.append(df)

    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"合并完成！新文件已保存到：{output_file}")

# 数据准备
time_slots = ['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4']
cache_sizes = [768, 1024, 2048]
models = ['LRU', 'ARC', 'LeCaR', 'LSTM', 'Transformer']

# 每个时间段的命中率数据（顺序为 LRU, ARC, LeCaR, LSTM, Transformer）
hit_rates = {
    'Dataset 1': [
        [0.6929, 0.6929, 0.6929],
        [0.7583, 0.7583, 0.7583],
        [0.7583, 0.7583, 0.7583],
        [0.8380, 0.8779, 0.9011],
        [0.9093, 0.9191, 0.9265]
    ],
    'Dataset 2': [
        [0.7277, 0.7277, 0.7277],
        [0.8134, 0.8179, 0.8474],
        [0.7686, 0.8178, 0.8307],
        [0.8067, 0.8925, 0.9426],
        [0.8493, 0.9304, 0.9440]
    ],
    'Dataset 3': [
        [0.7322, 0.7322, 0.7322],
        [0.8320, 0.8403, 0.8504],
        [0.8331, 0.8468, 0.8496],
        [0.7764, 0.8730, 0.9479],
        [0.8337, 0.8860, 0.9455]
    ],
    'Dataset 4': [
        [0.7148, 0.7148, 0.7148],
        [0.7177, 0.7680, 0.7889],
        [0.7174, 0.7676, 0.7889],
        [0.7749, 0.8751, 0.9208],
        [0.8402, 0.8750, 0.9215]
    ]
}

colors = ['#f39c12', '#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

def create_hit_rate_figure():
    # 设置图形参数
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (10, 9)
    
    # 创建图形和子图
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    
    # 缓存大小标签
    cache_size_labels = ['768', '1024', '2048']
    
    # 绘制每个时间段的图表
    for i, time_slot in enumerate(time_slots):
        ax = axes[i]
        x = np.arange(len(cache_sizes))
        width = 0.1  
        
        ax.grid(True, axis='y', alpha=0.3, zorder=1)
        
        for j, model in enumerate(models):
            positions = x + j * width
            ax.bar(positions, [x * 100 for x in hit_rates[time_slot][j]], width, 
                   label=model, color=colors[j], alpha=0.8, zorder=2)
        
        # 设置子图标题
        ax.set_title(f'{time_slot}', fontsize=12, pad=15)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(cache_size_labels)
        ax.set_ylim(60, 100)
        
        if i % 2 == 0:
            ax.set_ylabel('Hit Rate (%)', fontsize=10)
        
        if i >= 2:
            ax.set_xlabel('Cache Size', fontsize=10)
    
    # 将图例放在顶部
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', 
               bbox_to_anchor=(0.52, 0.98), ncol=len(models), 
               fontsize=10)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    create_hit_rate_figure()