"""
可视化模块

用于绘制仿真和理论分析结果的图表。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any
import os


def plot_latency_throughput(df_sim: pd.DataFrame, df_theory: pd.DataFrame,
                            y_col: str, y_label: str, title: str, output_path: str):
    """绘制延迟或吞吐量 vs. 到达率图表"""
    plt.figure(figsize=(10, 6))
    
    # 仿真结果（平均值和标准差）
    sim_grouped = df_sim.groupby('arrival_rate')[y_col].agg(['mean', 'std']).reset_index()
    plt.errorbar(sim_grouped['arrival_rate'], sim_grouped['mean'], 
                 yerr=sim_grouped['std'], fmt='o-', capsize=5, label='仿真平均值 (带标准差)', color='blue')
    
    # 理论结果
    if not df_theory.empty:
        plt.plot(df_theory['arrival_rates'], df_theory[y_col], 'x--', label='理论值', color='red')
    
    plt.xlabel('到达率 (请求/秒)')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"图表已保存至: {output_path}")


def plot_queue_length(df_sim: pd.DataFrame, df_theory: pd.DataFrame,
                      queue_col: str, queue_label: str, title: str, output_path: str):
    """绘制队列长度 vs. 到达率图表"""
    plt.figure(figsize=(10, 6))
    
    # 仿真结果（平均值和标准差）
    # 数据已经扁平化，直接使用列名，例如 'stage_generation_avg_queue_length'
    stage_name = queue_col.replace("_queue_length","")
    col_name = f"stage_{stage_name}_avg_queue_length"
    
    if col_name not in df_sim.columns:
        print(f"警告: 列 {col_name} 不存在于仿真数据中")
        print(f"可用列: {list(df_sim.columns)}")
        return
    
    sim_grouped = df_sim.groupby('arrival_rate')[col_name].agg(['mean', 'std']).reset_index()
    
    plt.errorbar(sim_grouped['arrival_rate'], sim_grouped['mean'], 
                 yerr=sim_grouped['std'], fmt='o-', capsize=5, label='仿真平均队列长度', color='blue')
    
    # 理论结果
    if not df_theory.empty:
        plt.plot(df_theory['arrival_rates'], df_theory[queue_col], 'x--', label='理论队列长度', color='red')
    
    plt.xlabel('到达率 (请求/秒)')
    plt.ylabel(queue_label)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"图表已保存至: {output_path}")


def plot_utilization(df_sim: pd.DataFrame, df_theory: pd.DataFrame,
                     util_col: str, util_label: str, title: str, output_path: str):
    """绘制资源利用率 vs. 到达率图表"""
    plt.figure(figsize=(10, 6))
    
    # 仿真结果（平均值和标准差）
    # 数据已经扁平化，直接使用列名，例如 'stage_generation_avg_utilization'
    stage_name = util_col.replace("_utilization","")
    col_name = f"stage_{stage_name}_avg_utilization"
    
    if col_name not in df_sim.columns:
        print(f"警告: 列 {col_name} 不存在于仿真数据中")
        print(f"可用列: {list(df_sim.columns)}")
        return
    
    sim_grouped = df_sim.groupby('arrival_rate')[col_name].agg(['mean', 'std']).reset_index()

    plt.errorbar(sim_grouped['arrival_rate'], sim_grouped['mean'], 
                 yerr=sim_grouped['std'], fmt='o-', capsize=5, label='仿真平均利用率', color='blue')

    # 理论结果
    if not df_theory.empty:
        plt.plot(df_theory['arrival_rates'], df_theory[util_col], 'x--', label='理论利用率', color='red')
    
    plt.xlabel('到达率 (请求/秒)')
    plt.ylabel(util_label)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"图表已保存至: {output_path}")


def plot_theory_vs_simulation(df_sim: pd.DataFrame, df_theory: pd.DataFrame, output_dir: str):
    """绘制理论与仿真对比图（通用函数，可扩展）"""
    # 此处可以添加更多具体的对比图，例如：
    # 1. 延迟对比
    # 2. 吞吐量对比
    # 3. 各阶段等待时间对比
    
    # 示例：绘制延迟对比图
    plot_latency_throughput(
        df_sim, df_theory, 
        y_col='avg_latency', y_label='平均延迟 (秒)',
        title='平均延迟：理论 vs 仿真',
        output_path=os.path.join(output_dir, "avg_latency_theory_vs_sim.png")
    )


if __name__ == "__main__":
    # 示例用法
    # 创建一些虚拟数据用于测试
    sim_data = {
        'arrival_rate': [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
        'avg_latency': [2.5, 2.6, 3.0, 3.1, 4.5, 4.7],
        'throughput': [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
        'stage_metrics': [
            {'embedding': {'avg_queue_length': 0.01, 'avg_utilization': 0.1}},
            {'embedding': {'avg_queue_length': 0.02, 'avg_utilization': 0.11}},
            {'embedding': {'avg_queue_length': 0.03, 'avg_utilization': 0.12}},
            {'embedding': {'avg_queue_length': 0.04, 'avg_utilization': 0.13}},
            {'embedding': {'avg_queue_length': 0.05, 'avg_utilization': 0.14}},
            {'embedding': {'avg_queue_length': 0.06, 'avg_utilization': 0.15}},
        ],
    }
    theory_data = {
        'arrival_rates': [0.1, 0.2, 0.3],
        'avg_latency': [2.4, 2.9, 4.0],
        'throughput': [0.1, 0.2, 0.3],
        'generation_queue_length': [0.05, 0.2, 0.8],
        'generation_utilization': [0.2, 0.4, 0.6],
    }
    
    df_sim_test = pd.DataFrame(sim_data)
    df_theory_test = pd.DataFrame(theory_data)
    
    # 确保输出目录存在
    os.makedirs("temp_plots", exist_ok=True)
    
    plot_latency_throughput(df_sim_test, df_theory_test, 
                            y_col='avg_latency', y_label='平均延迟 (秒)',
                            title='测试延迟图', output_path="temp_plots/test_latency.png")
    
    plot_queue_length(df_sim_test, df_theory_test, 
                      queue_col='embedding_queue_length', queue_label='嵌入阶段平均队列长度',
                      title='测试队列长度图', output_path="temp_plots/test_queue_length.png")
    
    plot_utilization(df_sim_test, df_theory_test, 
                     util_col='embedding_utilization', util_label='嵌入阶段利用率',
                     title='测试利用率图', output_path="temp_plots/test_utilization.png")

    print("测试图表已生成在 temp_plots/ 目录下。") 