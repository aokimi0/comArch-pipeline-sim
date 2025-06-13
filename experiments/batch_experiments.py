#!/usr/bin/env python3
"""
批量实验脚本

运行不同到达率下的RAG流水线仿真实验，收集数据并保存。
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.simulation.simulator import RAGSimulator
from src.analysis.theory import QueueingTheoryAnalyzer
from src.analysis.visualization import plot_latency_throughput, plot_queue_length, plot_utilization, plot_theory_vs_simulation

def main():
    """主批量实验函数"""
    print("=" * 60)
    print("RAG流水线批量仿真实验")
    print("=" * 60)
    
    # 确保输出目录存在
    output_data_dir = "results/data"
    output_plots_dir = "results/plots"
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)
    
    # 获取配置
    config = Config()
    exp_config = config.experiment
    sim_config = config.simulation
    
    # 初始化仿真器和分析器
    simulator = RAGSimulator(sim_config)
    theory_analyzer = QueueingTheoryAnalyzer(sim_config)
    
    all_results = []
    theory_results_for_plots = {
        "arrival_rates": [],
        "avg_latency": [],
        "throughput": [],
        "generation_utilization": [],
        "generation_queue_length": []
    }
    
    # 运行批量实验
    print(f"\n开始运行 {len(exp_config.arrival_rates) * exp_config.num_runs} 次仿真...")
    df_results = simulator.run_batch_experiments(exp_config)
    
    if df_results.empty:
        print("没有收集到任何有效的仿真结果。")
        return
    
    # 保存原始数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_data_dir, f"simulation_results_{timestamp}.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\n仿真原始结果已保存至: {output_file}")
    
    # 准备理论数据用于绘图
    for rate in exp_config.arrival_rates:
        theory_analysis = theory_analyzer.analyze_rag_pipeline(rate)
        if theory_analysis["pipeline_metrics"]["is_stable"]:
            theory_results_for_plots["arrival_rates"].append(rate)
            theory_results_for_plots["avg_latency"].append(theory_analysis["pipeline_metrics"]["total_avg_response_time"])
            theory_results_for_plots["throughput"].append(rate) # 理论上稳定系统吞吐量等于到达率
            
            # 检查generation阶段是否存在且没有错误
            generation_stage = theory_analysis["stages"]["generation"]
            if "error" not in generation_stage:
                theory_results_for_plots["generation_utilization"].append(generation_stage["utilization"])
                theory_results_for_plots["generation_queue_length"].append(generation_stage["avg_queue_length"])
            else:
                # 如果generation阶段不稳定，则跳过该数据点
                print(f"警告: 到达率 {rate} 时generation阶段不稳定: {generation_stage['error']}")
                # 移除已添加的数据
                theory_results_for_plots["arrival_rates"].pop()
                theory_results_for_plots["avg_latency"].pop()
                theory_results_for_plots["throughput"].pop()
        else:
            # 对于不稳定状态，可以标记为无穷大或不包含在图中
            # 这里我们为了绘图方便，如果不稳定则不添加该点，或者可以添加一个很大的值
            print(f"警告: 到达率 {rate} 时系统不稳定")
            pass # 或者添加一个特殊值如NaN，让绘图函数处理
            
    df_theory = pd.DataFrame(theory_results_for_plots)
    
    # 绘图
    print("\n开始生成图表...")
    
    # 字体设置以支持中文
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 
        'Noto Sans CJK SC', 
        'WenQuanYi Zen Hei',
        'Source Han Sans CN',
        'DejaVu Sans'
    ]
    plt.rcParams['axes.unicode_minus'] = False
    
    # 图3：平均端到端延迟 vs. 到达率
    plot_latency_throughput(
        df_results, df_theory, 
        y_col='avg_latency', y_label='平均端到端延迟 (秒)',
        title='平均端到端延迟 vs. 到达率', 
        output_path=os.path.join(output_plots_dir, "avg_latency_vs_arrival_rate.png")
    )
    
    # 图4：系统吞吐量 vs. 到达率
    plot_latency_throughput(
        df_results, df_theory, 
        y_col='throughput', y_label='系统吞吐量 (请求/秒)',
        title='系统吞吐量 vs. 到达率', 
        output_path=os.path.join(output_plots_dir, "throughput_vs_arrival_rate.png")
    )
    
    # 图5：各阶段平均队列长度 vs. 到达率 (只绘制generation阶段的理论和仿真队列长度)
    plot_queue_length(
        df_results, df_theory, 
        queue_col='generation_queue_length', queue_label='生成阶段平均队列长度',
        title='生成阶段平均队列长度 vs. 到达率',
        output_path=os.path.join(output_plots_dir, "generation_queue_length_vs_arrival_rate.png")
    )
    
    # 图6：各阶段资源利用率 vs. 到达率 (只绘制generation阶段的理论和仿真利用率)
    plot_utilization(
        df_results, df_theory, 
        util_col='generation_utilization', util_label='生成阶段利用率',
        title='生成阶段资源利用率 vs. 到达率',
        output_path=os.path.join(output_plots_dir, "generation_utilization_vs_arrival_rate.png")
    )
    
    # 更多对比图 (如果需要，可以在这里添加)
    # plot_theory_vs_simulation(df_results, df_theory, output_plots_dir)
    
    print(f"所有图表已保存至: {output_plots_dir}")
    print("\n批量实验和绘图完成!")

if __name__ == "__main__":
    main() 