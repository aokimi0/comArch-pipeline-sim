#!/usr/bin/env python3
"""
主实验脚本

运行RAG流水线仿真实验并生成基本分析结果。
"""

import os
import sys
import random
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import SimulationConfig
from src.simulation.simulator import RAGSimulator
from src.analysis.theory import QueueingTheoryAnalyzer
from src.pipeline.fsm import create_fsm_diagram

def main():
    """主实验函数"""
    print("=" * 60)
    print("RAG流水线仿真实验")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    # 1. 生成FSM状态转移图
    print("\n1. 生成有限状态机图...")
    create_fsm_diagram("results/plots/rag_fsm_diagram")
    
    # 2. 配置仿真参数
    config = SimulationConfig()
    config.simulation_time = 500.0  # 减少仿真时间用于快速测试
    config.random_seed = 42
    
    print(f"\n2. 仿真配置:")
    print(f"   仿真时长: {config.simulation_time}s")
    print(f"   资源配置: GPU槽位={config.num_gpu_slots}, 嵌入器={config.num_embedders}")
    
    # 3. 运行单次仿真
    print("\n3. 运行基础仿真...")
    simulator = RAGSimulator(config)
    
    arrival_rate = 0.4  # 测试到达率
    sim_result = simulator.run_single_simulation(
        arrival_rate=arrival_rate,
        warm_up_time=50.0,
        verbose=True
    )
    
    if "error" in sim_result:
        print(f"仿真失败: {sim_result['error']}")
        return
    
    # 4. 理论分析
    print("\n4. 进行理论分析...")
    theory_analyzer = QueueingTheoryAnalyzer(config)
    theory_result = theory_analyzer.analyze_rag_pipeline(arrival_rate)
    
    print(f"\n理论分析结果:")
    print(f"   瓶颈阶段: {theory_result['pipeline_metrics']['bottleneck_stage']}")
    print(f"   最大利用率: {theory_result['pipeline_metrics']['max_utilization']:.3f}")
    print(f"   理论总延迟: {theory_result['pipeline_metrics']['total_avg_response_time']:.3f}s")
    
    # 5. 理论与仿真对比
    print("\n5. 理论与仿真结果对比:")
    comparison = theory_analyzer.compare_with_simulation(theory_result, sim_result)
    
    if "latency" in comparison["metrics_comparison"]:
        latency_comp = comparison["metrics_comparison"]["latency"]
        print(f"   延迟 - 仿真: {latency_comp['simulation']:.3f}s, 理论: {latency_comp['theory']:.3f}s")
        print(f"        相对误差: {latency_comp['relative_error']:.1%}")
    
    print("\n   各阶段等待时间对比:")
    for stage, comp in comparison["stage_comparison"].items():
        print(f"   {stage:12}: 仿真={comp['simulation_wait_time']:.3f}s, "
              f"理论={comp['theory_wait_time']:.3f}s, "
              f"误差={comp['relative_error']:.1%}")
    
    # 6. 测试不同到达率
    print("\n6. 测试不同到达率的性能...")
    test_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print(f"{'到达率':>8} {'仿真延迟':>10} {'理论延迟':>10} {'仿真吞吐':>10} {'GPU利用率':>10}")
    print("-" * 55)
    
    for rate in test_rates:
        # 简化仿真（更短时间）
        test_config = SimulationConfig()
        test_config.simulation_time = 200.0
        test_config.random_seed = 42 + int(rate * 10)
        
        test_simulator = RAGSimulator(test_config)
        test_sim = test_simulator.run_single_simulation(
            arrival_rate=rate,
            warm_up_time=20.0,
            verbose=False
        )
        
        test_theory = theory_analyzer.analyze_rag_pipeline(rate)
        
        if "error" not in test_sim and test_theory["pipeline_metrics"]["is_stable"]:
            sim_latency = test_sim["avg_latency"]
            theory_latency = test_theory["pipeline_metrics"]["total_avg_response_time"]
            sim_throughput = test_sim["throughput"]
            gpu_util = test_theory["stages"]["generation"]["utilization"]
            
            print(f"{rate:8.1f} {sim_latency:10.3f} {theory_latency:10.3f} "
                  f"{sim_throughput:10.3f} {gpu_util:10.1%}")
        else:
            print(f"{rate:8.1f} {'不稳定':>10} {'不稳定':>10} {'不稳定':>10} {'不稳定':>10}")
    
    print("\n实验完成! 查看results/plots/目录获取生成的图表。")
    print("运行 'python experiments/batch_experiments.py' 进行完整的批量实验。")


if __name__ == "__main__":
    main() 