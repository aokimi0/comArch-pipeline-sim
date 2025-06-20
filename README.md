# RAG流水线性能分析项目

基于有限状态机和仿真的检索增强生成（RAG）流水线性能分析研究项目。

## 项目概述

本项目将RAG工作流建模为五阶段流水线，使用有限状态机（FSM）描述请求生命周期，通过SimPy离散事件仿真和排队论理论分析，深入研究RAG系统的性能特征和瓶颈。

## 主要特点

- **流水线建模**：将RAG分解为嵌入、检索、增强、生成、后处理五个阶段
- **FSM状态建模**：精确描述请求状态转移过程
- **离散事件仿真**：基于SimPy的高精度性能仿真
- **理论分析**：基于排队论的M/M/c模型分析
- **可视化分析**：丰富的性能图表和FSM状态图
- **数据驱动**：基于35次独立仿真运行的统计分析

## 关键发现

### 性能瓶颈识别
- **生成阶段为压倒性瓶颈**：GPU利用率最先饱和
- **临界负载点**：到达率超过0.4 req/s时系统性能急剧恶化
- **队列积压严重**：高负载下生成阶段队列长度呈指数增长

### 实验结果摘要

| 到达率 | 平均延迟 | 系统吞吐量 | GPU利用率 | 生成阶段队列长度 |
|--------|----------|------------|-----------|------------------|
| 0.1    | 2.89s    | 0.111 req/s| 22.2%     | 0.05            |
| 0.2    | 3.99s    | 0.206 req/s| 43.6%     | 0.29            |
| 0.3    | 4.75s    | 0.304 req/s| 60.5%     | 0.69            |
| 0.4    | 17.80s   | 0.403 req/s| 82.3%     | 6.64            |
| 0.5    | 26.06s   | 0.483 req/s| 96.1%     | 11.96           |
| 0.6    | 79.48s   | 0.503 req/s| 99.4%     | 46.30           |
| 0.7    | 174.02s  | 0.483 req/s| 100.0%    | 124.30          |

### 理论与仿真对比
- 在稳定区间（λ≤0.4），理论预测与仿真结果高度一致（相对误差5-15%）
- 理论分析准确预测了系统稳定性边界
- 高负载下仿真显示的性能衰减比理论预测更为剧烈

## 项目结构

```
├── src/                    # 源代码
│   ├── pipeline/          # RAG流水线实现
│   │   ├── fsm.py        # 有限状态机定义
│   │   └── rag_pipeline.py # 流水线主类
│   ├── simulation/        # 仿真模块
│   │   └── simulator.py  # 仿真器实现
│   ├── analysis/          # 分析模块
│   │   ├── theory.py     # 排队论分析
│   │   └── visualization.py # 可视化工具
│   └── utils/            # 工具模块
│       └── config.py     # 配置管理
├── experiments/           # 实验脚本
│   ├── run_simulation.py # 单次仿真实验
│   └── batch_experiments.py # 批量实验
├── results/              # 实验结果
│   ├── data/            # 仿真原始数据
│   └── plots/           # 生成的图表
├── docs/                # 文档
│   └── report.md        # 完整研究报告
└── requirements.txt     # Python依赖
```

## 环境要求

- Python 3.12+
- WSL Ubuntu 24.04
- 虚拟环境：`/opt/venvs/base`

## 快速开始

### 1. 安装依赖

```bash
source /opt/venvs/base/bin/activate
pip install -r requirements.txt
sudo apt-get install graphviz  # 用于FSM图生成
```

### 2. 运行单次仿真

```bash
python experiments/run_simulation.py
```

### 3. 运行批量实验

```bash
python experiments/batch_experiments.py
```

### 4. 查看结果

- 仿真数据：`results/data/simulation_results_*.csv`
- 图表：`results/plots/`
- 完整报告：`docs/report.md`

## 生成的图表

1. **FSM状态转移图** - RAG请求完整生命周期
2. **平均延迟 vs 到达率** - 系统响应时间分析
3. **系统吞吐量 vs 到达率** - 处理能力分析
4. **队列长度 vs 到达率** - 资源等待情况
5. **资源利用率 vs 到达率** - 瓶颈识别

## 优化建议

基于实验结果，提出以下优化策略：

1. **硬件扩展**：增加GPU数量从1个到2-3个
2. **连续批处理**：实现动态批次管理
3. **算法优化**：采用投机解码等高效推理算法
4. **负载均衡**：多GPU间智能路由

预期优化效果：
- 系统稳定边界从0.4 req/s提升到0.8 req/s
- 在0.4 req/s负载下延迟从17.80s降低到3-4s

## 学术价值

本研究展示了经典计算机体系结构原理在现代AI系统分析中的适用性：

- 建立了AI系统性能分析的新方法论
- 融合了体系结构、排队论和AI系统工程
- 为AI基础设施建设提供科学依据
- 为计算机体系结构教学提供现代案例