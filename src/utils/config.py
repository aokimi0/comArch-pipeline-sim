"""
配置管理模块

包含仿真参数的定义和管理。
"""

import random
from dataclasses import dataclass
from typing import Callable, Dict, Any


@dataclass
class SimulationConfig:
    """仿真配置参数"""
    
    # 随机种子
    random_seed: int = 42
    
    # 资源配置
    num_embedders: int = 2
    num_retrievers: int = 4
    num_augmenters: int = 8
    num_gpu_slots: int = 1
    num_postprocessors: int = 4
    
    # 负载配置
    arrival_rate: float = 0.5  # 请求/秒
    simulation_time: float = 1000.0  # 仿真时长（秒）
    
    # 处理时间分布（返回函数以便每次调用产生新的随机值）
    @staticmethod
    def t_embed() -> float:
        """嵌入阶段处理时间（正态分布）"""
        return max(0.01, random.normalvariate(0.1, 0.02))
    
    @staticmethod
    def t_retrieve() -> float:
        """检索阶段处理时间（正态分布）"""
        return max(0.01, random.normalvariate(0.3, 0.05))
    
    @staticmethod
    def t_augment() -> float:
        """增强阶段处理时间（均匀分布）"""
        return random.uniform(0.01, 0.02)
    
    @staticmethod
    def t_generate() -> float:
        """LLM生成阶段处理时间（指数分布）"""
        return random.expovariate(1.0 / 2.0)
    
    @staticmethod
    def t_postprocess() -> float:
        """后处理阶段处理时间（正态分布）"""
        return max(0.01, random.normalvariate(0.05, 0.01))


@dataclass
class ExperimentConfig:
    """实验配置"""
    
    # 到达率范围实验
    arrival_rates: list[float] = None
    num_runs: int = 5  # 每个配置运行次数
    warm_up_time: float = 100.0  # 预热时间
    
    # 输出配置
    output_dir: str = "results"
    save_raw_data: bool = True
    save_plots: bool = True
    
    def __post_init__(self):
        if self.arrival_rates is None:
            self.arrival_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


class Config:
    """全局配置管理器"""
    
    def __init__(self):
        self.simulation = SimulationConfig()
        self.experiment = ExperimentConfig()
    
    def update_simulation(self, **kwargs):
        """更新仿真配置"""
        for key, value in kwargs.items():
            if hasattr(self.simulation, key):
                setattr(self.simulation, key, value)
    
    def update_experiment(self, **kwargs):
        """更新实验配置"""
        for key, value in kwargs.items():
            if hasattr(self.experiment, key):
                setattr(self.experiment, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "simulation": self.simulation.__dict__,
            "experiment": self.experiment.__dict__
        }


# 全局配置实例
config = Config() 