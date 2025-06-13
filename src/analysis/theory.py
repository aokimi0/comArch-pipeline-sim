"""
排队论理论分析模块

基于M/M/c模型和利特尔法则进行理论性能预测。
"""

import math
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np
from ..utils.config import SimulationConfig


@dataclass
class QueueingModel:
    """排队模型参数"""
    
    arrival_rate: float  # λ - 到达率
    service_rate: float  # μ - 服务率
    num_servers: int     # c - 服务器数量
    
    @property
    def traffic_intensity(self) -> float:
        """业务强度 ρ = λ/μ"""
        return self.arrival_rate / self.service_rate
    
    @property
    def utilization(self) -> float:
        """系统利用率 ρ/c"""
        return self.traffic_intensity / self.num_servers
    
    @property
    def is_stable(self) -> bool:
        """系统是否稳定"""
        return self.utilization < 1.0


class QueueingTheoryAnalyzer:
    """排队论分析器"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
    def analyze_single_stage(self, arrival_rate: float, 
                           avg_service_time: float, 
                           num_servers: int) -> Dict[str, float]:
        """分析单个M/M/c队列"""
        service_rate = 1.0 / avg_service_time
        model = QueueingModel(arrival_rate, service_rate, num_servers)
        
        if not model.is_stable:
            return {
                "error": f"系统不稳定，利用率 = {model.utilization:.3f} >= 1.0",
                "utilization": model.utilization,
                "traffic_intensity": model.traffic_intensity
            }
        
        # 计算基本性能指标
        rho = model.traffic_intensity
        c = model.num_servers
        utilization = model.utilization
        
        # 简化的M/M/c公式
        if c == 1:
            # M/M/1队列
            avg_queue_length = rho * rho / (1 - rho)
            avg_wait_time = avg_queue_length / arrival_rate
        else:
            # M/M/c队列近似
            avg_wait_time = (rho / (c * service_rate)) * (utilization / (1 - utilization)) * 0.5
            avg_queue_length = arrival_rate * avg_wait_time
        
        avg_response_time = avg_wait_time + avg_service_time
        
        return {
            "arrival_rate": arrival_rate,
            "service_rate": service_rate,
            "num_servers": num_servers,
            "utilization": utilization,
            "traffic_intensity": rho,
            "avg_queue_length": avg_queue_length,
            "avg_wait_time": avg_wait_time,
            "avg_response_time": avg_response_time
        }
    
    def analyze_rag_pipeline(self, arrival_rate: float) -> Dict[str, Any]:
        """分析完整RAG流水线"""
        # 各阶段平均服务时间
        stage_service_times = {
            "embedding": 0.1,
            "retrieval": 0.3,
            "augmentation": 0.015,
            "generation": 2.0,
            "postprocessing": 0.05
        }
        
        stage_servers = {
            "embedding": self.config.num_embedders,
            "retrieval": self.config.num_retrievers,
            "augmentation": self.config.num_augmenters,
            "generation": self.config.num_gpu_slots,
            "postprocessing": self.config.num_postprocessors
        }
        
        pipeline_analysis = {
            "arrival_rate": arrival_rate,
            "stages": {},
            "pipeline_metrics": {}
        }
        
        total_avg_response_time = 0.0
        bottleneck_stage = None
        max_utilization = 0.0
        
        # 分析各阶段
        for stage_name in stage_service_times.keys():
            stage_result = self.analyze_single_stage(
                arrival_rate=arrival_rate,
                avg_service_time=stage_service_times[stage_name],
                num_servers=stage_servers[stage_name]
            )
            
            pipeline_analysis["stages"][stage_name] = stage_result
            
            if "error" not in stage_result:
                total_avg_response_time += stage_result["avg_response_time"]
                
                if stage_result["utilization"] > max_utilization:
                    max_utilization = stage_result["utilization"]
                    bottleneck_stage = stage_name
        
        # 流水线整体指标
        pipeline_analysis["pipeline_metrics"] = {
            "total_avg_response_time": total_avg_response_time,
            "bottleneck_stage": bottleneck_stage,
            "max_utilization": max_utilization,
            "is_stable": max_utilization < 1.0
        }
        
        return pipeline_analysis
    
    def compare_with_simulation(self, theory_results: Dict, 
                              simulation_results: Dict) -> Dict[str, Any]:
        """对比理论与仿真结果"""
        comparison = {
            "metrics_comparison": {},
            "stage_comparison": {}
        }
        
        # 主要指标对比
        if "avg_latency" in simulation_results and "total_avg_response_time" in theory_results["pipeline_metrics"]:
            sim_latency = simulation_results["avg_latency"]
            theory_latency = theory_results["pipeline_metrics"]["total_avg_response_time"]
            
            comparison["metrics_comparison"]["latency"] = {
                "simulation": sim_latency,
                "theory": theory_latency,
                "relative_error": abs(sim_latency - theory_latency) / theory_latency if theory_latency > 0 else 0
            }
        
        # 各阶段对比
        if "stage_metrics" in simulation_results:
            for stage in ["embedding", "retrieval", "augmentation", "generation", "postprocessing"]:
                if (stage in simulation_results["stage_metrics"] and 
                    stage in theory_results["stages"]):
                    
                    sim_wait = simulation_results["stage_metrics"][stage]["avg_wait_time"]
                    theory_wait = theory_results["stages"][stage].get("avg_wait_time", 0)
                    
                    comparison["stage_comparison"][stage] = {
                        "simulation_wait_time": sim_wait,
                        "theory_wait_time": theory_wait,
                        "relative_error": abs(sim_wait - theory_wait) / theory_wait if theory_wait > 0 else 0
                    }
        
        return comparison


if __name__ == "__main__":
    # 测试排队论分析器
    config = SimulationConfig()
    analyzer = QueueingTheoryAnalyzer(config)
    
    # 分析单个阶段
    stage_result = analyzer.analyze_single_stage(
        arrival_rate=0.4,
        avg_service_time=2.0,
        num_servers=1
    )
    print("单阶段分析:")
    for key, value in stage_result.items():
        print(f"  {key}: {value}")
    
    # 分析完整流水线
    print("\n流水线分析:")
    pipeline_result = analyzer.analyze_rag_pipeline(arrival_rate=0.4)
    
    print(f"瓶颈阶段: {pipeline_result['pipeline_metrics']['bottleneck_stage']}")
    print(f"最大利用率: {pipeline_result['pipeline_metrics']['max_utilization']:.3f}")
    print(f"理论总延迟: {pipeline_result['pipeline_metrics']['total_avg_response_time']:.3f}s") 