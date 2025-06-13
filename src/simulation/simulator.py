"""
SimPy仿真引擎

实现RAG流水线的离散事件仿真。
"""

import simpy
import random
import pandas as pd
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from tqdm import tqdm

from ..utils.config import SimulationConfig, ExperimentConfig
from ..pipeline.rag_pipeline import RAGPipeline, RequestMetrics


class WorkloadGenerator:
    """工作负载生成器"""
    
    def __init__(self, env: simpy.Environment, pipeline: RAGPipeline, 
                 arrival_rate: float, simulation_time: float):
        self.env = env
        self.pipeline = pipeline
        self.arrival_rate = arrival_rate
        self.simulation_time = simulation_time
        self.request_count = 0
        
    def generate_workload(self):
        """生成请求工作负载"""
        while self.env.now < self.simulation_time:
            # 生成新请求
            self.pipeline.process_request(self.request_count)
            self.request_count += 1
            
            # 泊松到达过程
            inter_arrival_time = random.expovariate(self.arrival_rate)
            yield self.env.timeout(inter_arrival_time)


class RAGSimulator:
    """RAG流水线仿真器"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.console = Console()
        
    def run_single_simulation(self, arrival_rate: float, 
                            simulation_time: float = None,
                            warm_up_time: float = 0.0,
                            verbose: bool = True) -> Dict[str, Any]:
        """运行单次仿真
        
        Args:
            arrival_rate: 请求到达率
            simulation_time: 仿真时长
            warm_up_time: 预热时间
            verbose: 是否显示详细信息
            
        Returns:
            仿真结果字典
        """
        if simulation_time is None:
            simulation_time = self.config.simulation_time
            
        # 设置随机种子
        random.seed(self.config.random_seed)
        
        # 创建仿真环境
        env = simpy.Environment()
        
        # 创建流水线
        sim_config = self.config
        sim_config.arrival_rate = arrival_rate
        pipeline = RAGPipeline(env, sim_config)
        
        # 创建工作负载生成器
        workload_gen = WorkloadGenerator(env, pipeline, arrival_rate, simulation_time)
        env.process(workload_gen.generate_workload())
        
        if verbose:
            self.console.print(f"[bold blue]开始仿真[/bold blue] - 到达率: {arrival_rate:.2f} req/s, 时长: {simulation_time:.0f}s")
        
        # 运行仿真
        env.run(until=simulation_time)
        
        # 等待所有请求完成（给予额外时间）
        remaining_requests = len(pipeline.active_requests)
        if remaining_requests > 0 and verbose:
            self.console.print(f"等待 {remaining_requests} 个未完成请求...")
            env.run(until=simulation_time + 50.0)  # 额外50秒等待时间
        
        # 过滤预热期数据
        completed_requests = [
            req for req in pipeline.completed_requests 
            if req.completion_time >= warm_up_time
        ]
        
        if not completed_requests:
            return {
                "error": "仿真期间无请求完成",
                "total_requests": len(pipeline.completed_requests),
                "arrival_rate": arrival_rate
            }
        
        # 计算性能指标
        result = self._calculate_metrics(completed_requests, arrival_rate, \
                                       simulation_time - warm_up_time)
        
        # 添加队列长度和利用率数据
        result["queue_data"] = pipeline.get_queue_length_data()
        result["utilization_data"] = pipeline.get_utilization_data()
        result["raw_requests"] = completed_requests
        
        # 添加各阶段的平均队列长度和平均利用率
        average_stage_metrics = pipeline.get_average_stage_metrics(warm_up_time)
        for stage_name, metrics in average_stage_metrics.items():
            # 将这些指标直接添加到 result["stage_metrics"] 中
            if stage_name in result["stage_metrics"]:
                result["stage_metrics"][stage_name]["avg_queue_length"] = metrics["avg_queue_length"]
                result["stage_metrics"][stage_name]["avg_utilization"] = metrics["avg_utilization"]
            else:
                result["stage_metrics"][stage_name] = metrics
        
        if verbose:
            self._print_results(result)
            
        return result
    
    def run_batch_experiments(self, experiment_config: ExperimentConfig) -> pd.DataFrame:
        """运行批量实验
        
        Args:
            experiment_config: 实验配置
            
        Returns:
            实验结果DataFrame
        """
        results = []
        total_runs = len(experiment_config.arrival_rates) * experiment_config.num_runs
        
        self.console.print(f"[bold green]开始批量实验[/bold green] - 总计 {total_runs} 次运行")
        
        with Progress() as progress:
            task = progress.add_task("实验进度", total=total_runs)
            
            for arrival_rate in experiment_config.arrival_rates:
                for run_id in range(experiment_config.num_runs):
                    # 更新随机种子以确保每次运行的独立性
                    self.config.random_seed = self.config.random_seed + run_id
                    
                    result = self.run_single_simulation(
                        arrival_rate=arrival_rate,
                        warm_up_time=experiment_config.warm_up_time,
                        verbose=False
                    )
                    
                    if "error" not in result:
                        result["arrival_rate"] = arrival_rate
                        result["run_id"] = run_id
                        results.append(result)
                    
                    progress.advance(task)
        
        if not results:
            self.console.print("[bold red]错误: 所有仿真运行都失败了[/bold red]")
            return pd.DataFrame()
        
        # 扁平化stage_metrics
        processed_results = []
        for res in results:
            flat_res = {k: v for k, v in res.items() if k != "stage_metrics"}
            for stage_name, stage_metrics in res["stage_metrics"].items():
                for metric_name, value in stage_metrics.items():
                    flat_res[f"stage_{stage_name}_{metric_name}"] = value
            processed_results.append(flat_res)

        df = pd.DataFrame(processed_results)
        
        # 计算聚合统计
        self._print_batch_summary(df, experiment_config)
        
        return df
    
    def _calculate_metrics(self, requests: List[RequestMetrics], \
                          arrival_rate: float, effective_time: float) -> Dict[str, Any]:
        """计算性能指标"""
        if not requests:
            return {}
            
        # 基本统计
        latencies = [req.total_latency for req in requests]
        wait_times = [req.total_wait_time for req in requests]
        
        # 各阶段等待时间
        stage_waits = {}
        stage_processing = {}
        
        for stage in ['embedding', 'retrieval', 'augmentation', 'generation', 'postprocessing']:
            stage_waits[stage] = [getattr(req, f"{stage}_wait_time") for req in requests]
            stage_processing[stage] = [getattr(req, f"{stage}_processing_time") for req in requests]
        
        metrics = {
            # 基本指标
            "completed_requests": len(requests),
            "avg_latency": sum(latencies) / len(latencies),
            "max_latency": max(latencies),
            "min_latency": min(latencies),
            "latency_p95": pd.Series(latencies).quantile(0.95),
            "latency_p99": pd.Series(latencies).quantile(0.99),
            
            "avg_wait_time": sum(wait_times) / len(wait_times),
            "throughput": len(requests) / effective_time,
            "effective_arrival_rate": arrival_rate,
            
            # 各阶段指标 (这里只包含处理和等待时间，平均队列长度和利用率由 pipeline 传入)
            "stage_metrics": {}
        }
        
        for stage in stage_waits.keys():
            metrics["stage_metrics"][stage] = {
                "avg_wait_time": sum(stage_waits[stage]) / len(stage_waits[stage]),
                "max_wait_time": max(stage_waits[stage]),
                "avg_processing_time": sum(stage_processing[stage]) / len(stage_processing[stage]),
                "max_processing_time": max(stage_processing[stage])
            }
        
        return metrics
    
    def _print_results(self, result: Dict[str, Any]):
        """打印仿真结果"""
        if "error" in result:
            self.console.print(f"[bold red]仿真失败: {result['error']}[/bold red]")
            return
            
        table = Table(title="仿真结果摘要")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="yellow")
        
        table.add_row("完成请求数", str(result["completed_requests"]))
        table.add_row("平均延迟", f"{result['avg_latency']:.3f}s")
        table.add_row("P95延迟", f"{result['latency_p95']:.3f}s")
        table.add_row("P99延迟", f"{result['latency_p99']:.3f}s")
        table.add_row("平均等待时间", f"{result['avg_wait_time']:.3f}s")
        table.add_row("系统吞吐量", f"{result['throughput']:.3f} req/s")
        
        self.console.print(table)
        
        # 各阶段性能表
        stage_table = Table(title="各阶段性能")
        stage_table.add_column("阶段", style="cyan")
        stage_table.add_column("平均等待时间", style="yellow")
        stage_table.add_column("最大等待时间", style="red")
        stage_table.add_column("平均处理时间", style="green")
        stage_table.add_column("平均队列长度", style="magenta")
        stage_table.add_column("平均利用率", style="blue")
        
        for stage, metrics in result["stage_metrics"].items():
            avg_q_len = metrics.get("avg_queue_length", float('nan'))
            avg_util = metrics.get("avg_utilization", float('nan'))
            stage_table.add_row(
                stage.capitalize(),
                f"{metrics['avg_wait_time']:.3f}s",
                f"{metrics['max_wait_time']:.3f}s",
                f"{metrics['avg_processing_time']:.3f}s",
                f"{avg_q_len:.3f}",
                f"{avg_util:.1%}"
            )
        
        self.console.print(stage_table)
    
    def _print_batch_summary(self, df: pd.DataFrame, config: ExperimentConfig):
        """打印批量实验摘要"""
        self.console.print("\n[bold green]批量实验完成![/bold green]")
        
        # 按到达率分组统计
        summary_cols = {
            'avg_latency': ['mean', 'std'],
            'throughput': ['mean', 'std'],
            'completed_requests': 'mean',
            'stage_generation_avg_queue_length': ['mean', 'std'],
            'stage_generation_avg_utilization': ['mean', 'std'],
        }

        # 动态添加所有阶段的平均队列长度和利用率到 summary_cols
        for stage_name in ['embedding', 'retrieval', 'augmentation', 'generation', 'postprocessing']:
            summary_cols[f'stage_{stage_name}_avg_queue_length'] = ['mean', 'std']
            summary_cols[f'stage_{stage_name}_avg_utilization'] = ['mean', 'std']

        summary = df.groupby('arrival_rate').agg(summary_cols).round(4)
        
        self.console.print(summary)
    
    def export_results(self, results: pd.DataFrame, output_path: str):
        """导出实验结果"""
        results.to_csv(output_path, index=False)
        self.console.print(f"[green]实验结果已保存至: {output_path}[/green]")


if __name__ == "__main__":
    # 测试仿真器
    config = SimulationConfig()
    config.simulation_time = 100.0
    
    simulator = RAGSimulator(config)
    
    # 运行单次仿真
    result = simulator.run_single_simulation(arrival_rate=0.3)
    
    print("\n仿真完成!") 