"""
RAG流水线核心实现

定义五阶段RAG流水线的架构和资源管理。
"""

import simpy
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from ..utils.config import SimulationConfig
from .fsm import RAGRequestFSM, EventType, RequestState


@dataclass
class PipelineStage:
    """流水线阶段定义"""
    
    name: str
    resource: simpy.Resource
    processing_time_func: callable
    description: str = ""
    
    def __post_init__(self):
        self.requests_processed = 0
        self.total_processing_time = 0.0
        self.queue_length_history = []


@dataclass 
class RequestMetrics:
    """请求性能指标"""
    
    request_id: int
    arrival_time: float
    completion_time: float = 0.0
    
    # 各阶段时间戳
    embed_start: float = 0.0
    embed_end: float = 0.0
    retrieve_start: float = 0.0
    retrieve_end: float = 0.0
    augment_start: float = 0.0
    augment_end: float = 0.0
    generate_start: float = 0.0
    generate_end: float = 0.0
    postprocess_start: float = 0.0
    postprocess_end: float = 0.0
    
    # 等待时间
    embed_wait_time: float = 0.0
    retrieve_wait_time: float = 0.0
    augment_wait_time: float = 0.0
    generate_wait_time: float = 0.0
    postprocess_wait_time: float = 0.0
    
    # 处理时间
    embed_processing_time: float = 0.0
    retrieve_processing_time: float = 0.0
    augment_processing_time: float = 0.0
    generate_processing_time: float = 0.0
    postprocess_processing_time: float = 0.0
    
    # 状态历史
    fsm_states: List[RequestState] = field(default_factory=list)
    
    @property
    def total_latency(self) -> float:
        """总端到端延迟"""
        if self.completion_time > 0:
            return self.completion_time - self.arrival_time
        return 0.0
    
    @property
    def total_wait_time(self) -> float:
        """总等待时间"""
        return (self.embed_wait_time + self.retrieve_wait_time + 
                self.augment_wait_time + self.generate_wait_time + 
                self.postprocess_wait_time)
    
    @property
    def total_processing_time(self) -> float:
        """总处理时间"""
        return (self.embed_processing_time + self.retrieve_processing_time + 
                self.augment_processing_time + self.generate_processing_time + 
                self.postprocess_processing_time)


class RAGPipeline:
    """RAG流水线主类"""
    
    def __init__(self, env: simpy.Environment, config: SimulationConfig):
        self.env = env
        self.config = config
        
        # 创建各阶段资源
        self.stages = self._create_stages()
        
        # 性能指标收集
        self.completed_requests: List[RequestMetrics] = []
        self.active_requests: Dict[int, RequestMetrics] = {}
        
        # 系统监控
        self.system_metrics = {
            'timestamp': [],
            'queue_lengths': {stage.name: [] for stage in self.stages.values()},
            'resource_utilization': {stage.name: [] for stage in self.stages.values()},
            'active_requests': []
        }
        
        # 启动监控进程
        self.monitor_process = env.process(self._monitor_system())
    
    def _create_stages(self) -> Dict[str, PipelineStage]:
        """创建流水线各阶段"""
        stages = {}
        
        # 阶段1：嵌入
        stages['embedding'] = PipelineStage(
            name='embedding',
            resource=simpy.Resource(self.env, capacity=self.config.num_embedders),
            processing_time_func=self.config.t_embed,
            description='查询嵌入阶段'
        )
        
        # 阶段2：检索
        stages['retrieval'] = PipelineStage(
            name='retrieval',
            resource=simpy.Resource(self.env, capacity=self.config.num_retrievers),
            processing_time_func=self.config.t_retrieve,
            description='上下文检索阶段'
        )
        
        # 阶段3：增强
        stages['augmentation'] = PipelineStage(
            name='augmentation',
            resource=simpy.Resource(self.env, capacity=self.config.num_augmenters),
            processing_time_func=self.config.t_augment,
            description='提示词增强阶段'
        )
        
        # 阶段4：生成（瓶颈阶段）
        stages['generation'] = PipelineStage(
            name='generation',
            resource=simpy.Resource(self.env, capacity=self.config.num_gpu_slots),
            processing_time_func=self.config.t_generate,
            description='LLM生成阶段'
        )
        
        # 阶段5：后处理
        stages['postprocessing'] = PipelineStage(
            name='postprocessing',
            resource=simpy.Resource(self.env, capacity=self.config.num_postprocessors),
            processing_time_func=self.config.t_postprocess,
            description='后处理阶段'
        )
        
        return stages
    
    def process_request(self, request_id: int) -> simpy.Process:
        """处理单个RAG请求
        
        Args:
            request_id: 请求唯一标识
            
        Returns:
            SimPy进程对象
        """
        return self.env.process(self._request_lifecycle(request_id))
    
    def _request_lifecycle(self, request_id: int):
        """请求完整生命周期处理"""
        # 初始化请求指标和FSM
        metrics = RequestMetrics(
            request_id=request_id,
            arrival_time=self.env.now
        )
        fsm = RAGRequestFSM()
        
        self.active_requests[request_id] = metrics
        
        try:
            # FSM: 请求到达
            fsm.transition(EventType.REQUEST_RECEIVED)
            metrics.fsm_states.append(fsm.current_state)
            
            # 阶段1：嵌入
            yield from self._process_stage('embedding', metrics, fsm)
            
            # 阶段2：检索
            yield from self._process_stage('retrieval', metrics, fsm)
            
            # 阶段3：增强
            yield from self._process_stage('augmentation', metrics, fsm)
            
            # 阶段4：生成
            yield from self._process_stage('generation', metrics, fsm)
            
            # 阶段5：后处理
            yield from self._process_stage('postprocessing', metrics, fsm)
            
            # 请求完成
            metrics.completion_time = self.env.now
            metrics.fsm_states.append(RequestState.COMPLETED)
            
        except Exception as e:
            # 请求失败
            metrics.completion_time = self.env.now
            metrics.fsm_states.append(RequestState.FAILED)
            print(f"请求 {request_id} 处理失败: {e}")
        
        finally:
            # 移动到完成队列
            self.completed_requests.append(metrics)
            if request_id in self.active_requests:
                del self.active_requests[request_id]
    
    def _process_stage(self, stage_name: str, metrics: RequestMetrics, fsm: RAGRequestFSM):
        """处理单个流水线阶段
        
        Args:
            stage_name: 阶段名称
            metrics: 请求指标对象
            fsm: 有限状态机实例
        """
        stage = self.stages[stage_name]
        
        # 等待资源
        wait_start = self.env.now
        with stage.resource.request() as request:
            yield request
            
            # FSM: 获取资源
            fsm.transition(EventType.RESOURCE_ACQUIRED)
            metrics.fsm_states.append(fsm.current_state)
            
            wait_end = self.env.now
            wait_time = wait_end - wait_start
            
            # 记录等待时间
            setattr(metrics, f"{stage_name}_wait_time", wait_time)
            setattr(metrics, f"{stage_name}_start", wait_end)
            
            # 处理请求
            processing_time = stage.processing_time_func()
            yield self.env.timeout(processing_time)
            
            process_end = self.env.now
            setattr(metrics, f"{stage_name}_end", process_end)
            setattr(metrics, f"{stage_name}_processing_time", processing_time)
            
            # 更新阶段统计
            stage.requests_processed += 1
            stage.total_processing_time += processing_time
            
            # FSM: 处理完成
            fsm.transition(EventType.PROCESSING_COMPLETE)
            metrics.fsm_states.append(fsm.current_state)
    
    def _monitor_system(self):
        """系统性能监控进程"""
        while True:
            # 记录当前时间戳
            self.system_metrics['timestamp'].append(self.env.now)
            
            # 记录各阶段队列长度和利用率
            for stage_name, stage in self.stages.items():
                current_queue_length = len(stage.resource.queue)
                self.system_metrics['queue_lengths'][stage_name].append(current_queue_length)
                
                # 利用率 = 当前正在使用的资源数量 / 总资源容量
                current_utilization = len(stage.resource.users) / stage.resource.capacity
                self.system_metrics['resource_utilization'][stage_name].append(current_utilization)
            
            # 记录活跃请求数
            self.system_metrics['active_requests'].append(len(self.active_requests))
            
            # 等待监控间隔
            yield self.env.timeout(1.0)  # 每秒监控一次
    
    def get_average_stage_metrics(self, warm_up_time: float) -> Dict[str, Any]:
        """
        计算各阶段的平均队列长度和平均利用率，考虑预热时间。
        
        Args:
            warm_up_time: 预热时间，此时间之前的数据将被忽略。
            
        Returns:
            包含各阶段平均队列长度和平均利用率的字典。
        """
        avg_stage_metrics = {}
        timestamps = self.system_metrics['timestamp']
        
        # 找到预热时间后的起始索引
        start_index = next((i for i, ts in enumerate(timestamps) if ts >= warm_up_time), 0)
        
        # 确保有足够的数据点在预热时间之后
        if start_index >= len(timestamps) - 1:
            print(f"警告: 预热时间 {warm_up_time}s 过长，或仿真时间过短，无法计算平均阶段指标。")
            # 返回默认值或空字典，取决于如何处理此边界情况
            for stage_name in self.stages.keys():
                avg_stage_metrics[stage_name] = {
                    "avg_queue_length": 0.0,
                    "avg_utilization": 0.0
                }
            return avg_stage_metrics
        
        for stage_name in self.stages.keys():
            queue_history_after_warmup = self.system_metrics['queue_lengths'][stage_name][start_index:]
            utilization_history_after_warmup = self.system_metrics['resource_utilization'][stage_name][start_index:]
            
            avg_stage_metrics[stage_name] = {
                "avg_queue_length": sum(queue_history_after_warmup) / len(queue_history_after_warmup) if queue_history_after_warmup else 0.0,
                "avg_utilization": sum(utilization_history_after_warmup) / len(utilization_history_after_warmup) if utilization_history_after_warmup else 0.0
            }
        return avg_stage_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要统计"""
        if not self.completed_requests:
            return {"error": "暂无完成的请求"}
        
        latencies = [req.total_latency for req in self.completed_requests]
        wait_times = [req.total_wait_time for req in self.completed_requests]
        processing_times = [req.total_processing_time for req in self.completed_requests]
        
        summary = {
            "总请求数": len(self.completed_requests),
            "平均端到端延迟": sum(latencies) / len(latencies),
            "平均等待时间": sum(wait_times) / len(wait_times),
            "平均处理时间": sum(processing_times) / len(processing_times),
            "最大延迟": max(latencies),
            "最小延迟": min(latencies),
            "吞吐量": len(self.completed_requests) / max(req.completion_time for req in self.completed_requests),
            
            # 各阶段性能
            "阶段性能": {}
        }
        
        # 计算各阶段的平均等待时间
        for stage_name in self.stages.keys():
            stage_waits = [getattr(req, f"{stage_name}_wait_time") for req in self.completed_requests]
            stage_processing = [getattr(req, f"{stage_name}_processing_time") for req in self.completed_requests]
            
            summary["阶段性能"][stage_name] = {
                "平均等待时间": sum(stage_waits) / len(stage_waits),
                "平均处理时间": sum(stage_processing) / len(stage_processing),
                "总处理请求数": self.stages[stage_name].requests_processed
            }
        
        return summary
    
    def get_queue_length_data(self) -> Dict[str, List]:
        """获取队列长度历史数据"""
        return {
            "时间戳": self.system_metrics['timestamp'],
            "队列长度": self.system_metrics['queue_lengths']
        }
    
    def get_utilization_data(self) -> Dict[str, List]:
        """获取资源利用率历史数据"""
        return {
            "时间戳": self.system_metrics['timestamp'],
            "资源利用率": self.system_metrics['resource_utilization']
        } 