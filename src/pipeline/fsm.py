"""
有限状态机模块

定义RAG请求在系统中的状态迁移。
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import graphviz


class RequestState(Enum):
    """RAG请求状态枚举"""
    
    # 起始状态
    IDLE = auto()
    
    # 等待状态
    AWAITING_EMBEDDING = auto()
    AWAITING_RETRIEVAL = auto()
    AWAITING_AUGMENTATION = auto()
    AWAITING_GENERATION = auto()
    AWAITING_POSTPROCESSING = auto()
    
    # 处理状态
    EMBEDDING = auto()
    RETRIEVING = auto()
    AUGMENTING = auto()
    GENERATING = auto()
    POSTPROCESSING = auto()
    
    # 终止状态
    COMPLETED = auto()
    FAILED = auto()


class EventType(Enum):
    """状态转换事件类型"""
    
    REQUEST_RECEIVED = auto()
    RESOURCE_ACQUIRED = auto()
    PROCESSING_COMPLETE = auto()
    PROCESSING_FAILED = auto()


@dataclass
class StateTransition:
    """状态转换定义"""
    
    from_state: RequestState
    event: EventType
    to_state: RequestState
    action: Optional[str] = None


class RAGRequestFSM:
    """RAG请求有限状态机"""
    
    def __init__(self):
        self.current_state = RequestState.IDLE
        self.transitions = self._build_transitions()
        self.state_history: List[RequestState] = [RequestState.IDLE]
        self.event_history: List[EventType] = []
    
    def _build_transitions(self) -> Dict[tuple, RequestState]:
        """构建状态转换表"""
        transitions = {
            # 请求进入系统
            (RequestState.IDLE, EventType.REQUEST_RECEIVED): 
                RequestState.AWAITING_EMBEDDING,
            
            # 嵌入阶段
            (RequestState.AWAITING_EMBEDDING, EventType.RESOURCE_ACQUIRED): 
                RequestState.EMBEDDING,
            (RequestState.EMBEDDING, EventType.PROCESSING_COMPLETE): 
                RequestState.AWAITING_RETRIEVAL,
            (RequestState.EMBEDDING, EventType.PROCESSING_FAILED): 
                RequestState.FAILED,
            
            # 检索阶段
            (RequestState.AWAITING_RETRIEVAL, EventType.RESOURCE_ACQUIRED): 
                RequestState.RETRIEVING,
            (RequestState.RETRIEVING, EventType.PROCESSING_COMPLETE): 
                RequestState.AWAITING_AUGMENTATION,
            (RequestState.RETRIEVING, EventType.PROCESSING_FAILED): 
                RequestState.FAILED,
            
            # 增强阶段
            (RequestState.AWAITING_AUGMENTATION, EventType.RESOURCE_ACQUIRED): 
                RequestState.AUGMENTING,
            (RequestState.AUGMENTING, EventType.PROCESSING_COMPLETE): 
                RequestState.AWAITING_GENERATION,
            (RequestState.AUGMENTING, EventType.PROCESSING_FAILED): 
                RequestState.FAILED,
            
            # 生成阶段
            (RequestState.AWAITING_GENERATION, EventType.RESOURCE_ACQUIRED): 
                RequestState.GENERATING,
            (RequestState.GENERATING, EventType.PROCESSING_COMPLETE): 
                RequestState.AWAITING_POSTPROCESSING,
            (RequestState.GENERATING, EventType.PROCESSING_FAILED): 
                RequestState.FAILED,
            
            # 后处理阶段
            (RequestState.AWAITING_POSTPROCESSING, EventType.RESOURCE_ACQUIRED): 
                RequestState.POSTPROCESSING,
            (RequestState.POSTPROCESSING, EventType.PROCESSING_COMPLETE): 
                RequestState.COMPLETED,
            (RequestState.POSTPROCESSING, EventType.PROCESSING_FAILED): 
                RequestState.FAILED,
        }
        return transitions
    
    def transition(self, event: EventType) -> bool:
        """执行状态转换
        
        Args:
            event: 触发事件
            
        Returns:
            转换是否成功
        """
        key = (self.current_state, event)
        if key in self.transitions:
            old_state = self.current_state
            self.current_state = self.transitions[key]
            self.state_history.append(self.current_state)
            self.event_history.append(event)
            return True
        return False
    
    def is_terminal_state(self) -> bool:
        """检查是否为终止状态"""
        return self.current_state in {RequestState.COMPLETED, RequestState.FAILED}
    
    def is_waiting_state(self) -> bool:
        """检查是否为等待状态"""
        waiting_states = {
            RequestState.AWAITING_EMBEDDING,
            RequestState.AWAITING_RETRIEVAL,
            RequestState.AWAITING_AUGMENTATION,
            RequestState.AWAITING_GENERATION,
            RequestState.AWAITING_POSTPROCESSING
        }
        return self.current_state in waiting_states
    
    def is_processing_state(self) -> bool:
        """检查是否为处理状态"""
        processing_states = {
            RequestState.EMBEDDING,
            RequestState.RETRIEVING,
            RequestState.AUGMENTING,
            RequestState.GENERATING,
            RequestState.POSTPROCESSING
        }
        return self.current_state in processing_states
    
    def get_stage_from_state(self) -> Optional[str]:
        """从状态获取对应的流水线阶段"""
        state_to_stage = {
            RequestState.AWAITING_EMBEDDING: "embedding",
            RequestState.EMBEDDING: "embedding",
            RequestState.AWAITING_RETRIEVAL: "retrieval",
            RequestState.RETRIEVING: "retrieval",
            RequestState.AWAITING_AUGMENTATION: "augmentation",
            RequestState.AUGMENTING: "augmentation",
            RequestState.AWAITING_GENERATION: "generation",
            RequestState.GENERATING: "generation",
            RequestState.AWAITING_POSTPROCESSING: "postprocessing",
            RequestState.POSTPROCESSING: "postprocessing"
        }
        return state_to_stage.get(self.current_state)
    
    def reset(self):
        """重置状态机"""
        self.current_state = RequestState.IDLE
        self.state_history = [RequestState.IDLE]
        self.event_history = []
    
    def get_transition_table(self) -> List[Dict]:
        """获取状态转换表（用于文档化）"""
        table = []
        for (from_state, event), to_state in self.transitions.items():
            table.append({
                "当前状态": from_state.name,
                "触发事件": event.name,
                "下一状态": to_state.name
            })
        return table


def create_fsm_diagram(output_path: str = "results/plots/rag_fsm_diagram"):
    """创建FSM状态转移图
    
    Args:
        output_path: 输出文件路径（不含扩展名）
    """
    dot = graphviz.Digraph(comment='RAG Request FSM')
    dot.attr(rankdir='TB', size='12,8')
    
    # 添加状态节点
    waiting_states = {
        RequestState.AWAITING_EMBEDDING,
        RequestState.AWAITING_RETRIEVAL,
        RequestState.AWAITING_AUGMENTATION,
        RequestState.AWAITING_GENERATION,
        RequestState.AWAITING_POSTPROCESSING
    }
    
    processing_states = {
        RequestState.EMBEDDING,
        RequestState.RETRIEVING,
        RequestState.AUGMENTING,
        RequestState.GENERATING,
        RequestState.POSTPROCESSING
    }
    
    for state in RequestState:
        if state == RequestState.IDLE:
            dot.node(state.name, state.name.replace('_', ' '), shape='circle', style='filled', fillcolor='lightgray')
        elif state in waiting_states:
            dot.node(state.name, state.name.replace('_', ' '), shape='box', style='filled', fillcolor='lightyellow')
        elif state in processing_states:
            dot.node(state.name, state.name.replace('_', ' '), shape='ellipse', style='filled', fillcolor='lightgreen')
        elif state == RequestState.COMPLETED:
            dot.node(state.name, state.name.replace('_', ' '), shape='doublecircle', style='filled', fillcolor='lightblue')
        elif state == RequestState.FAILED:
            dot.node(state.name, state.name.replace('_', ' '), shape='doublecircle', style='filled', fillcolor='lightcoral')
    
    # 添加转换边
    fsm = RAGRequestFSM()
    for (from_state, event), to_state in fsm.transitions.items():
        edge_label = event.name.replace('_', ' ')
        dot.edge(from_state.name, to_state.name, label=edge_label)
    
    # 保存图形
    try:
        dot.render(output_path, format='png', cleanup=True)
        print(f"FSM状态转移图已保存至: {output_path}.png")
    except Exception as e:
        print(f"生成FSM图形时出错: {e}")
    
    return dot


if __name__ == "__main__":
    # 测试FSM
    fsm = RAGRequestFSM()
    print(f"初始状态: {fsm.current_state}")
    
    # 模拟请求处理流程
    events = [
        EventType.REQUEST_RECEIVED,
        EventType.RESOURCE_ACQUIRED,  # 进入嵌入
        EventType.PROCESSING_COMPLETE,  # 嵌入完成
        EventType.RESOURCE_ACQUIRED,  # 进入检索
        EventType.PROCESSING_COMPLETE,  # 检索完成
        EventType.RESOURCE_ACQUIRED,  # 进入增强
        EventType.PROCESSING_COMPLETE,  # 增强完成
        EventType.RESOURCE_ACQUIRED,  # 进入生成
        EventType.PROCESSING_COMPLETE,  # 生成完成
        EventType.RESOURCE_ACQUIRED,  # 进入后处理
        EventType.PROCESSING_COMPLETE,  # 后处理完成
    ]
    
    for event in events:
        success = fsm.transition(event)
        print(f"事件: {event.name}, 新状态: {fsm.current_state.name}, 成功: {success}")
    
    print(f"最终状态: {fsm.current_state}")
    print(f"是否完成: {fsm.is_terminal_state()}")
    
    # 生成状态转移图
    import os
    os.makedirs("results/plots", exist_ok=True)
    create_fsm_diagram() 