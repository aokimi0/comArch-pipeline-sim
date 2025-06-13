"""
流水线加速检索增强生成技术 - 源代码包

本包包含RAG流水线仿真的核心实现代码。
"""

__version__ = "1.0.0"
__author__ = "计算机组成原理大作业"

from .utils.config import Config
from .pipeline.rag_pipeline import RAGPipeline
from .simulation.simulator import RAGSimulator

__all__ = ["Config", "RAGPipeline", "RAGSimulator"] 