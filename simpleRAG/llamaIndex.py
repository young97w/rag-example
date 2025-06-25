"""
生产级RAG问答系统
支持基于本地文档的智能问答，使用DeepSeek LLM和中文embedding模型
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """RAG系统配置类"""
    deepseek_model: str = "deepseek-reasoner"
    embedding_model: str = "BAAI/bge-small-zh"
    api_key_env: str = "DEEPSEEK_API_KEY"
    chunk_size: int = 1024
    chunk_overlap: int = 20
    similarity_top_k: int = 3


class RAGSystemError(Exception):
    """RAG系统自定义异常"""
    pass


class ModelInitializationError(RAGSystemError):
    """模型初始化异常"""
    pass


class DocumentLoadError(RAGSystemError):
    """文档加载异常"""
    pass


class QueryError(RAGSystemError):
    """查询异常"""
    pass


class RAGSystem:
    """生产级RAG问答系统"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        初始化RAG系统
        
        Args:
            config: RAG系统配置，如果为None则使用默认配置
        """
        self.config = config or RAGConfig()
        self.llm = None
        self.embed_model = None
        self.index = None
        self.query_engine = None
        self._is_initialized = False
        
        logger.info("RAG系统初始化开始")
        self._validate_environment()
    
    def _validate_environment(self) -> None:
        """验证环境变量和依赖"""
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise ModelInitializationError(
                f"环境变量 {self.config.api_key_env} 未设置"
            )
        
        logger.info(f"API密钥验证通过: {api_key[:8]}...")
    
    def _initialize_models(self) -> None:
        """初始化LLM和embedding模型"""
        try:
            logger.info("正在初始化DeepSeek LLM...")
            self.llm = DeepSeek(
                model=self.config.deepseek_model,
                api_key=os.getenv(self.config.api_key_env)
            )
            
            logger.info("正在初始化embedding模型...")
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.config.embedding_model
            )
            
            # 设置全局配置
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = self.config.chunk_size
            Settings.chunk_overlap = self.config.chunk_overlap
            
            logger.info("模型初始化完成")
            
        except Exception as e:
            raise ModelInitializationError(f"模型初始化失败: {str(e)}") from e
    
    def load_documents(self, file_paths: List[str]) -> None:
        """
        加载文档并构建索引
        
        Args:
            file_paths: 文档文件路径列表
        """
        if not self.llm or not self.embed_model:
            self._initialize_models()
        
        try:
            # 验证文件路径
            validated_paths = self._validate_file_paths(file_paths)
            
            logger.info(f"正在加载 {len(validated_paths)} 个文档...")
            documents = SimpleDirectoryReader(
                input_files=validated_paths
            ).load_data()
            
            if not documents:
                raise DocumentLoadError("未能加载任何文档")
            
            logger.info(f"成功加载 {len(documents)} 个文档段落")
            
            # 构建向量索引
            logger.info("正在构建向量索引...")
            self.index = VectorStoreIndex.from_documents(documents)
            
            # 创建查询引擎
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=self.config.similarity_top_k
            )
            
            self._is_initialized = True
            logger.info("文档索引构建完成")
            
        except Exception as e:
            raise DocumentLoadError(f"文档加载失败: {str(e)}") from e
    
    def _validate_file_paths(self, file_paths: List[str]) -> List[str]:
        """验证文件路径"""
        validated_paths = []
        for path in file_paths:
            file_path = Path(path)
            if not file_path.exists():
                logger.warning(f"文件不存在: {path}")
                continue
            if not file_path.is_file():
                logger.warning(f"路径不是文件: {path}")
                continue
            validated_paths.append(str(file_path))
        
        if not validated_paths:
            raise DocumentLoadError("没有有效的文件路径")
        
        return validated_paths
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        执行问答查询
        
        Args:
            question: 问题文本
            **kwargs: 额外的查询参数
            
        Returns:
            包含答案和元数据的字典
        """
        if not self._is_initialized:
            raise QueryError("系统未初始化，请先加载文档")
        
        if not question or not question.strip():
            raise QueryError("问题不能为空")
        
        try:
            logger.info(f"正在处理问题: {question[:50]}...")
            
            response = self.query_engine.query(question)
            
            result = {
                "question": question,
                "answer": str(response),
                "metadata": {
                    "source_nodes": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0,
                    "model": self.config.deepseek_model,
                    "embedding_model": self.config.embedding_model
                }
            }
            
            logger.info("查询处理完成")
            return result
            
        except Exception as e:
            raise QueryError(f"查询处理失败: {str(e)}") from e
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        批量查询
        
        Args:
            questions: 问题列表
            
        Returns:
            答案结果列表
        """
        results = []
        for i, question in enumerate(questions, 1):
            try:
                logger.info(f"处理第 {i}/{len(questions)} 个问题")
                result = self.query(question)
                results.append(result)
            except Exception as e:
                logger.error(f"第 {i} 个问题处理失败: {str(e)}")
                results.append({
                    "question": question,
                    "answer": None,
                    "error": str(e)
                })
        
        return results
    
    @contextmanager
    def error_handling(self):
        """错误处理上下文管理器"""
        try:
            yield
        except RAGSystemError as e:
            logger.error(f"RAG系统错误: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"未知错误: {str(e)}")
            raise RAGSystemError(f"系统运行出错: {str(e)}") from e
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "initialized": self._is_initialized,
            "config": {
                "deepseek_model": self.config.deepseek_model,
                "embedding_model": self.config.embedding_model,
                "chunk_size": self.config.chunk_size,
                "similarity_top_k": self.config.similarity_top_k
            },
            "models_loaded": self.llm is not None and self.embed_model is not None,
            "index_built": self.index is not None
        }


def create_rag_system(config: Optional[RAGConfig] = None) -> RAGSystem:
    """
    工厂函数：创建RAG系统实例
    
    Args:
        config: 可选的配置对象
        
    Returns:
        RAGSystem实例
    """
    return RAGSystem(config)


def main():
    """主函数示例"""
    try:
        # 创建配置
        config = RAGConfig(
            deepseek_model="deepseek-reasoner",
            embedding_model="BAAI/bge-small-zh",
            similarity_top_k=3
        )
        
        # 创建RAG系统
        rag_system = create_rag_system(config)
        
        with rag_system.error_handling():
            # 加载文档
            file_paths = ["data/2024年中国财政政策执行情况报告.txt"]
            rag_system.load_documents(file_paths)
            
            # 单个查询
            result = rag_system.query("2024中国的税收总额多少?")
            print(f"问题: {result['question']}")
            print(f"答案: {result['answer']}")
            print(f"元数据: {result['metadata']}")
            
            # 批量查询示例
            questions = [
                "2024年财政收入的主要来源是什么？",
                "税收政策有哪些重要调整？",
                "财政支出的重点领域有哪些？"
            ]
            
            batch_results = rag_system.batch_query(questions)
            for result in batch_results:
                if result.get("error"):
                    print(f"错误: {result['error']}")
                else:
                    print(f"Q: {result['question']}")
                    print(f"A: {result['answer'][:100]}...")
                    print("-" * 50)
            
            # 系统信息
            info = rag_system.get_system_info()
            print(f"系统信息: {info}")
    
    except RAGSystemError as e:
        logger.error(f"RAG系统错误: {str(e)}")
    except Exception as e:
        logger.error(f"程序运行错误: {str(e)}")


if __name__ == "__main__":
    main()