# 导入相关的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # 需要pip install llama-index-embeddings-huggingface
from llama_index.llms.deepseek import DeepSeek  # 需要pip install llama-index-llms-deepseek

from llama_index.core import Settings # 可以看看有哪些Setting
# https://docs.llamaindex.ai/en/stable/examples/llm/deepseek/
# Settings.llm = DeepSeek(model="deepseek-chat")
# Settings.embed_model = HuggingFaceEmbedding(model="BAAI/bge-small-zh")

# 加载环境变量
from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()

# 创建 Deepseek LLM（通过API调用最新的DeepSeek大模型）
llm = DeepSeek(
    model="deepseek-reasoner", # 使用最新的推理模型R1
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量获取API key
)

# 加载本地嵌入模型
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh" # 模型路径和名称（首次执行时会从HuggingFace下载）
)

# 设置全局配置
Settings.llm = llm  # 设置默认 LLM 为 DeepSeek
Settings.embed_model = embed_model  # 设置默认 embedding 模型

# 加载数据
documents = SimpleDirectoryReader(input_files=["data/wukong.txt"]).load_data() 

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    # 不需要在这里指定 embed_model，因为已经在全局设置中配置了
)

# 创建问答引擎
query_engine = index.as_query_engine(
    # 不需要在这里指定 llm，因为已经在全局设置中配置了
)

# 开始问答
print(query_engine.query("黑神话悟空中有哪些战斗工具?"))