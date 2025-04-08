import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# load doc
loader = WebBaseLoader("https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/202503/t20250324_3960464.htm#:~:text=2024%E5%B9%B4%EF%BC%8C%E5%85%A8%E5%9B%BD%E7%A8%8E%E6%94%B6%E6%94%B6%E5%85%A5,%EF%BC%8C%E4%B8%AA%E4%BA%BA%E6%89%80%E5%BE%97%E7%A8%8E%E4%B8%8B%E9%99%8D1.7%25%E3%80%82")

documents = loader.load()

# split doc
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents)

# set embedding model
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# set vector store
vector_store = InMemoryVectorStore(embedding=embedding)
vector_store.add_documents(all_splits)

# create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# set prompt template
prompt = ChatPromptTemplate.from_template(
    template="""
    上下文：{context}
    问题：{question}
    答案：
    """
)

# use llm to answer question
llm = ChatOpenAI(
    model="deepseek-chat", 
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,
    max_tokens=1000,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# create the chain
chain = (
    {
        "context": lambda x: "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(x["question"])]),
        "question": lambda x: x["question"]
    }
    | prompt 
    | llm
    | StrOutputParser()
)

# invoke the chain
answer = chain.invoke({"question": "总结2024年税收收入情况"})

print(answer)





