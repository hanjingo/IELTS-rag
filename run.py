from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 设置你的知识库路径
DATA_PATH = "./IELTS"

# 创建一个加载器，它会自动处理路径下的 .md, .txt, .pdf 等文件
loader = DirectoryLoader(DATA_PATH, silent_errors=True)
raw_documents = loader.load()

print(f"成功加载了 {len(raw_documents)} 份文档。")

# 初始化一个文本切分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个chunk的大小
    chunk_overlap=50, # 相邻chunk的重叠大小
    separators=["\n\n", "\n", " ", ""] # 分隔符
)

# 开始切分
documents = text_splitter.split_documents(raw_documents)
print(f"文档被切分成了 {len(documents)} 个片段。")

# 指定我们要用的Embedding模型
embedding_model_name = 'BAAI/bge-large-zh-v1.5'

# 初始化Embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cpu'}, # 如果你有GPU，可以改成 'cuda'
    encode_kwargs={'normalize_embeddings': True}
)

# 使用我们准备好的文档片段和Embedding模型，构建FAISS索引
vectorstore = FAISS.from_documents(documents, embeddings)

# [可选] 我们可以把建好的索引存到本地，下次就不用重新构建了
vectorstore.save_local("faiss_index_directory")

# 加载方式：
# vectorstore = FAISS.load_local("faiss_index_directory", embeddings, allow_dangerous_deserialization=True)

# 把它变成一个检索器（Retriever）
retriever = vectorstore.as_retriever()

# 初始化一个LLM
llm = Ollama(model="qwen:7b-chat") # 替换成你自己的模型

# 创建一个问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "stuff"模式会把所有检索到的内容一次性塞给LLM
    retriever=retriever # 使用我们第三步创建的检索器
)

# 让我们来问个问题！
question = "请问，RAG系统中的混合检索是什么？"
answer = qa_chain.invoke(question)

print(answer)