from pathlib import Path
import os
import subprocess
import sys
import warnings

# macOS: 避免 FAISS 与 libomp 重复加载导致 abort（OMP Error #15）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_ollama import OllamaLLM  # type: ignore[reportMissingImports]
    USING_COMMUNITY_OLLAMA = False
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM
    USING_COMMUNITY_OLLAMA = True


# 设置你的知识库路径
DATA_PATH = "./IELTS"
INDEX_PATH = "faiss_index_directory"
EMBEDDING_MODEL_NAME = "nomic-embed-text"  # 使用本地 Ollama embedding 模型，无需联网
DEFAULT_OLLAMA_MODEL = "qwen:4b"

if USING_COMMUNITY_OLLAMA:
    try:
        from langchain_core._api.deprecation import LangChainDeprecationWarning

        warnings.filterwarnings(
            "ignore",
            category=LangChainDeprecationWarning,
            message=r"The class `Ollama` was deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=LangChainDeprecationWarning,
            message=r"The class `OllamaEmbeddings` was deprecated.*",
        )
    except Exception:
        pass


def load_raw_documents(data_path: str):
    raw_documents = []
    for file_path in Path(data_path).rglob("*.txt"):
        try:
            single_loader = DirectoryLoader(
                str(file_path.parent),
                glob=file_path.name,
                loader_cls=TextLoader,
                silent_errors=True,
            )
            docs = single_loader.load()
            raw_documents.extend(docs)
        except Exception as e:
            print(f"[警告] 加载文件失败: {file_path}，原因: {e}")
    return raw_documents


def build_vectorstore(embeddings):
    raw_documents = load_raw_documents(DATA_PATH)
    print(f"成功加载了 {len(raw_documents)} 份文档。")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"文档被切分成了 {len(documents)} 个片段。")

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"已保存索引到: {INDEX_PATH}")
    return vectorstore


def init_embeddings():
    # 使用本地 Ollama 的 nomic-embed-text，完全离线，无需 HuggingFace
    return OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)


def list_ollama_models():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    models = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def select_ollama_model():
    preferred = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    available = list_ollama_models()
    if not available:
        return preferred, available
    if preferred in available:
        return preferred, available
    return available[0], available


def call_llm_direct(question: str):
    model_name, available_models = select_ollama_model()
    llm = OllamaLLM(model=model_name)
    try:
        answer = llm.invoke(question)
    except Exception as e:
        print("[错误] 调用 Ollama 失败。")
        print(f"原因: {e}")
        if available_models:
            print(f"[提示] 当前可用模型: {', '.join(available_models)}")
            print("[提示] 可通过环境变量 OLLAMA_MODEL 指定模型，例如: OLLAMA_MODEL=<你的模型名> python3 run.py")
        else:
            print(f"[提示] 未检测到可用模型，请先执行: ollama pull {DEFAULT_OLLAMA_MODEL}")
        sys.exit(1)
    return answer


def main():
    index_file = Path(INDEX_PATH) / "index.faiss"
    has_local_index = index_file.exists()

    try:
        embeddings = init_embeddings()
    except Exception as e:
        print("[错误] 初始化 Embedding 模型失败。")
        print(f"原因: {e}")
        print("[提示] 当前将降级为非 RAG 模式（仅使用本地 Ollama 直答）。")
        print(f"[提示] 请确保已执行: ollama pull {EMBEDDING_MODEL_NAME}")
        print("已就绪（非 RAG 模式），输入问题后按回车，输入 exit 或按 Ctrl+C 退出。\n")
        while True:
            try:
                question = input("Q> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n再见！")
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                print("再见！")
                break
            answer = call_llm_direct(question)
            print(f"A> {answer}\n")
        sys.exit(0)

    if has_local_index:
        try:
            vectorstore = FAISS.load_local(
                INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"已加载本地索引: {INDEX_PATH}")
        except Exception as e:
            print(f"[警告] 加载本地索引失败，尝试重建。原因: {e}")
            vectorstore = build_vectorstore(embeddings)
    else:
        print("未检测到本地索引，开始构建新索引...")
        vectorstore = build_vectorstore(embeddings)

    retriever = vectorstore.as_retriever()
    model_name, available_models = select_ollama_model()
    llm = OllamaLLM(model=model_name)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    print("已就绪，输入问题后按回车，输入 exit 或按 Ctrl+C 退出。\n")
    while True:
        try:
            question = input("Q> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("再见！")
            break

        try:
            result = qa_chain.invoke(question)
        except Exception as e:
            print("[错误] RAG 问答执行失败。")
            print(f"原因: {e}")
            if available_models:
                print(f"[提示] 当前可用模型: {', '.join(available_models)}")
            else:
                print(f"[提示] 未检测到可用模型，请先执行: ollama pull {DEFAULT_OLLAMA_MODEL}")
            sys.exit(1)

        answer = result.get("result", result) if isinstance(result, dict) else result
        print(f"A> {answer}\n")

        # 打印检索到的素材来源，帮助判断答案是否来自 IELTS 素材库
        source_docs = result.get("source_documents", []) if isinstance(result, dict) else []
        if source_docs:
            print("[来源] 以下素材被用于生成上述回答（来自 IELTS 素材库）：")
            seen = set()
            for doc in source_docs:
                src = doc.metadata.get("source", "未知文件")
                if src not in seen:
                    seen.add(src)
                    print(f"  - {src}")
        else:
            print("[来源] 未检索到相关素材，回答完全由 Qwen 模型生成。")
        print()


if __name__ == "__main__":
    main()