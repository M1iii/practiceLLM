"""
第一阶段 POC：验证 LlamaIndex 集成流程完整性。

执行流程：
  1. SimpleDirectoryReader 加载 docs/ 下的测试文档
  2. SentenceSplitter 分块（chunk_size=256, chunk_overlap=30）
  3. 使用项目现有的 EmbeddingClient 生成向量（DashScopeEmbedding 适配器）
  4. 使用 VectorStoreIndex（内存模式）构建索引
  5. 查询并打印结果

用法：
  cd 项目根目录
  python -m rag.llamaindex.poc
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def main():
    print("=" * 60)
    print("Step 1: 加载测试文档")
    print("=" * 60)

    from llama_index.core import SimpleDirectoryReader

    docs_dir = Path(__file__).resolve().parent.parent.parent / "docs"
    reader = SimpleDirectoryReader(
        input_dir=str(docs_dir),
        required_exts=[".md"],
        # 只读取我们创建的测试文档，避免加载过多文件
        filename_as_id=True,
    )
    documents = reader.load_data()
    print(f"  加载了 {len(documents)} 个文档:")
    for doc in documents:
        src = doc.metadata.get("file_name", "未知")
        length = len(doc.text)
        print(f"    - {src} ({length} 字符)")

    print()
    print("=" * 60)
    print("Step 2: SentenceSplitter 分块")
    print("=" * 60)

    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=30,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"  分块完成，共 {len(nodes)} 个节点")
    for i, node in enumerate(nodes):
        src = node.metadata.get("file_name", "未知")
        preview = node.text[:60].replace("\n", " ")
        print(f"    [{i + 1}] ({src}) {preview}...")

    print()
    print("=" * 60)
    print("Step 3: 构建 VectorStoreIndex（内存模式）")
    print("=" * 60)

    from src.tools.rag.llamaindex.adapters import DashScopeEmbedding

    embed_model = DashScopeEmbedding()
    print(f"  Embedding 模型: {embed_model.class_name()}")

    from llama_index.core import VectorStoreIndex

    index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
    print("  索引构建完成")

    print()
    print("=" * 60)
    print("Step 4: 检索测试（无 LLM 生成）")
    print("=" * 60)

    # 使用纯检索模式（不依赖 LLM）
    retriever = index.as_retriever(similarity_top_k=3)

    test_queries = [
        "什么是 RAG？",
        "RAG 的核心组件有哪些？",
        "LlamaIndex 是什么？",
    ]

    for query in test_queries:
        print(f"\n  查询: \"{query}\"")
        results = retriever.retrieve(query)
        print(f"  检索到 {len(results)} 个结果:")
        for i, result in enumerate(results):
            score = result.score
            preview = result.text[:80].replace("\n", " ")
            if score is not None:
                print(f"    [{i + 1}] (score={score:.4f}) {preview}...")
            else:
                print(f"    [{i + 1}] {preview}...")

    print()
    print("=" * 60)
    print("Step 5: 端到端问答（含 LLM 生成）")
    print("=" * 60)

    from src.tools.rag.llamaindex.adapters import QwenLLM

    llm = QwenLLM()
    print(f"  LLM: {llm.model}")

    query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

    test_questions = [
        "RAG 技术解决了大语言模型的哪些问题？",
        "什么是 Parent-Child 分层索引？",
    ]

    for question in test_questions:
        print(f"\n  问题: \"{question}\"")
        response = query_engine.query(question)
        print(f"  回答: {response}")
        if hasattr(response, "source_nodes") and response.source_nodes:
            print(f"  引用来源 ({len(response.source_nodes)} 个):")
            for i, sn in enumerate(response.source_nodes):
                src = sn.node.metadata.get("file_name", "未知")
                preview = sn.node.text[:60].replace("\n", " ")
                score = sn.score
                if score is not None:
                    print(f"    [{i + 1}] ({src}, score={score:.4f}) {preview}...")
                else:
                    print(f"    [{i + 1}] ({src}) {preview}...")

    print()
    print("=" * 60)
    print("POC 验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()