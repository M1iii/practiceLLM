"""离线脚本：接入 PGVectorStore，构建 Parent-Child 索引存入 PostgreSQL。"""

import os
import argparse
from pathlib import Path


from dotenv import load_dotenv
load_dotenv()

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.node_parser import (
    SentenceSplitter,
    HierarchicalNodeParser,
    get_leaf_nodes,
)

from src.tools.rag.llamaindex.adapters import DashScopeEmbedding, QwenLLM
from src.tools.rag.llamaindex.structured_index import MetadataExtractor, StructuredFilter


def parse_args():
    parser = argparse.ArgumentParser(
        description="离线构建索引并存入 PGVectorStore（PostgreSQL + pgvector）"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置表格：删除旧数据并重建",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "docs"),
        help="文档目录，默认 docs/",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="llamaindex_rag",
        help="PostgreSQL 表名",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=2560,
        help="嵌入维度（qwen3-vl-embedding = 2560）",
    )
    return parser.parse_args()


def get_pgvector_store(args) -> PGVectorStore:
    """从环境变量读取配置，创建 PGVectorStore。"""
    host = os.getenv("PG_HOST", "127.0.0.1")
    port = int(os.getenv("PG_PORT", "5433"))
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    database = os.getenv("PG_DATABASE", "practice_llm")

    print(f"  连接配置:")
    print(f"    host: {host}:{port}")
    print(f"    database: {database}")
    print(f"    table: {args.table_name}")
    print(f"    embed_dim: {args.embed_dim}")

    if args.reset:
        print(f"\n  --reset 模式：将删除旧表 {args.table_name} 并重建")
        # 先手动清空表
        from sqlalchemy import create_engine, text
        url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {args.table_name}"))
            conn.commit()

    # 使用 from_params，hnsw_kwargs=None 跳过索引创建
    # 2560 维超 pgvector hnsw/ivfflat 2000 维上限
    vector_store = PGVectorStore.from_params(
        database=database,
        host=host,
        port=port,
        user=user,
        password=password,
        table_name=args.table_name,
        embed_dim=args.embed_dim,
        hnsw_kwargs=None,
    )
    print(f"  表结构已创建（无 pgvector 索引，精确搜索不受影响）")

    return vector_store


def _extract_and_describe_images(documents: list, docs_dir: str) -> list:
    """从文档中提取图片引用，用 VL 模型生成描述并返回补充文档列表。

    支持 markdown 图片语法: ![alt](path/to/image.png)
    图片路径相对于 docs_dir 解析。
    """
    import re
    image_docs = []

    # 检查 VL 客户端是否可用
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("  [WARN] DASHSCOPE_API_KEY 未设置，跳过图片索引")
        return image_docs

    try:
        from src.tools.vl_media.vl_client import VLClient
        client = VLClient()
    except Exception as e:
        print(f"  [WARN] VLClient 初始化失败: {e}，跳过图片索引")
        return image_docs

    # 收集所有图片引用: (源文档, 图片路径, alt文字)
    refs = []
    for doc in documents:
        text = doc.text
        src_file = doc.metadata.get("file_name", "未知")
        # 匹配 markdown 图片语法
        for match in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', text):
            alt = match.group(1)
            img_path = match.group(2)
            refs.append((doc, img_path, alt, src_file))

    if not refs:
        return image_docs

    # 去重（同一张图片可能被多个文档引用）
    seen = set()
    unique_refs = []
    for ref in refs:
        img_path = ref[1]
        if img_path not in seen:
            seen.add(img_path)
            unique_refs.append(ref)

    print(f"  发现 {len(unique_refs)} 张图片引用（{len(refs)} 次，去重后）")

    for doc, img_path, alt, src_file in unique_refs:
        # 解析图片路径
        full_path = os.path.join(docs_dir, img_path)
        if not os.path.exists(full_path):
            # 尝试相对于源文件目录
            parent_dir = os.path.dirname(
                doc.metadata.get("file_path", docs_dir))
            full_path = os.path.join(parent_dir, img_path)
        if not os.path.exists(full_path):
            print(f"    ⚠️  {img_path} 未找到，跳过")
            continue

        # 检查文件大小
        try:
            fsize = os.path.getsize(full_path)
            if fsize > 10 * 1024 * 1024:
                print(f"    ⚠️  {img_path} 超过 10MB，跳过")
                continue
        except OSError:
            continue

        # 用 VL 模型生成描述
        prompt = f"请详细描述这张图片的内容。图片源自文档《{src_file}》，alt文字为「{alt}」。"
        try:
            result = client.analyze_image(full_path, prompt)
            description = result.strip()
            if not description or "抱歉" in description[:20]:
                print(f"    ⚠️  {img_path}: VL 描述不完整，跳过")
                continue
        except Exception as e:
            print(f"    ⚠️  {img_path}: VL 分析失败: {e}，跳过")
            continue

        # 创建描述文档
        from llama_index.core.schema import Document
        img_doc = Document(
            text=f"【图片描述】{description}\n\n"
                 f"（图片路径: {img_path}，alt文字: {alt}，来源文档: {src_file}）",
            metadata={
                "file_name": f"{src_file}_img_{os.path.basename(img_path)}",
                "doc_type": "image",
                "tags": ["image", "multimodal"],
                "source_file": src_file,
                "image_path": img_path,
                "image_alt": alt,
            },
        )
        image_docs.append(img_doc)
        print(f"    [OK] {img_path}: 描述已生成 ({len(description)} 字符)")

    return image_docs


def build_parent_child_index(vector_store, docs_dir: str, embed_model, args):
    print("\n--- Step 1: 加载文档 ---")
    reader = SimpleDirectoryReader(
        input_dir=docs_dir,
        required_exts=[".md", ".txt"],
        filename_as_id=True,
    )
    documents = reader.load_data()
    print(f"  加载了 {len(documents)} 个文档")
    for doc in documents:
        src = doc.metadata.get("file_name", "未知")
        length = len(doc.text)
        print(f"    - {src} ({length} 字符)")

    extractor = MetadataExtractor()
    enriched_docs = []
    for doc in documents:
        struct_meta = extractor.extract(doc)
        # 合并到文档原有 metadata
        doc.metadata.update(struct_meta)
        enriched_docs.append(doc)
        fn = struct_meta.get("file_name", "未知")
        dt = struct_meta.get("doc_type", "?")
        tags = struct_meta.get("tags", [])
        print(f"    {fn}: doc_type={dt}, tags={tags}")

    print(f"\n  结构化元数据抽取完成，共 {len(enriched_docs)} 个文档")

    image_docs = _extract_and_describe_images(documents, docs_dir)
    if image_docs:
        print(f"  生成了 {len(image_docs)} 个图片描述文档")
        enriched_docs.extend(image_docs)
    else:
        print("  未发现图片或图片描述生成失败（跳过）")

    print("\n=" * 60)
    print("Step 2: 构建 Parent-Child 节点")
    print("=" * 60)

    # 一次解析：用 enriched_docs（含元数据），确保节点 ID 唯一
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1024, 256],
        chunk_overlap=30,
    )
    nodes = node_parser.get_nodes_from_documents(enriched_docs)
    leaf_nodes = get_leaf_nodes(nodes)

    print(f"  切分完成:")
    print(f"    总节点数: {len(nodes)}")
    print(f"    叶子节点 (leaf): {len(leaf_nodes)}")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 直接使用已解析的叶子节点，避免重复解析导致 KeyError
    index = VectorStoreIndex(
        nodes=leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print(f"\n  索引构建完成，数据已写入 PostgreSQL 表 {args.table_name}")

    sf = StructuredFilter(table_name=args.table_name)
    created = sf.create_gin_index()
    if created:
        print("  ✅ GIN 索引创建成功")
    else:
        print("  ⚠️  GIN 索引已存在或创建失败")

    print()

    return index, leaf_nodes, documents


def test_retrieval(index, llm):
    retriever = index.as_retriever(similarity_top_k=3)

    test_queries = [
        "RAG 技术解决了什么问题？",
        "介绍一下 Parent-Child 分层索引",
        "LlamaIndex 的核心概念有哪些？",
    ]

    for query in test_queries:
        print(f"\n  查询: \"{query}\"")
        results = retriever.retrieve(query)
        print(f"  检索到 {len(results)} 个结果:")
        for i, result in enumerate(results):
            score = result.score
            src = result.node.metadata.get("file_name", "未知")
            level = result.node.metadata.get("level", "?")
            preview = result.node.text[:60].replace("\n", " ")
            if score is not None:
                print(f"    [{i + 1}] (score={score:.4f}, level={level}, src={src}) {preview}...")

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
    )

    questions = [
        "RAG 技术和纯 LLM 相比有哪些优势？",
        "Parent-Child 索引设计解决了什么问题？",
    ]

    for question in questions:
        print(f"\n  问题: \"{question}\"")
        response = query_engine.query(question)
        print(f"  回答:\n{response}")
        if hasattr(response, "source_nodes") and response.source_nodes:
            print(f"\n  引用来源 ({len(response.source_nodes)} 个):")
            for i, sn in enumerate(response.source_nodes):
                src = sn.node.metadata.get("file_name", "未知")
                level = sn.node.metadata.get("level", "?")
                score = sn.score
                preview = sn.node.text[:50].replace("\n", " ")
                if score is not None:
                    print(f"    [{i + 1}] (level={level}, score={score:.4f}) {preview}...")

    print("\n" + "=" * 60)
    print("构建与测试完成！")
    print("=" * 60)


def main():
    args = parse_args()

    print("=" * 60)
    print("LlamaIndex 第二阶段：PGVectorStore + Parent-Child 索引构建")
    print("=" * 60)
    print()
    print(f"  文档目录: {args.docs_dir}")
    print(f"  表名: {args.table_name}")
    print(f"  嵌入维度: {args.embed_dim}")

    # 1. 创建嵌入模型
    print("\n=" * 60)
    print("Step 0: 初始化组件")
    print("=" * 60)

    embed_model = DashScopeEmbedding()
    llm = QwenLLM()
    print(f"  嵌入模型: {embed_model.class_name()}")
    print(f"  LLM: {llm.model}")

    # 2. 创建 PGVectorStore
    vector_store = get_pgvector_store(args)

    # 3. 构建 Parent-Child 索引
    index, leaf_nodes, documents = build_parent_child_index(
        vector_store, args.docs_dir, embed_model, args
    )

    test_retrieval(index, llm)

    print(f"\n  总结:")
    print(f"    文档数: {len(documents)}")
    print(f"    叶子节点数: {len(leaf_nodes)}")
    print(f"    已存入 PostgreSQL: {os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}.{args.table_name}")
    print()
    print("  可使用以下代码从数据库加载索引:")
    print("""
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

vector_store = PGVectorStore.from_params(...)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
)
""")


if __name__ == "__main__":
    main()