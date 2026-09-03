# RAG 技术原理与实践指南

## 什么是 RAG

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合检索系统与大语言模型的架构范式。
其核心思想是：在 LLM 生成回答之前，先从外部知识库中检索与用户问题相关的内容片段，
将这些内容作为上下文注入提示词，让 LLM 基于检索到的知识生成更准确、更可信的回答。

## RAG 的核心组件

### 文档加载与解析
RAG 系统的第一步是加载外部知识源。常见的文档格式包括：
- 文本文件（TXT、Markdown）
- 办公文档（PDF、Word、Excel、PPT）
- 代码文件（Python、JavaScript、SQL）
- 网页内容（HTML）

文档加载器负责将这些格式统一转换为可处理的文本内容。

### 文本分块（Chunking）
由于 LLM 的上下文窗口有限，且检索粒度需要精细控制，文档需要被切分为合适大小的文本块。
常见的分块策略包括：
- **固定大小分块**：按字符数或 Token 数切分，带重叠窗口
- **递归字符分块**：按段落 → 句子 → 字符的优先级逐级切分
- **语义分块**：基于嵌入相似度检测语义边界，在语义完整的位置切分
- **结构化分块**：按文档的标题层级（Markdown 的 #/##/###）切分

### 向量嵌入（Embedding）
将文本块转换为稠密向量，便于语义相似度计算。常用的嵌入模型包括：
- 阿里云百炼 text-embedding-v3 / qwen3-vl-embedding
- OpenAI text-embedding-3-small / 3-large
- 开源模型 BGE、GTE、E5 等

### 向量存储与检索
嵌入向量存储在向量数据库中，支持高效的相似度搜索：
- **余弦距离**：衡量向量间的夹角，范围 [-1, 1]
- **欧氏距离**：衡量向量间的直线距离
- **内积距离**：衡量向量的大小和方向

### 检索增强生成
当用户提问时，流程如下：
1. 将用户问题编码为向量
2. 在向量库中检索最相似的 top-k 文本块
3. 将检索结果拼入提示词上下文
4. LLM 基于上下文生成回答

## 高级 RAG 技术

### 多查询扩展（MQE）
使用 LLM 生成用户问题的多个语义等价表述，分别检索后合并结果，
提升召回覆盖范围。

### 假设文档嵌入（HyDE）
先让 LLM 基于问题生成一个"假设答案"，再用这个假设答案去检索。
假设答案往往比原始问题包含更多语义信息，能提升检索准确率。

### 混合检索
结合语义检索（嵌入向量）和关键词检索（BM25/TF-IDF），
兼顾语义理解和精确匹配，是工业界最常用的方案。

### 重排序（Re-Ranking）
第一阶段检索出 top-20 候选结果后，使用交叉编码器重排序模型
对候选结果进行精确评分，再取 top-k 喂给 LLM。

### Parent-Child 分层索引
将文档切分为两级：
- **Child Chunk**：小粒度文本块，用于检索（精确匹配）
- **Parent Chunk**：大粒度文本块，包含 Child 的完整上下文，用于 LLM 生成

## LlamaIndex 框架简介

LlamaIndex 是一个专为 RAG 场景设计的数据框架，提供完整的工具链：

### 核心概念
- **Document**：外部数据源的原始文档
- **Node**：文档切分后的基本单元（带 metadata）
- **Index**：对 Node 构建的索引结构
- **Retriever**：从 Index 中检索相关 Node
- **ResponseSynthesizer**：将检索结果与 LLM 结合生成回答
- **QueryEngine**：端到端的查询引擎

### 典型工作流
```python
# 1. 加载文档
documents = SimpleDirectoryReader("data").load_data()

# 2. 构建索引
index = VectorStoreIndex.from_documents(documents)

# 3. 查询
query_engine = index.as_query_engine()
response = query_engine.query("什么是 RAG？")
```

## 总结

RAG 技术有效解决了大语言模型的幻觉问题、知识滞后问题和可解释性问题。
通过将外部知识库与 LLM 结合，RAG 能够：
1. 基于最新、最相关的知识回答
2. 提供可追溯的引用来源
3. 降低 LLM 产生幻觉的概率
4. 支持私域知识的快速接入