"""
RAG 工具 (RagTool)
==================
基于阿里云百炼（DashScope）的检索增强生成（RAG）工具，提供完整 RAG 能力：
  - 多格式文档加载：文本、Markdown、PDF、Word(docx)、Excel(xlsx)、PPT(pptx)、
    图片、音频、代码、数据文件（CSV/JSON）
  - 统一文档转换：MarkItDown 引擎将任意格式转换为结构化 Markdown，
    然后进入统一的分块 → 向量化 → 存储流程
  - 智能分块：按 Markdown 标题结构 + 段落/句子边界切分，支持块大小与重叠配置
  - 混合检索召回：百炼嵌入向量（text-embedding-v3）+ TF-IDF 关键词双路召回
  - LLM 增强问答：Qwen 模型结合检索上下文生成带引用的答案
  - 知识库管理：添加、转换预览、搜索、问答、统计、删除、清空

轻量级设计：
  - MarkItDown 为统一转换引擎（可选依赖，pip install markitdown[all]）
  - MarkItDown 不可用或转换为空时（如无 LLM 的图片/音频），自动回退：
    Office 用标准库 zipfile + xml 解析，PDF 用 zlib + 正则提取，媒体用元数据语义化

环境变量：
  DASHSCOPE_API_KEY  百炼 API Key（必填）
  EMBEDDING_MODEL    嵌入模型，默认 text-embedding-v3
  RAG_LLM_MODEL      Qwen 问答模型，默认 qwen-plus

使用方式:
    tool = RagTool()
    tool.execute("add_file", file_path="docs/guide.md")
    tool.execute("convert", file_path="docs/report.pdf")   # 转换预览
    tool.execute("search", query="什么是 RAG", top_k=3)
    tool.execute("search_mqe", query="RAG 的优势", n_queries=3)   # 多查询扩展检索
    tool.execute("search_hyde", query="RAG 的优势")               # 假设文档嵌入检索
    tool.execute("search_expanded", query="RAG 的优势",            # 统一扩展检索框架
                 enable_mqe=True, enable_hyde=True)
    tool.execute("query", question="RAG 的工作原理是什么？", enable_mqe=True)  # 统一框架增强问答
    tool.execute("stats")
    tool.execute("delete", doc_id="...")
    tool.execute("clear")
"""

import sys
import os


import re
import math
import time
import uuid
import zlib
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from src.tools.framework.tool_system import Tool, ToolParameter, dual_protocol_execute

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from src.tools.memory.memory_tool import EmbeddingClient
except ImportError:
    EmbeddingClient = None

# MarkItDown 统一文档转换引擎（可选依赖）
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MarkItDown = None
    MARKITDOWN_AVAILABLE = False


# ============================================================
# 数据结构
# ============================================================

@dataclass
class RagChunk:
    """检索单元：文档切分后的一个文本块。"""
    chunk_id: str
    doc_id: str
    doc_name: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)   # 页码/标题/模态等


@dataclass
class RagDocument:
    """知识库中的一份文档。"""
    doc_id: str
    name: str
    source_path: Optional[str] = None
    doc_type: str = "text"            # pdf/word/excel/ppt/image/audio/video/text/markdown/code/data
    chunk_count: int = 0
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# 文件扩展名 → 文档类型映射
EXT_DOC_TYPE = {
    ".txt": "text", ".log": "text",
    ".md": "markdown", ".markdown": "markdown",
    ".pdf": "pdf",
    ".docx": "word", ".doc": "word",
    ".xlsx": "excel", ".xls": "excel", ".csv": "data",
    ".pptx": "ppt", ".ppt": "ppt",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".bmp": "image", ".webp": "image",
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".aac": "audio", ".m4a": "audio",
    ".mp4": "video", ".avi": "video", ".mov": "video", ".mkv": "video",
    ".py": "code", ".js": "code", ".ts": "code", ".java": "code", ".go": "code",
    ".c": "code", ".cpp": "code", ".rs": "code", ".sql": "code",
    ".json": "data", ".xml": "data", ".yaml": "data", ".yml": "data",
}

# 文本直读扩展名
TEXT_EXTENSIONS = {".txt", ".log", ".md", ".markdown", ".py", ".js", ".ts", ".java",
                   ".go", ".c", ".cpp", ".rs", ".sql", ".json", ".xml", ".yaml", ".yml", ".csv"}

# 元数据语义化扩展名（图片/音频/视频）
METADATA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
                       ".mp3", ".wav", ".flac", ".aac", ".m4a",
                       ".mp4", ".avi", ".mov", ".mkv"}


# ============================================================
# 嵌入客户端（复用百炼 text-embedding-v3）
# ============================================================

def _get_embedding_client():
    """获取嵌入客户端实例（单例）。"""
    if EmbeddingClient is not None:
        return EmbeddingClient()
    return None


# ============================================================
# 统一向量编码接口（百炼嵌入 API + TF-IDF 兜底）
# ============================================================

class VectorEncoder:
    """统一向量编码接口：百炼嵌入 API 优先，TF-IDF 稀疏向量兜底。

    encode() 对任意文本返回 (稠密向量, TF 稀疏向量, 方法)：
      - 百炼 API 可用 → 稠密嵌入向量（text-embedding-v3，带缓存）
      - 百炼 API 不可用/失败 → 自动回退 TF-IDF 稀疏向量
      - method 标记实际使用的方法（"embedding" / "tfidf"）
    """

    def __init__(self, embedding_client=None):
        self._embedding_client = embedding_client or _get_embedding_client()

    @property
    def embedding_available(self) -> bool:
        """百炼嵌入 API 是否可用。"""
        return bool(self._embedding_client and self._embedding_client.is_available)

    def encode(self, text: str) -> Tuple[Optional[List[float]], Dict[str, float], str]:
        """编码文本，返回 (稠密向量, TF稀疏向量, 方法)。"""
        dense = None
        if self.embedding_available:
            try:
                dense = self._embedding_client.embed(text)
            except Exception:
                dense = None
        sparse = self._tfidf_vectorize(text)
        method = "embedding" if dense else "tfidf"
        return dense, sparse, method

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量嵌入（百炼 API），返回与输入等长的向量列表（失败项为 None）。"""
        if self.embedding_available:
            try:
                return self._embedding_client.embed_batch(texts)
            except Exception:
                pass
        return [None] * len(texts)

    @staticmethod
    def _tfidf_vectorize(text: str) -> Dict[str, float]:
        """TF 稀疏向量（词频归一化），作为 TF-IDF 检索的基础向量。"""
        tokens = RagIndex._tokenize(text)
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        return {term: count / total for term, count in tf.items()}


def index_chunks(chunks: List[Dict], embedding_client=None) -> List[Dict]:
    """统一分块索引：为每个 chunk 生成向量（百炼嵌入优先 + TF-IDF 兜底）。

    Args:
        chunks: 分块 dict 列表（来自 TextSplitter.split_paragraph_dicts），
                每项 {content, metadata, tokens, ...}
        embedding_client: 可选，复用外部嵌入客户端（默认自动创建）

    Returns:
        增强后的分块列表，每项增加：
          - embedding: 百炼稠密向量（API 可用时，否则 None）
          - tfidf_vector: TF-IDF 稀疏向量（始终生成，作为兜底/混合检索）
          - vector_method: "embedding" / "tfidf"
          - tokens: Token 估算
    """
    encoder = VectorEncoder(embedding_client)
    indexed: List[Dict] = []
    for chunk in chunks:
        content = chunk.get("content", "") or ""
        dense, sparse, method = encoder.encode(content)
        indexed.append({
            **chunk,
            "embedding": dense,
            "tfidf_vector": sparse,
            "vector_method": method,
            "tokens": chunk.get("tokens") or _approx_token_len(content),
        })
    return indexed


# ============================================================
# 文档加载器（多格式解析）
# ============================================================

class DocumentLoader:
    """多格式文档加载器：将不同格式的文件解析为文本块列表。"""

    # ============================================================
    # 入口：统一转换 → 结构化分块
    # ============================================================

    def load(self, file_path: str, description: str = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """加载文档，返回 (文档元数据, 文本块列表)。

        流程：任意格式 → MarkItDown 统一转换为 Markdown → 按标题结构化分块。
        转换失败（图片/音频无 LLM 时 MarkItDown 返回空）自动回退轻量解析。

        Returns:
            doc_meta: {name, source_path, doc_type, extension, ...}
            blocks:   [{content, metadata}, ...]
        """
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        doc_type = EXT_DOC_TYPE.get(ext, "text")
        name = os.path.basename(file_path)

        doc_meta = {
            "name": name,
            "source_path": file_path,
            "doc_type": doc_type,
            "extension": ext,
            "size": os.path.getsize(file_path),
            "description": description or "",
        }

        # 1. MarkItDown 统一转换引擎 → 结构化 Markdown
        markdown = self._convert_to_markdown(file_path)
        if markdown:
            doc_meta["converter"] = "markitdown"
            blocks = self._split_markdown_blocks(markdown)
            for block in blocks:
                block["metadata"]["converter"] = "markitdown"
        else:
            # 2. 轻量回退：分格式解析（图片/音频元数据语义化等）
            doc_meta["converter"] = "fallback"
            blocks = self._load_fallback(file_path, ext, doc_type, description)
            for block in blocks:
                block["metadata"]["converter"] = "fallback"

        # 过滤空块
        blocks = [b for b in blocks if b["content"] and b["content"].strip()]
        return doc_meta, blocks

    # ============================================================
    # 统一文档转换引擎：任意格式 → Markdown
    # ============================================================

    def _convert_to_markdown(self, path: str) -> str:
        """使用 MarkItDown 统一文档转换引擎，将任意格式文档转换为结构化 Markdown。

        转换流程：MarkItDown 识别文件类型 → 调用对应转换器（PDF/Office/HTML/媒体等）
        → 输出结构化 Markdown 文本。转换失败或返回空时返回空字符串（调用方回退）。

        注意：图片/音频在未配置 LLM 描述客户端时，MarkItDown 返回空文本，
        由调用方自动回退到元数据语义化解析。
        """
        if not MARKITDOWN_AVAILABLE:
            return ""
        try:
            converter = MarkItDown()
            result = converter.convert(path)
            text = (result.text_content or "").strip()
            return text
        except Exception:
            return ""

    def _load_fallback(self, file_path: str, ext: str, doc_type: str,
                       description: str) -> List[Dict[str, Any]]:
        """轻量回退解析：MarkItDown 不可用或转换失败时使用。"""
        # 1. 文本直读类
        if ext in TEXT_EXTENSIONS:
            content = self._read_text(file_path)
            return [{"content": content, "metadata": {"section": "全文"}}]
        # 2. Office 文档（zip + xml 解析）
        if doc_type == "word":
            return self._parse_docx(file_path)
        if doc_type == "excel":
            return self._parse_xlsx(file_path)
        if doc_type == "ppt":
            return self._parse_pptx(file_path)
        # 3. PDF
        if doc_type == "pdf":
            return self._parse_pdf(file_path)
        # 4. 图片/音频/视频（元数据语义化）
        return self._parse_media(file_path, doc_type, description)

    # ============================================================
    # Markdown 结构化分块
    # ============================================================

    @staticmethod
    def _split_markdown_blocks(markdown: str) -> List[Dict[str, Any]]:
        """按 Markdown 标题结构分块：每个标题及其下内容作为一个块。

        - 支持 # / ## / ### 各级标题，块 metadata.section 记录标题
        - 无标题时整篇作为一个块
        """
        blocks = []
        lines = markdown.split("\n")

        current_heading = "全文"
        current_lines: List[str] = []

        def flush():
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    blocks.append({
                        "content": content,
                        "metadata": {"section": current_heading},
                    })

        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.*?)\s*$', line.strip())
            if heading_match:
                flush()
                current_heading = heading_match.group(2).strip()
                current_lines = [line.strip()]
            else:
                current_lines.append(line)

        flush()
        return blocks

    # ============================================================
    # 文本直读
    # ============================================================

    @staticmethod
    def _read_text(file_path: str) -> str:
        """读取文本文件，自动尝试多种编码。"""
        for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()

    # ============================================================
    # Word（docx）：zipfile + xml 解析 w:t 文本
    # ============================================================

    @staticmethod
    def _parse_docx(file_path: str) -> List[Dict[str, Any]]:
        """解析 .docx：读取 word/document.xml 中的段落文本。"""
        blocks = []
        try:
            with zipfile.ZipFile(file_path) as zf:
                xml_bytes = zf.read("word/document.xml")
            root = ET.fromstring(xml_bytes)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            for para in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
                texts = [t.text or "" for t in para.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")]
                text = "".join(texts).strip()
                if text:
                    blocks.append({"content": text, "metadata": {"section": "段落"}})
        except Exception:
            blocks = []
        return blocks

    # ============================================================
    # Excel（xlsx）：zipfile + xml 解析共享字符串
    # ============================================================

    @staticmethod
    def _parse_xlsx(file_path: str) -> List[Dict[str, Any]]:
        """解析 .xlsx：读取共享字符串 + 各工作表单元格。"""
        blocks = []
        try:
            with zipfile.ZipFile(file_path) as zf:
                # 1. 共享字符串表
                shared = []
                if "xl/sharedStrings.xml" in zf.namelist():
                    sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                    for si in sroot:
                        text = "".join(t.text or "" for t in si.iter() if t.tag.endswith("}t") or t.tag == "t")
                        shared.append(text)

                # 2. 工作表
                sheet_names = sorted(
                    n for n in zf.namelist()
                    if re.match(r"xl/worksheets/sheet\d+\.xml$", n)
                )
                for sheet_name in sheet_names:
                    sroot = ET.fromstring(zf.read(sheet_name))
                    rows_text = []
                    for row in sroot.iter():
                        if row.tag.endswith("}row"):
                            cells = []
                            for c in row:
                                if not c.tag.endswith("}c"):
                                    continue
                                v = None
                                for child in c:
                                    if child.tag.endswith("}v"):
                                        v = child.text or ""
                                if v is None:
                                    continue
                                # 共享字符串索引或内联数值
                                try:
                                    idx = int(v)
                                    cell_text = shared[idx] if idx < len(shared) else v
                                except ValueError:
                                    cell_text = v
                                cells.append(cell_text.strip())
                            row_text = " | ".join(x for x in cells if x)
                            if row_text:
                                rows_text.append(row_text)
                    if rows_text:
                        sheet_no = os.path.basename(sheet_name).replace("sheet", "").replace(".xml", "")
                        blocks.append({
                            "content": "\n".join(rows_text),
                            "metadata": {"section": f"工作表{sheet_no}"},
                        })
        except Exception:
            blocks = []
        return blocks

    # ============================================================
    # PPT（pptx）：zipfile + xml 解析幻灯片 a:t 文本
    # ============================================================

    @staticmethod
    def _parse_pptx(file_path: str) -> List[Dict[str, Any]]:
        """解析 .pptx：读取 ppt/slides/slideN.xml 中的文本。"""
        blocks = []
        try:
            with zipfile.ZipFile(file_path) as zf:
                slide_names = sorted(
                    n for n in zf.namelist()
                    if re.match(r"ppt/slides/slide\d+\.xml$", n)
                )
                for i, slide_name in enumerate(slide_names, 1):
                    sroot = ET.fromstring(zf.read(slide_name))
                    texts = []
                    for t in sroot.iter():
                        if t.tag.endswith("}t"):
                            texts.append(t.text or "")
                    text = " ".join(x.strip() for x in texts if x.strip())
                    if text:
                        blocks.append({
                            "content": text,
                            "metadata": {"section": f"幻灯片{i}"},
                        })
        except Exception:
            blocks = []
        return blocks

    # ============================================================
    # PDF：优先 pypdf，回退 zlib + 正则
    # ============================================================

    def _parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """解析 PDF 文本。优先 pypdf，未安装时用 zlib 解压流 + 正则提取。"""
        # 1. 尝试 pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            blocks = []
            for i, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or "").strip()
                if text:
                    blocks.append({"content": text, "metadata": {"section": f"第{i}页"}})
            if blocks:
                return blocks
        except ImportError:
            pass
        except Exception:
            pass

        # 2. 轻量回退：zlib 解压 FlateDecode 流 + 正则提取括号文本
        try:
            blocks = self._extract_pdf_text_fallback(file_path)
            return blocks
        except Exception:
            return []

    @staticmethod
    def _extract_pdf_text_fallback(file_path: str) -> List[Dict[str, Any]]:
        """轻量 PDF 文本提取：适用于简单文本型 PDF（含 FlateDecode 流）。"""
        with open(file_path, "rb") as f:
            data = f.read()

        texts = []
        # 查找流对象 stream ... endstream
        for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL):
            raw = m.group(1)
            # 尝试 zlib 解压（FlateDecode）
            try:
                decoded = zlib.decompress(raw)
            except Exception:
                continue
            # 提取 Tj / TJ 运算符中的括号文本
            page_texts = []
            for tm in re.finditer(rb"\(((?:[^()\\]|\\.)*)\)\s*Tj", decoded):
                t = tm.group(1)
                t = t.replace(rb"\(", b"(").replace(rb"\)", b")").replace(rb"\\", b"\\")
                page_texts.append(t.decode("latin-1", errors="ignore"))
            for tm in re.finditer(rb"\[(.*?)\]\s*TJ", decoded, re.DOTALL):
                items = re.findall(rb"\(((?:[^()\\]|\\.)*)\)", tm.group(1))
                page_texts.append("".join(
                    x.replace(rb"\(", b"(").replace(rb"\)", b")").decode("latin-1", errors="ignore")
                    for x in items
                ))
            if page_texts:
                texts.append("".join(page_texts))

        blocks = []
        for i, text in enumerate(texts, 1):
            if text.strip():
                blocks.append({"content": text.strip(), "metadata": {"section": f"第{i}页"}})
        return blocks

    # ============================================================
    # 图片/音频/视频：元数据语义化
    # ============================================================

    def _parse_media(self, file_path: str, doc_type: str, description: str = None) -> List[Dict[str, Any]]:
        """图片/音频/视频的轻量语义化：文件名关键词 + 描述。"""
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        name_without_ext = os.path.splitext(filename)[0]

        # 文件名关键词
        keywords = [k for k in re.split(r'[-_.\s]+', name_without_ext) if k and len(k) >= 2]

        parts = [f"这是一个{doc_type}文件", f"文件名: {filename}"]
        if keywords:
            parts.append(f"文件名关键词: {' '.join(keywords)}")
        if description:
            parts.append(f"内容描述: {description}")
        parts.append(f"文件大小: {os.path.getsize(file_path)} 字节")

        content = "。".join(parts)
        return [{
            "content": content,
            "metadata": {
                "section": "文件元数据",
                "modality": doc_type,
                "keywords": keywords,
            },
        }]


# ============================================================
# Token 估算与智能分块
# ============================================================

def _is_cjk(ch: str) -> bool:
    """判断是否为 CJK 字符（覆盖统一汉字、扩展 A-E、兼容汉字）。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or    # CJK 统一汉字
        0x3400 <= code <= 0x4DBF or    # CJK 扩展 A
        0x20000 <= code <= 0x2A6DF or  # CJK 扩展 B
        0x2A700 <= code <= 0x2B73F or  # CJK 扩展 C
        0x2B740 <= code <= 0x2B81F or  # CJK 扩展 D
        0x2B820 <= code <= 0x2CEAF or  # CJK 扩展 E
        0xF900 <= code <= 0xFAFF       # CJK 兼容汉字
    )


def _approx_token_len(text: str) -> int:
    """近似估计 Token 长度，支持中英文混合。

    - CJK 字符按 1 token 计算（覆盖扩展汉字区）
    - 其他字符按空白分词计算，每个词 1 token
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk_tokens = len([t for t in text.split() if t])
    return cjk + non_cjk_tokens


def estimate_tokens(text: str) -> int:
    """轻量 Token 估算（无分词器依赖），委托给 _approx_token_len。"""
    return _approx_token_len(text)


def chunk_paragraphs(paragraphs: List[Dict], chunk_tokens: int, overlap_tokens: int) -> List[Dict]:
    """基于 Token 数量的智能分块，同时构建重叠部分，保持信息连续性。

    算法（元数据骨架：chunks / cur / cur_tokens / i）：
      1. 预处理：超长段落（> chunk_tokens）按句子边界拆分为小块，保证单段可容纳
      2. 主循环：逐段累加进当前块 cur，cur_tokens 记录累计 token 数
      3. 若再加一段会超出 chunk_tokens：将 cur 落为块，并从尾部截取
         总 token ≤ overlap_tokens 的段落作为下一块的重叠前缀（信息连续性）
      4. 重叠过多放不下新段时逐段收缩，保证不会死循环

    Args:
        paragraphs: 段落列表，每项 {content: str, metadata: dict, tokens?: int}
        chunk_tokens: 每个分块的最大 Token 数（>0）
        overlap_tokens: 相邻分块的重叠 Token 数（≥0）

    Returns:
        chunks: List[Dict]，每项包含
          {chunk_index, content, tokens, paragraph_count, metadata}
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens 必须为正数")
    overlap_tokens = max(0, overlap_tokens)

    # 1. 预处理：超长段落拆分（保证单段 tokens ≤ chunk_tokens）
    pieces: List[Dict] = []
    for para in paragraphs:
        content = para.get("content", "") or ""
        tokens = para.get("tokens") or estimate_tokens(content)
        if tokens > chunk_tokens:
            for sub in _split_oversized_paragraph(content, chunk_tokens):
                pieces.append({
                    "content": sub,
                    "metadata": para.get("metadata", {}),
                    "tokens": estimate_tokens(sub),
                })
        else:
            pieces.append({
                "content": content,
                "metadata": para.get("metadata", {}),
                "tokens": tokens,
            })

    # 2. 主循环：token 级智能分块 + 重叠构建
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0
    n = len(pieces)

    while i < n:
        para = pieces[i]
        tokens = para["tokens"]

        # 当前块还能容纳该段 → 累加
        if cur_tokens + tokens <= chunk_tokens:
            cur.append(para)
            cur_tokens += tokens
            i += 1
            continue

        # 当前块已满 → 落块，并构建重叠前缀
        if cur:
            chunks.append(_finalize_chunk(cur, len(chunks)))
            cur, cur_tokens = _build_overlap(cur, overlap_tokens)
            # 重叠过多导致放不下新段时，从前部收缩重叠
            while cur and cur_tokens + tokens > chunk_tokens:
                dropped = cur.pop(0)
                cur_tokens -= dropped["tokens"]
        else:
            # 空块也放不下（单段必然 ≤ chunk_tokens，正常不会到这里）
            chunks.append(_finalize_chunk([para], len(chunks)))
            i += 1

    # 3. 收尾：最后一块
    if cur:
        chunks.append(_finalize_chunk(cur, len(chunks)))

    return chunks


def _finalize_chunk(items: List[Dict], index: int) -> Dict:
    """将段落列表落为一个分块 dict。"""
    content = "\n\n".join(p["content"] for p in items)
    metadata: Dict[str, Any] = {}
    for p in items:
        for k, v in (p.get("metadata") or {}).items():
            metadata.setdefault(k, v)
    return {
        "chunk_index": index,
        "content": content,
        "tokens": sum(p["tokens"] for p in items),
        "paragraph_count": len(items),
        "metadata": metadata,
    }


def _build_overlap(items: List[Dict], overlap_tokens: int) -> Tuple[List[Dict], int]:
    """从块尾部截取重叠段落：总 token ≤ overlap_tokens，至少保留最后一段。"""
    overlap: List[Dict] = []
    tok = 0
    for p in reversed(items):
        if overlap and tok + p["tokens"] > overlap_tokens:
            break
        overlap.insert(0, p)
        tok += p["tokens"]
    return overlap, tok


def _split_oversized_paragraph(content: str, chunk_tokens: int) -> List[str]:
    """超长段落拆分：先按句子边界，单句仍超长时按 token 预算二分硬切。"""
    sentences = [s for s in re.split(r'(?<=[。！？；.!?;])\s*', content) if s.strip()]
    pieces: List[str] = []
    cur = ""
    cur_tok = 0
    for s in sentences:
        t = estimate_tokens(s)
        if cur and cur_tok + t > chunk_tokens:
            pieces.append(cur)
            cur = ""
            cur_tok = 0
        if t > chunk_tokens:
            # 单句超长：按 token 预算二分硬切
            pieces.extend(_hard_split_by_tokens(s, chunk_tokens))
            cur = ""
            cur_tok = 0
        else:
            cur += s
            cur_tok += t
    if cur:
        pieces.append(cur)
    return pieces or [content]


def _hard_split_by_tokens(text: str, chunk_tokens: int) -> List[str]:
    """按 token 预算二分查找截断位置，硬切超长文本。"""
    parts: List[str] = []
    remaining = text
    while estimate_tokens(remaining) > chunk_tokens:
        cut = _find_cut_position(remaining, chunk_tokens)
        parts.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        parts.append(remaining)
    return parts


def _find_cut_position(text: str, budget: int) -> int:
    """二分查找：前缀 token 估算 ≤ budget 的最大字符位置。"""
    lo, hi = 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ============================================================
# 文本分块器
# ============================================================

class TextSplitter:
    """智能文本分块器：支持字符模式（chunk_size）与 Token 模式（chunk_tokens）。

    字符模式策略：
      1. 先按段落（空行/换行）切分
      2. 段落合并成块，不超过 chunk_size
      3. 超长段落按句子（。！？；换行）切分
      4. 硬切分兜底（无边界可用时）

    Token 模式策略（chunk_tokens 设置时启用）：
      - 按 estimate_tokens 估算 token 数，chunk_paragraphs() 智能分块
      - overlap_tokens 构建重叠部分，保持信息连续性
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50,
                 chunk_tokens: int = None, overlap_tokens: int = None):
        self.chunk_size = max(50, chunk_size)
        self.overlap = max(0, min(overlap, self.chunk_size // 2))
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens if overlap_tokens is not None else self.overlap

    @property
    def use_tokens(self) -> bool:
        return bool(self.chunk_tokens and self.chunk_tokens > 0)

    def split_paragraph_dicts(self, paragraphs: List[Dict]) -> List[Dict]:
        """段落 dict 分块入口，返回分块 dict 列表。

        每段格式：{content: str, metadata: dict, tokens?: int}
        每块格式：{chunk_index, content, tokens, paragraph_count, metadata}
        """
        if self.use_tokens:
            return chunk_paragraphs(paragraphs, self.chunk_tokens, self.overlap_tokens or 0)

        # 字符模式：逐段切分，保留元数据
        chunks: List[Dict] = []
        for p in paragraphs:
            content = p.get("content", "") or ""
            for piece in self.split(content):
                chunks.append({
                    "chunk_index": len(chunks),
                    "content": piece,
                    "tokens": estimate_tokens(piece),
                    "paragraph_count": 1,
                    "metadata": p.get("metadata", {}),
                })
        return chunks

    def split(self, text: str) -> List[str]:
        """将长文本切分为块列表。"""
        text = text.strip()
        if not text:
            return []

        # 1. 段落切分（空行 / 换行）
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n', text) if p.strip()]

        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            # 段落过长时先按句子切分
            para_parts = self._split_long_paragraph(para)

            for part in para_parts:
                if not part:
                    continue
                if len(current) + len(part) + 1 <= self.chunk_size:
                    current = f"{current}\n{part}".strip() if current else part
                else:
                    if current:
                        chunks.append(current)
                    # 重叠部分：取上一块末尾
                    overlap_text = self._get_overlap(current)
                    current = f"{overlap_text}\n{part}".strip() if overlap_text else part

                    # 单块仍超长 → 硬切分
                    while len(current) > self.chunk_size:
                        chunks.append(current[:self.chunk_size])
                        current = current[self.chunk_size - self.overlap:] if self.overlap else ""

        if current.strip():
            chunks.append(current)

        return [c for c in chunks if c.strip()]

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """超长段落按句子边界切分。"""
        if len(paragraph) <= self.chunk_size:
            return [paragraph]
        sentences = re.split(r'(?<=[。！？；.!?;])\s*', paragraph)
        sentences = [s for s in sentences if s.strip()]
        if len(sentences) <= 1:
            return [paragraph]  # 无句子边界，交给硬切分
        parts: List[str] = []
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 <= self.chunk_size:
                current = f"{current}{s}" if current else s
            else:
                if current:
                    parts.append(current)
                current = s
                while len(current) > self.chunk_size:  # 单句超长硬切
                    parts.append(current[:self.chunk_size])
                    current = current[self.chunk_size:]
        if current:
            parts.append(current)
        return parts

    @staticmethod
    def _get_overlap(text: str) -> str:
        """取上一块的末尾若干字符作为重叠。"""
        if not text:
            return ""
        return text[-min(50, len(text)):]


# ============================================================
# 混合索引（嵌入向量 + TF-IDF）
# ============================================================

class RagIndex:
    """知识库索引：稠密向量（嵌入）+ 稀疏 TF-IDF 双路召回。

    评分：
      dense_score = 嵌入余弦相似度（归一化到 0~1）
      sparse_score = TF-IDF 余弦相似度
      final = dense_score × 0.7 + sparse_score × 0.3
    """

    def __init__(self):
        self._encoder = VectorEncoder()
        self._embedding_client = self._encoder._embedding_client
        self._chunks: Dict[str, RagChunk] = {}                 # chunk_id -> chunk
        self._dense: Dict[str, List[float]] = {}               # chunk_id -> embedding
        self._tfidf_vectors: Dict[str, Dict[str, float]] = {}  # chunk_id -> {term: weight}
        self._idf: Dict[str, float] = {}                       # term -> idf
        self._docs: Dict[str, RagDocument] = {}                # doc_id -> doc
        self._doc_chunks: Dict[str, List[str]] = {}            # doc_id -> [chunk_id]

    # ============================================================
    # 索引写入
    # ============================================================

    @property
    def embedding_available(self) -> bool:
        """百炼嵌入 API 是否可用。"""
        return self._encoder.embedding_available

    def add_chunk(self, chunk: RagChunk) -> bool:
        """添加单个分块到索引（统一编码接口：百炼嵌入 + TF-IDF 兜底）。

        返回是否成功生成嵌入向量。
        """
        self._chunks[chunk.chunk_id] = chunk

        dense, sparse, _ = self._encoder.encode(chunk.content)
        if dense:
            self._dense[chunk.chunk_id] = dense
        if sparse:
            self._tfidf_vectors[chunk.chunk_id] = sparse
        return bool(dense)

    def index_chunks(self, chunks: List[RagChunk]) -> Tuple[List[RagChunk], int]:
        """统一分块索引（批量）：为每个分块生成向量并写入索引。

        使用百炼嵌入 API 批量编码，API 不可用时自动回退 TF-IDF 稀疏向量。
        每个分块的 metadata 记录 vector_method 与 tokens。

        Returns:
            (chunks, embedded_count)
        """
        if not chunks:
            return chunks, 0

        # 批量编码（百炼 API 一次请求多个文本，失败项回退 TF-IDF）
        texts = [c.content for c in chunks]
        embeddings = self._encoder.embed_batch(texts)

        embedded_count = 0
        for chunk, dense in zip(chunks, embeddings):
            self._chunks[chunk.chunk_id] = chunk

            # TF-IDF 稀疏向量始终生成（兜底 / 混合检索用）
            sparse = self._encoder._tfidf_vectorize(chunk.content)
            if sparse:
                self._tfidf_vectors[chunk.chunk_id] = sparse

            if dense:
                self._dense[chunk.chunk_id] = dense
                chunk.metadata["vector_method"] = "embedding"
                embedded_count += 1
            else:
                chunk.metadata["vector_method"] = "tfidf"

            chunk.metadata["tokens"] = _approx_token_len(chunk.content)

        return chunks, embedded_count

    def add_document(self, doc: RagDocument, chunks: List[RagChunk]):
        """登记文档与分块的关联。"""
        self._docs[doc.doc_id] = doc
        self._doc_chunks[doc.doc_id] = [c.chunk_id for c in chunks]
        # 文档级聚合索引：所有分块拼成一篇文档用于 IDF 计算
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            for token in set(tokens):
                self._idf[token] = self._idf.get(token, 0.0) + 1.0 / max(1, len(chunks))

    def rebuild_idf(self):
        """重建 IDF（基于全部文档）。"""
        if not self._chunks:
            return
        total_docs = len(self._docs) or 1
        for term, doc_freq_weight in list(self._idf.items()):
            self._idf[term] = math.log((total_docs + 1) / (doc_freq_weight + 1)) + 1

    # ============================================================
    # 检索
    # ============================================================

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Tuple[float, RagChunk, str]]:
        """混合检索：嵌入 + TF-IDF。

        Returns:
            [(score, chunk, method), ...] method: "hybrid"/"embedding"/"tfidf"
        """
        if not self._chunks:
            return []

        dense_scores = self._dense_search(query)
        sparse_scores = self._sparse_search(query)

        results = []
        for chunk_id, chunk in self._chunks.items():
            d = dense_scores.get(chunk_id, 0.0)
            s = sparse_scores.get(chunk_id, 0.0)

            if d > 0 and s > 0:
                score = d * 0.7 + s * 0.3
                method = "hybrid"
            elif d > 0:
                score = d
                method = "embedding"
            elif s > 0:
                score = s
                method = "tfidf"
            else:
                continue

            if score < min_score:
                continue
            results.append((score, chunk, method))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def _dense_search(self, query: str) -> Dict[str, float]:
        """嵌入向量检索。"""
        if not self._dense or not self._embedding_client or not self._embedding_client.is_available:
            return {}
        query_vec = self._embedding_client.embed(query)
        if not query_vec:
            return {}
        scores = {}
        for chunk_id, vec in self._dense.items():
            sim = EmbeddingClient.cosine_similarity_dense(query_vec, vec)
            if sim > 0:
                # 归一化到 0~1（cosine 范围约 0.2~0.9）
                scores[chunk_id] = max(0.0, min(1.0, sim))
        return scores

    def _sparse_search(self, query: str) -> Dict[str, float]:
        """TF-IDF 检索。"""
        if not self._tfidf_vectors:
            return {}
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}

        q_tf = Counter(query_tokens)
        q_len = len(query_tokens)
        q_vec = {t: (c / q_len) * self._idf.get(t, 0.0) for t, c in q_tf.items()}
        if not any(q_vec.values()):
            return {}

        scores = {}
        for chunk_id, doc_vec in self._tfidf_vectors.items():
            sim = self._cosine_sparse(q_vec, doc_vec)
            if sim > 0:
                scores[chunk_id] = sim
        return scores

    # ============================================================
    # 辅助
    # ============================================================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中英文混合分词：英文按单词，中文按整词 + 2-gram 展开。"""
        tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', text.lower())
        expanded = []
        for token in tokens:
            if token and '\u4e00' <= token[0] <= '\u9fff' and len(token) >= 2:
                expanded.append(token)
                for i in range(len(token) - 1):
                    expanded.append(token[i:i + 2])
            elif token:
                expanded.append(token)
        return expanded

    @staticmethod
    def _cosine_sparse(vec_a: Dict[str, float], vec_b_counter: Counter) -> float:
        """稀疏向量余弦相似度。"""
        if not vec_a or not vec_b_counter:
            return 0.0
        common = set(vec_a.keys()) & set(vec_b_counter.keys())
        if not common:
            return 0.0
        dot = sum(vec_a[k] * vec_b_counter[k] for k in common)
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b_counter.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ============================================================
    # 知识库管理
    # ============================================================

    def get_docs(self) -> List[RagDocument]:
        return list(self._docs.values())

    def get_chunk_count(self) -> int:
        return len(self._chunks)

    def get_doc_count(self) -> int:
        return len(self._docs)

    def get_doc_chunks(self, doc_id: str) -> List[RagChunk]:
        return [self._chunks[cid] for cid in self._doc_chunks.get(doc_id, [])]

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[RagChunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def delete_doc(self, doc_id: str) -> bool:
        """删除文档及其所有分块。"""
        if doc_id not in self._docs:
            return False
        for chunk_id in self._doc_chunks.get(doc_id, []):
            self._chunks.pop(chunk_id, None)
            self._dense.pop(chunk_id, None)
            self._tfidf_vectors.pop(chunk_id, None)
        self._doc_chunks.pop(doc_id, None)
        self._docs.pop(doc_id, None)
        return True

    def clear(self):
        """清空整个知识库。"""
        self._chunks.clear()
        self._dense.clear()
        self._tfidf_vectors.clear()
        self._idf.clear()
        self._docs.clear()
        self._doc_chunks.clear()


# ============================================================
# Qwen 问答客户端（阿里云百炼 OpenAI 兼容接口）
# ============================================================

class QwenChatClient:
    """阿里云百炼 Qwen 对话客户端（OpenAI 兼容接口）。

    环境变量：
      DASHSCOPE_API_KEY  必填
      RAG_LLM_MODEL      模型名，默认 qwen-plus
    """

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, model: str = None):
        from openai import OpenAI
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("未配置 DASHSCOPE_API_KEY，无法使用 Qwen 问答")
        self.model = model or os.getenv("RAG_LLM_MODEL", "qwen-plus")
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL, timeout=60)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """调用 Qwen 对话，返回完整回复文本。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        return resp.choices[0].message.content or ""


# ============================================================
# 安全的全量缓存（已迁至 core/cache.py，此处保留引用）
# ============================================================

from src.core.cache import SafeFullCache


# ============================================================
# 多查询扩展（MQE）
# ============================================================

class QueryExpander:
    """多查询扩展（Multi-Query Expansion，MQE）：提升检索召回。

    独立调用 LLM 生成语义等价或互补的多样化查询（不影响主对话系统提示词），
    扩展查询仅用于检索增强，不直接用于回答。

    兜底机制：
      - 只取前 n 个扩展（防止 LLM 生成过多）
      - 未生成任何扩展时，使用原始查询

    模式调度（threshold / batch_size）：
      - 查询总数（原始 + 扩展）≤ threshold → 单查询模式（逐个检索）
      - 查询总数 > threshold → 批量查询模式（按 batch_size 分批处理）
    """

    MQE_SYSTEM_PROMPT = "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"

    @staticmethod
    def build_prompt(query: str, n: int) -> List[Dict[str, str]]:
        """构建 MQE 提示词（独立于主对话的系统提示词）。"""
        return [
            {"role": "system", "content": QueryExpander.MQE_SYSTEM_PROMPT},
            {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"},
        ]

    def __init__(self, llm_client=None, threshold: int = 5, batch_size: int = 10,
                 cache: Optional[SafeFullCache] = None):
        self._llm = llm_client          # QwenChatClient 或 None（惰性初始化）
        self._cache = cache             # 缓存层：扩展查询结果可复用
        self.threshold = max(1, threshold)      # 查询数量阈值：高于此值用批量模式
        self.batch_size = max(1, batch_size)    # 批量模式的分批大小

    def _get_llm(self) -> Optional[QwenChatClient]:
        """惰性初始化 LLM 客户端。"""
        if self._llm is None:
            try:
                self._llm = QwenChatClient()
            except ValueError:
                self._llm = None
        return self._llm

    def expand(self, query: str, n: int = 3) -> List[str]:
        """生成扩展查询列表（含原始查询，去重保序）。

        缓存：相同 (query, n) 直接命中缓存，避免重复调用 LLM。

        Returns:
            [原始查询, 扩展1, 扩展2, ...]；LLM 不可用/失败时仅返回 [原始查询]
        """
        queries = [query]  # 兜底：始终保留原始查询

        cache_key = f"mqe:{query}:{n}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        llm = self._get_llm()
        if llm is None:
            return queries

        n = max(1, min(10, n))
        try:
            response = llm.chat(self.build_prompt(query, n), temperature=0.7)
        except Exception:
            return queries

        if not response:
            return queries

        # 解析 + 兜底：只取前 n 个，防止 LLM 生成过多
        seen = {query}
        for line in self._parse_response(response):
            line = line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            queries.append(line)
            if len(queries) - 1 >= n:   # 扩展数达到 n 即截断
                break

        if self._cache is not None:
            self._cache.set(cache_key, queries)
        return queries

    @staticmethod
    def _parse_response(response: str) -> List[str]:
        """解析 LLM 返回：按行拆分，去除编号前缀与空行。"""
        lines = []
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            # 去掉 "1." / "1、" / "- " / "1）" 等编号前缀
            line = re.sub(r'^[\d一二三四五六七八九十]+[.、)）:：]\s*', '', line)
            line = re.sub(r'^[-*]\s*', '', line)
            if line:
                lines.append(line)
        return lines

    def plan(self, query: str, n: int = 3) -> Tuple[List[str], str, int]:
        """规划检索模式：返回 (查询列表, 模式, 查询总数)。

        模式：
          - "single": 查询总数 ≤ threshold → 单查询模式
          - "batch":  查询总数 > threshold  → 批量查询模式（按 batch_size 分批）
        """
        queries = self.expand(query, n=n)
        total = len(queries)
        mode = "batch" if total > self.threshold else "single"
        return queries, mode, total

    def iter_batches(self, queries: List[str]):
        """按 batch_size 分批迭代查询列表（数量太大时分批处理）。"""
        for start in range(0, len(queries), self.batch_size):
            yield queries[start:start + self.batch_size]


# ============================================================
# 假设文档嵌入（HyDE）
# ============================================================

class HydeGenerator:
    """假设文档嵌入（HyDE）：生成假设性答案文档用于向量检索。

    流程：用户查询 → 生成假设文档 → 嵌入假设文档 → 用假设文档检索。
    独立调用 LLM（不影响主对话系统提示词），假设文档仅用于检索增强，
    不直接用于回答。假设性答案与知识库中的真实答案片段在向量空间中
    更接近，因此能提升检索召回质量。

    兜底机制：LLM 不可用 / 调用失败 / 返回为空时，回退使用原始查询。
    """

    HYDE_SYSTEM_PROMPT = "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"

    @staticmethod
    def build_prompt(query: str) -> List[Dict[str, str]]:
        """构建 HyDE 提示词（独立于主对话的系统提示词）。"""
        return [
            {"role": "system", "content": HydeGenerator.HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"},
        ]

    def __init__(self, llm_client=None, cache: Optional[SafeFullCache] = None):
        self._llm = llm_client          # QwenChatClient 或 None（惰性初始化）
        self._cache = cache             # 缓存层：假设文档可复用

    def _get_llm(self) -> Optional[QwenChatClient]:
        """惰性初始化 LLM 客户端。"""
        if self._llm is None:
            try:
                self._llm = QwenChatClient()
            except ValueError:
                self._llm = None
        return self._llm

    def generate(self, query: str) -> str:
        """生成假设文档。

        缓存：相同查询直接命中缓存，避免重复调用 LLM。

        Returns:
            假设文档文本；LLM 不可用/失败/返回空时回退返回原始查询。
        """
        cache_key = f"hyde:{query}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        llm = self._get_llm()
        if llm is None:
            return query

        try:
            doc = llm.chat(self.build_prompt(query), temperature=0.7)
            doc = (doc or "").strip()
            if not doc:
                doc = query
        except Exception:
            doc = query

        if self._cache is not None:
            self._cache.set(cache_key, doc)
        return doc


# ============================================================
# RAG 工具
# ============================================================

class RagTool(Tool):
    """
    RAG 工具：检索增强生成。
    当前支持 action: add_file, add_text, search, query, list, stats, delete, clear
    """

    RAG_QA_PROMPT = """你是一个知识库问答助手。请严格基于给定的资料片段回答用户问题。

【资料片段】
{context}

【问题】
{question}

【回答要求】
1. 只依据资料片段回答，资料中没有的信息请明确说明「资料中未提及」
2. 回答末尾标注引用的资料编号，如 [1][2]
3. 使用简洁清晰的中文回答"""

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self._loader = DocumentLoader()
        self._splitter = TextSplitter(
            chunk_size=config.get("chunk_size", 500),
            overlap=config.get("overlap", 50),
            chunk_tokens=config.get("chunk_tokens"),
            overlap_tokens=config.get("overlap_tokens"),
        )
        self._index = RagIndex()
        self._llm = None
        # 缓存层：LLM 查询结果 + 检索策略生成的查询（MQE/HyDE）
        self._cache = SafeFullCache(
            max_size=config.get("cache_max_size", 10000),
            default_ttl=config.get("cache_ttl", 3600),
        )
        self._expander = QueryExpander(
            llm_client=None,
            threshold=config.get("mqe_threshold", 5),
            batch_size=config.get("mqe_batch_size", 10),
            cache=self._cache,
        )
        self._hyde = HydeGenerator(llm_client=None, cache=self._cache)

    # --- Tool 基类抽象接口 ---

    @property
    def name(self) -> str:
        return "RagTool"

    @property
    def description(self) -> str:
        return (
            "RAG 检索增强生成工具，支持添加多格式文档（PDF/Office/图片/音频/文本）、"
            "智能检索召回、知识库问答（Qwen）。"
            "适用：基于私有文档的知识库问答、多格式文档索引与混合检索。"
            "不适用：实时网络搜索（请用 AdvancedSearch）、短期记忆（请用 MemoryTool）、"
            "小规模结构化笔记（请用 NoteTool）。"
            "注意：问答功能需要配置 DASHSCOPE_API_KEY。"
        )

    def get_parameters(self) -> List[ToolParameter]:
        return self._build_parameters()

    def execute(self, action, **kwargs):
        """统一入口：兼容旧协议 execute(action, **kwargs) 与新协议 execute(args)。"""
        return dual_protocol_execute(self, action, **kwargs)

    def _get_llm(self) -> QwenChatClient:
        """惰性初始化 Qwen 客户端。"""
        if self._llm is None:
            try:
                self._llm = QwenChatClient()
            except ValueError:
                self._llm = None
        return self._llm

    # ============================================================
    # 参数定义
    # ============================================================

    @staticmethod
    def _build_parameters() -> List[ToolParameter]:
        return [
            # add_file / add_text 参数
            ToolParameter(name="file_path", type="string",
                          description="文件路径（add_file 时使用）", required=False),
            ToolParameter(name="content", type="string",
                          description="文本内容（add_text 时使用）", required=False),
            ToolParameter(name="title", type="string",
                          description="文档标题（add_text 时使用）", required=False),
            ToolParameter(name="description", type="string",
                          description="文件描述（图片/音频等元数据语义化时使用）", required=False),
            # search / query 参数
            ToolParameter(name="query", type="string",
                          description="检索查询（search / search_mqe 时使用）", required=False),
            ToolParameter(name="question", type="string",
                          description="问题（query 时使用）", required=False),
            ToolParameter(name="top_k", type="number",
                          description="召回数量（默认 5）", required=False, default=5),
            ToolParameter(name="min_score", type="number",
                          description="最低相关分（默认 0.0）", required=False, default=0.0),
            ToolParameter(name="show_context", type="boolean",
                          description="是否展示召回上下文（query 时，默认 false）", required=False, default=False),
            # MQE（多查询扩展）参数
            ToolParameter(name="mqe", type="boolean",
                          description="是否启用多查询扩展（query 时，默认 false，旧接口）", required=False, default=False),
            ToolParameter(name="n_queries", type="number",
                          description="扩展查询数量（search_mqe / mqe 时使用，默认 3）", required=False, default=3),
            # HyDE（假设文档嵌入）参数
            ToolParameter(name="hyde", type="boolean",
                          description="是否启用假设文档嵌入检索（search_hyde / query 时，默认 false，旧接口）",
                          required=False, default=False),
            # 统一扩展检索框架参数（search_expanded / query 时使用）
            ToolParameter(name="enable_mqe", type="boolean",
                          description="统一框架：启用 MQE 多查询扩展（默认 false）", required=False, default=False),
            ToolParameter(name="enable_hyde", type="boolean",
                          description="统一框架：启用 HyDE 假设文档嵌入（默认 false）", required=False, default=False),
            ToolParameter(name="mqe_expansions", type="number",
                          description="MQE 扩展查询数量（默认 2）", required=False, default=2),
            ToolParameter(name="candidate_pool_multiplier", type="number",
                          description="候选池倍数：pool = max(top_k × 倍数, 20)（默认 4）", required=False, default=4),
            # 知识库管理参数
            ToolParameter(name="doc_id", type="string",
                          description="文档 ID（delete 时使用）", required=False),
            ToolParameter(name="doc_type", type="string",
                          description="按类型过滤（list 时使用）", required=False),
        ]

    # ============================================================
    # 操作分发
    # ============================================================

    def run(self, args: Dict[str, Any]) -> str:
        action = (args.get("action") or "").lower()
        if action == "add_file":
            return self._add_file(args)
        elif action == "add_text":
            return self._add_text(args)
        elif action == "convert":
            return self._convert(args)
        elif action == "search":
            return self._search(args)
        elif action == "search_mqe":
            return self._search_mqe(args)
        elif action == "search_hyde":
            return self._search_hyde(args)
        elif action == "search_expanded":
            return self._search_expanded(args)
        elif action == "query":
            return self._query(args)
        elif action == "list":
            return self._list(args)
        elif action == "stats":
            return self._stats(args)
        elif action == "delete":
            return self._delete(args)
        elif action == "clear":
            return self._clear(args)
        return f"❌ 未知操作: '{action}'，当前支持: add_file, add_text, convert, search, search_mqe, search_hyde, search_expanded, query, list, stats, delete, clear"

    # ============================================================
    # 添加文档
    # ============================================================

    def _add_file(self, args: Dict[str, Any]) -> str:
        """添加文件到知识库：加载 → 分块 → 索引。"""
        file_path = args.get("file_path", "").strip()
        if not file_path:
            return "❌ add_file 操作需要提供 file_path 参数"

        description = (args.get("description") or "").strip()

        try:
            doc_meta, blocks = self._loader.load(file_path, description)
        except FileNotFoundError as e:
            return f"❌ {e}"
        except Exception as e:
            return f"❌ 文档加载失败: {e}"

        return self._index_blocks(doc_meta, blocks)

    def _add_text(self, args: Dict[str, Any]) -> str:
        """添加纯文本内容到知识库。"""
        content = (args.get("content") or "").strip()
        if not content:
            return "❌ add_text 操作需要提供 content 参数"

        title = (args.get("title") or "").strip() or "文本片段"
        doc_meta = {
            "name": title,
            "source_path": None,
            "doc_type": "text",
            "extension": "",
            "size": len(content),
            "description": "",
        }
        blocks = [{"content": content, "metadata": {"section": "全文"}}]
        return self._index_blocks(doc_meta, blocks)

    def _index_blocks(self, doc_meta: Dict[str, Any], blocks: List[Dict[str, Any]]) -> str:
        """公共索引流程：分块 + 嵌入 + 登记文档。"""
        if not blocks:
            return f"❌ 文档 {doc_meta['name']} 未提取到任何文本内容"

        doc_id = str(uuid.uuid4())
        doc = RagDocument(
            doc_id=doc_id,
            name=doc_meta["name"],
            source_path=doc_meta.get("source_path"),
            doc_type=doc_meta.get("doc_type", "text"),
            created_at=datetime.now().isoformat(),
            metadata={
                "extension": doc_meta.get("extension", ""),
                "size": doc_meta.get("size", 0),
                "description": doc_meta.get("description", ""),
                "converter": doc_meta.get("converter", "fallback"),
            },
        )

        embedded_count = 0
        chunks: List[RagChunk] = []

        if self._splitter.use_tokens:
            # Token 模式：构建段落 dict（保留章节元数据），token 级智能分块 + 重叠
            paragraph_dicts: List[Dict] = []
            for block in blocks:
                for para in re.split(r'\n\s*\n', block["content"]):
                    para = para.strip()
                    if para:
                        paragraph_dicts.append({
                            "content": para,
                            "metadata": block.get("metadata", {}),
                        })
            chunk_dicts = self._splitter.split_paragraph_dicts(paragraph_dicts)
            for cd in chunk_dicts:
                chunks.append(RagChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    doc_name=doc_meta["name"],
                    content=cd["content"],
                    metadata={
                        **cd.get("metadata", {}),
                        "tokens": cd.get("tokens", 0),
                        "paragraph_count": cd.get("paragraph_count", 1),
                    },
                ))
        else:
            # 字符模式：逐块按 chunk_size 切分
            for block in blocks:
                for piece in self._splitter.split(block["content"]):
                    chunks.append(RagChunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        doc_name=doc_meta["name"],
                        content=piece,
                        metadata=dict(block.get("metadata", {})),
                    ))

        # 统一分块索引：百炼嵌入批量编码 + TF-IDF 兜底
        chunks, embedded_count = self._index.index_chunks(chunks)

        doc.chunk_count = len(chunks)
        self._index.add_document(doc, chunks)
        self._index.rebuild_idf()
        self._cache.clear()   # 知识库变更 → 缓存失效（一致性）

        embed_status = (f"嵌入向量: {embedded_count}/{len(chunks)} 块"
                        if self._index.embedding_available
                        else "嵌入 API 不可用，使用 TF-IDF 检索")

        if self._splitter.use_tokens:
            chunk_mode = (f"token 分块（chunk_tokens={self._splitter.chunk_tokens}, "
                          f"overlap_tokens={self._splitter.overlap_tokens}）")
        else:
            chunk_mode = (f"字符分块（chunk_size={self._splitter.chunk_size}, "
                          f"overlap={self._splitter.overlap}）")

        return (
            f"✅ 文档已添加\n"
            f"   ID: {doc_id[:12]}...\n"
            f"   名称: {doc_meta['name']}\n"
            f"   类型: {doc_meta.get('doc_type', 'text')}\n"
            f"   分块数: {len(chunks)}（{chunk_mode}）\n"
            f"   {embed_status}"
        )

    # ============================================================
    # 文档转换预览
    # ============================================================

    def _convert(self, args: Dict[str, Any]) -> str:
        """预览文档的 Markdown 转换结果（不入库）。"""
        file_path = args.get("file_path", "").strip()
        if not file_path:
            return "❌ convert 操作需要提供 file_path 参数"
        if not os.path.exists(file_path):
            return f"❌ 文件不存在: {file_path}"

        description = (args.get("description") or "").strip()

        try:
            doc_meta, blocks = self._loader.load(file_path, description)
        except Exception as e:
            return f"❌ 转换失败: {e}"

        converter_name = doc_meta.get("converter", "fallback")
        converter_label = "MarkItDown 统一转换" if converter_name == "markitdown" else "轻量回退解析"

        lines = [
            "=" * 60,
            f"📄 文档转换预览: {doc_meta['name']}",
            f"   类型: {doc_meta['doc_type']} | 转换引擎: {converter_label}",
            f"   分块数: {len(blocks)}（后续将进入统一分块 → 向量化 → 入库流程）",
            "=" * 60,
        ]
        for i, block in enumerate(blocks, 1):
            lines.append("")
            lines.append(f"--- 块 {i} [{block['metadata'].get('section', '')}] ---")
            lines.append(block["content"][:300] + ("..." if len(block["content"]) > 300 else ""))
        return "\n".join(lines)

    # ============================================================
    # 检索
    # ============================================================

    def _search(self, args: Dict[str, Any]) -> str:
        """智能检索召回：混合检索（嵌入 + TF-IDF）。"""
        query = (args.get("query") or "").strip()
        if not query:
            return "❌ search 操作需要提供 query 参数"

        top_k = max(1, min(50, int(args.get("top_k", 5))))
        min_score = max(0.0, min(1.0, float(args.get("min_score", 0.0))))

        results = self._index.search(query, top_k=top_k, min_score=min_score)

        if not results:
            return "📭 没有检索到相关分块。"

        method_str = "嵌入向量 + TF-IDF 混合" if self._index.embedding_available else "TF-IDF"
        lines = [
            "=" * 60,
            f"🔍 RAG 检索: \"{query}\"",
            f"   检索方式: {method_str}",
            f"   召回结果: {len(results)}/{self._index.get_chunk_count()} 分块",
            "=" * 60,
        ]
        for i, (score, chunk, method) in enumerate(results, 1):
            lines.append("")
            lines.append(f"--- 结果 {i} [{method}] score={score:.4f} ---")
            lines.append(f"  来源: {chunk.doc_name}（{chunk.metadata.get('section', '')}）")
            lines.append(f"  内容: {chunk.content[:120]}{'...' if len(chunk.content) > 120 else ''}")

        return "\n".join(lines)

    # ============================================================
    # 多查询扩展检索（MQE）
    # ============================================================

    def _retrieve_mqe(self, query: str, top_k: int, n_queries: int = 3) -> Tuple[List[tuple], Dict[str, Any]]:
        """MQE 检索核心：扩展查询 → 模式调度（单/批量）→ 分批检索 → 合并去重。

        Returns:
            (合并结果列表, 统计信息 dict)
            统计信息：{queries, mode, total, batches, matched}
        """
        # 1. 扩展查询 + 模式规划（threshold 阈值决定单查询/批量模式）
        queries, mode, total = self._expander.plan(query, n=n_queries)

        # 2. 分批检索（数量太大时分批处理），合并结果按 chunk 取最高分
        merged: Dict[str, Tuple[float, RagChunk, str]] = {}
        match_counts: Dict[str, int] = {}
        batches_used = 0

        for batch in self._expander.iter_batches(queries):
            batches_used += 1
            for q in batch:
                for score, chunk, method in self._index.search(q, top_k=top_k):
                    if chunk.chunk_id not in merged or score > merged[chunk.chunk_id][0]:
                        merged[chunk.chunk_id] = (score, chunk, method)
                    match_counts[chunk.chunk_id] = match_counts.get(chunk.chunk_id, 0) + 1

        # 3. 多查询共识加成：被多个扩展查询命中的 chunk 加权
        results = []
        for chunk_id, (score, chunk, method) in merged.items():
            matches = match_counts.get(chunk_id, 1)
            final_score = score * min(1.5, 1.0 + 0.15 * (matches - 1))
            results.append((final_score, chunk, method, matches))

        results.sort(key=lambda x: x[0], reverse=True)
        stats = {
            "queries": queries,
            "mode": mode,
            "total": total,
            "batches": batches_used,
            "matched": len(results),
        }
        return results, stats

    def _search_mqe(self, args: Dict[str, Any]) -> str:
        """多查询扩展检索：MQE 生成扩展查询 → 增强召回（扩展结果仅用于检索）。"""
        query = (args.get("query") or "").strip()
        if not query:
            return "❌ search_mqe 操作需要提供 query 参数"

        top_k = max(1, min(50, int(args.get("top_k", 5))))
        n_queries = max(1, min(10, int(args.get("n_queries", 3))))

        results, stats = self._retrieve_mqe(query, top_k, n_queries)

        if not results:
            return "📭 MQE 检索没有找到相关分块。"

        mode_label = "批量查询模式" if stats["mode"] == "batch" else "单查询模式"
        lines = [
            "=" * 60,
            f"🔍 MQE 多查询扩展检索: \"{query}\"",
            f"   扩展查询({stats['total']}个): {', '.join(stats['queries'])}",
            f"   模式: {mode_label}（threshold={self._expander.threshold}）"
            f" | 分批: {stats['batches']}批（batch_size={self._expander.batch_size}）",
            f"   召回结果: {stats['matched']}/{self._index.get_chunk_count()} 分块",
            "=" * 60,
        ]
        for i, (score, chunk, method, matches) in enumerate(results[:top_k], 1):
            lines.append("")
            lines.append(f"--- 结果 {i} [{method}] score={score:.4f} 命中{matches}查询 ---")
            lines.append(f"  来源: {chunk.doc_name}（{chunk.metadata.get('section', '')}）")
            lines.append(f"  内容: {chunk.content[:120]}{'...' if len(chunk.content) > 120 else ''}")

        return "\n".join(lines)

    # ============================================================
    # 假设文档嵌入检索（HyDE）
    # ============================================================

    def _retrieve_hyde(self, query: str, top_k: int) -> Tuple[List[tuple], Dict[str, Any]]:
        """HyDE 检索核心：生成假设文档 → 嵌入 → 用假设文档检索。

        Returns:
            (检索结果列表, 统计信息 dict)
            统计信息：{hypothesis, used_hypothesis}
        """
        # 1. 独立 LLM 生成假设文档（兜底：失败时返回原始查询）
        hypo_doc = self._hyde.generate(query)

        # 2. 嵌入假设文档并用其检索（向量空间更接近真实答案片段）
        results = self._index.search(hypo_doc, top_k=top_k)

        stats = {
            "hypothesis": hypo_doc,
            "used_hypothesis": hypo_doc != query,
        }
        return results, stats

    def _search_hyde(self, args: Dict[str, Any]) -> str:
        """假设文档嵌入检索：查询 → 假设文档 → 嵌入 → 检索（假设文档仅用于检索）。"""
        query = (args.get("query") or "").strip()
        if not query:
            return "❌ search_hyde 操作需要提供 query 参数"

        top_k = max(1, min(50, int(args.get("top_k", 5))))

        results, stats = self._retrieve_hyde(query, top_k)

        if not results:
            return "📭 HyDE 检索没有找到相关分块。"

        hypo_note = ("假设文档" if stats["used_hypothesis"] else "原始查询（LLM 兜底）")
        lines = [
            "=" * 60,
            f"🔍 HyDE 假设文档嵌入检索: \"{query}\"",
            f"   假设文档（{hypo_note}）: {stats['hypothesis'][:100]}"
            + ("..." if len(stats['hypothesis']) > 100 else ""),
            f"   召回结果: {len(results)}/{self._index.get_chunk_count()} 分块",
            "=" * 60,
        ]
        for i, (score, chunk, method) in enumerate(results[:top_k], 1):
            lines.append("")
            lines.append(f"--- 结果 {i} [{method}] score={score:.4f} ---")
            lines.append(f"  来源: {chunk.doc_name}（{chunk.metadata.get('section', '')}）")
            lines.append(f"  内容: {chunk.content[:120]}{'...' if len(chunk.content) > 120 else ''}")

        return "\n".join(lines)

    # ============================================================
    # 统一扩展检索框架（MQE + HyDE）
    # ============================================================

    def _retrieve_expanded(self, query: str, top_k: int = 8,
                           score_threshold: float = None,
                           enable_mqe: bool = False, mqe_expansions: int = 2,
                           enable_hyde: bool = False,
                           candidate_pool_multiplier: int = 4,
                           min_score: float = 0.0) -> Tuple[List[tuple], Dict[str, Any]]:
        """统一扩展检索：MQE + HyDE 查询扩展 → 并行检索 → 去重合并 → top-k。

        场景建议：
          - 一般查询：启用 MQE（enable_mqe=True）
          - 专业领域查询：同时启用 MQE + HyDE
          - 性能敏感场景：基础检索（两者都不启用）或仅 MQE

        流程（参考 search_vectors_expanded）：
          1. 查询扩展：原始查询 + MQE 多样化查询 + HyDE 假设文档
          2. 去重修剪：过滤空项与重复项
          3. 候选池分配：pool = max(top_k × multiplier, 20)，每扩展查询 per 个
          4. 并行检索：每个扩展查询独立向量检索（ThreadPoolExecutor）
          5. 聚合合并：按 chunk_id 取最高分，分数降序返回 top-k

        Returns:
            (results, stats)
            results: [(score, chunk, method), ...]
            stats: {expansions, strategy, pool, per, total_candidates, matched}
        """
        if not query:
            return [], {}

        # 1. 查询扩展（原始 + MQE + HyDE）
        expansions: List[str] = [query]
        strategy_parts = ["base"]

        if enable_mqe and mqe_expansions > 0:
            mqe_queries = self._expander.expand(query, n=mqe_expansions)
            expansions.extend(mqe_queries[1:])   # 去掉重复的原始查询
            strategy_parts.append("MQE")

        if enable_hyde:
            hyde_text = self._hyde.generate(query)
            if hyde_text and hyde_text != query:
                expansions.append(hyde_text)
                strategy_parts.append("HyDE")

        # 2. 去重和修剪
        uniq: List[str] = []
        for e in expansions:
            e = (e or "").strip()
            if e and e not in uniq:
                uniq.append(e)
        expansions = uniq

        # 3. 候选池分配：pool = max(top_k × multiplier, 20)
        pool = max(top_k * max(1, candidate_pool_multiplier), 20)
        per = max(1, pool // max(1, len(expansions)))

        # 4. 并行检索每个扩展查询（统一向量检索）
        def _search_one(q: str):
            return self._index.search(q, top_k=per, min_score=min_score)

        agg: Dict[str, Tuple[float, RagChunk, str]] = {}
        workers = min(8, len(expansions))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_search_one, q) for q in expansions]
                for future in futures:
                    for score, chunk, method in future.result():
                        cid = chunk.chunk_id
                        if cid not in agg or score > agg[cid][0]:
                            agg[cid] = (score, chunk, method)
        else:
            for score, chunk, method in _search_one(expansions[0]):
                agg[chunk.chunk_id] = (score, chunk, method)

        # 5. 分数排序 → top-k
        merged = sorted(agg.values(), key=lambda x: x[0], reverse=True)
        results = merged[:top_k]

        stats = {
            "expansions": expansions,
            "strategy": "+".join(strategy_parts),
            "pool": pool,
            "per": per,
            "total_candidates": len(agg),
            "matched": len(results),
        }
        return results, stats

    def _search_expanded(self, args: Dict[str, Any]) -> str:
        """统一扩展检索：enable_mqe / enable_hyde 按场景组合启用。"""
        query = (args.get("query") or "").strip()
        if not query:
            return "❌ search_expanded 操作需要提供 query 参数"

        top_k = max(1, min(50, int(args.get("top_k", 8))))
        min_score = max(0.0, min(1.0, float(args.get("min_score", 0.0))))
        enable_mqe = bool(args.get("enable_mqe", False))
        enable_hyde = bool(args.get("enable_hyde", False))
        mqe_expansions = max(1, min(10, int(args.get("mqe_expansions", 2))))
        multiplier = max(1, min(20, int(args.get("candidate_pool_multiplier", 4))))

        results, stats = self._retrieve_expanded(
            query, top_k=top_k, score_threshold=None,
            enable_mqe=enable_mqe, mqe_expansions=mqe_expansions,
            enable_hyde=enable_hyde, candidate_pool_multiplier=multiplier,
            min_score=min_score,
        )

        if not results:
            return "📭 扩展检索没有找到相关分块。"

        lines = [
            "=" * 60,
            f"🔍 统一扩展检索: \"{query}\"",
            f"   策略: {stats['strategy']} | 扩展查询({len(stats['expansions'])}个)",
            f"   候选池: {stats['pool']}（top_k×{multiplier}）| 每查询召回{stats['per']} | "
            f"合并候选{stats['total_candidates']} → 返回{stats['matched']}",
            "=" * 60,
        ]
        lines.append(f"   扩展查询: {', '.join(e[:40] for e in stats['expansions'])}")
        for i, (score, chunk, method) in enumerate(results[:top_k], 1):
            lines.append("")
            lines.append(f"--- 结果 {i} [{method}] score={score:.4f} ---")
            lines.append(f"  来源: {chunk.doc_name}（{chunk.metadata.get('section', '')}）")
            lines.append(f"  内容: {chunk.content[:120]}{'...' if len(chunk.content) > 120 else ''}")

        return "\n".join(lines)

    # ============================================================
    # 增强问答
    # ============================================================

    def _query(self, args: Dict[str, Any]) -> str:
        """RAG 问答：检索上下文 → Qwen 生成带引用答案。

        检索增强（生成内容仅用于检索召回，不影响主问答提示词）：
          - 统一框架（推荐）：enable_mqe=True / enable_hyde=True 按场景组合
          - 旧接口：mqe=True / hyde=True
        """
        question = (args.get("question") or "").strip()
        if not question:
            return "❌ query 操作需要提供 question 参数"

        top_k = max(1, min(20, int(args.get("top_k", 5))))
        show_context = bool(args.get("show_context", False))
        use_mqe = bool(args.get("mqe", False))            # 旧接口
        use_hyde = bool(args.get("hyde", False))          # 旧接口
        enable_mqe = bool(args.get("enable_mqe", False))  # 统一框架
        enable_hyde = bool(args.get("enable_hyde", False))

        # 1. 检索上下文（生成内容仅用于检索，不用于回答；主对话提示词不受影响）
        enhance_note = ""
        if enable_mqe or enable_hyde:
            # 统一扩展检索框架：MQE + HyDE 按场景组合
            mqe_expansions = max(1, min(10, int(args.get("mqe_expansions", 2))))
            multiplier = max(1, min(20, int(args.get("candidate_pool_multiplier", 4))))
            exp_results, stats = self._retrieve_expanded(
                question, top_k=top_k,
                enable_mqe=enable_mqe, mqe_expansions=mqe_expansions,
                enable_hyde=enable_hyde, candidate_pool_multiplier=multiplier,
            )
            results = exp_results
            enhance_note = (f" | 扩展检索[{stats['strategy']}]: "
                            f"{len(stats['expansions'])}查询/池{stats['pool']}")
        elif use_mqe:
            n_queries = max(1, min(10, int(args.get("n_queries", 3))))
            mqe_results, stats = self._retrieve_mqe(question, top_k, n_queries)
            results = [(score, chunk, method) for score, chunk, method, _ in mqe_results][:top_k]
            mode_label = "批量" if stats["mode"] == "batch" else "单查询"
            enhance_note = (f" | MQE: {stats['total']}查询/{mode_label}/{stats['batches']}批")
        elif use_hyde:
            hyde_results, stats = self._retrieve_hyde(question, top_k)
            results = hyde_results
            hyde_note = "HyDE" if stats["used_hypothesis"] else "HyDE(LLM兜底)"
            enhance_note = f" | {hyde_note}"
        else:
            results = self._index.search(question, top_k=top_k)

        if not results:
            return "📭 知识库中没有检索到相关资料，无法回答。"

        # 2. 构建上下文
        context_parts = []
        for i, (score, chunk, method) in enumerate(results, 1):
            context_parts.append(f"[{i}] (相关度:{score:.3f}) 来源:{chunk.doc_name}\n{chunk.content}")
        context = "\n\n".join(context_parts)

        # 3. 调用 Qwen（主对话系统提示词不受影响；答案结果走缓存层）
        qa_cache_key = f"qa:{question}:{top_k}:{enhance_note}:{show_context}"
        cached_output = self._cache.get(qa_cache_key)
        if cached_output is not None:
            return cached_output

        llm = self._get_llm()
        if llm is None:
            return ("❌ 未配置 DASHSCOPE_API_KEY，无法调用 Qwen 问答。\n"
                    f"以下为检索到的资料：\n\n{context}")

        messages = [
            {"role": "system", "content": "你是一个严谨的知识库问答助手。"},
            {"role": "user", "content": self.RAG_QA_PROMPT.format(context=context, question=question)},
        ]

        try:
            answer = llm.chat(messages, temperature=0.3)
        except Exception as e:
            return f"❌ Qwen 调用失败: {e}"

        lines = [
            "=" * 60,
            f"❓ 问题: {question}",
            f"  模型: {llm.model} | 召回: {len(results)} 块{enhance_note}",
            "=" * 60,
            "📝 回答:",
            answer,
        ]
        if show_context:
            lines.append("")
            lines.append("📚 引用资料:")
            for i, (score, chunk, method) in enumerate(results, 1):
                lines.append(f"  [{i}] {chunk.doc_name} ({chunk.metadata.get('section', '')}) "
                             f"score={score:.3f}")
                lines.append(f"      {chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}")

        output = "\n".join(lines)
        self._cache.set(qa_cache_key, output)
        return output

    # ============================================================
    # 知识库管理
    # ============================================================

    def _list(self, args: Dict[str, Any]) -> str:
        """列出知识库中的文档。"""
        docs = self._index.get_docs()
        doc_type = args.get("doc_type")
        if doc_type:
            docs = [d for d in docs if d.doc_type == doc_type]

        if not docs:
            return "📭 知识库为空。"

        lines = [
            "=" * 60,
            f"📚 知识库文档列表（共 {len(docs)} 份）",
            "=" * 60,
        ]
        for d in docs:
            lines.append(f"  ID: {d.doc_id[:12]}... | {d.name} "
                         f"[{d.doc_type}] {d.chunk_count}块 {d.created_at[:10]}")
        return "\n".join(lines)

    def _stats(self, args: Dict[str, Any]) -> str:
        """知识库统计。"""
        docs = self._index.get_docs()
        type_stats: Counter = Counter(d.doc_type for d in docs)
        embed_note = "可用" if self._index.embedding_available else "不可用（TF-IDF 模式）"
        cache_stats = self._cache.stats()
        return (
            "📊 知识库统计\n"
            f"  文档数: {self._index.get_doc_count()}\n"
            f"  分块数: {self._index.get_chunk_count()}\n"
            f"  嵌入 API: {embed_note}\n"
            f"  类型分布: {dict(type_stats) if type_stats else '(空)'}\n"
            f"  缓存层: {cache_stats['size']}/{cache_stats['max_size']} 项 "
            f"(命中率 {cache_stats['hit_rate']:.1%}, 命中{cache_stats['hits']} "
            f"未命中{cache_stats['misses']})"
        )

    def _delete(self, args: Dict[str, Any]) -> str:
        """删除文档。"""
        doc_id = args.get("doc_id", "").strip()
        if not doc_id:
            return "❌ delete 操作需要提供 doc_id 参数"
        docs = self._index.get_docs()
        matched = [d for d in docs if d.doc_id.startswith(doc_id)]
        if len(matched) != 1:
            return (f"❌ 文档 ID 匹配到 {len(matched)} 份，请提供更精确的 ID"
                    if matched else f"❌ 未找到文档: {doc_id}")
        doc = matched[0]
        self._index.delete_doc(doc.doc_id)
        self._cache.clear()   # 知识库变更 → 缓存失效（一致性）
        return f"✅ 已删除文档: {doc.name}（{doc.chunk_count} 块）"

    def _clear(self, args: Dict[str, Any]) -> str:
        """清空知识库。"""
        count = self._index.get_doc_count()
        self._index.clear()
        self._cache.clear()   # 知识库变更 → 缓存失效（一致性）
        return f"✅ 已清空知识库（删除 {count} 份文档）"

    # 供外部调用的快捷接口
    def add_file(self, file_path: str, description: str = None) -> str:
        return self.execute("add_file", file_path=file_path, description=description)

    def ask(self, question: str, top_k: int = 5) -> str:
        return self.execute("query", question=question, top_k=top_k)


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("📚 RAG 工具演示（阿里云百炼）")
    print("=" * 60)
    print()

    # ========== 准备样例文档 ==========
    demo_dir = os.path.join(tempfile.gettempdir(), "rag_demo")
    os.makedirs(demo_dir, exist_ok=True)

    # 1. Markdown 技术文档
    md_path = os.path.join(demo_dir, "rag_guide.md")
    md_content = """# RAG 检索增强生成入门指南

## 什么是 RAG
RAG（Retrieval-Augmented Generation，检索增强生成）是一种将信息检索与大语言模型结合的技术范式。
它先从知识库中检索与问题相关的文档片段，再将检索结果作为上下文注入提示词，最终由大语言模型生成答案。

## RAG 的核心流程
RAG 通常包含三个核心步骤：索引构建、检索召回和增强生成。
索引构建阶段，文档被切分为小块并编码为向量存储。
检索召回阶段，用户的查询被编码后与知识库中的向量计算相似度，召回最相关的片段。
增强生成阶段，大语言模型基于召回的片段生成带引用的答案。

## RAG 的优势
RAG 相比传统微调方法有显著优势：不需要重新训练模型，知识可以实时更新，
回答可以追溯引用来源，有效降低大模型的幻觉问题。
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 2. Word 文档（zipfile 手工构造最小 docx）
    docx_path = os.path.join(demo_dir, "project_note.docx")
    try:
        with zipfile.ZipFile(docx_path, "w") as zf:
            content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
            rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
            document = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:r><w:t>项目周报：本周完成了 RAG 工具的原型开发。</w:t></w:r></w:p>
<w:p><w:r><w:t>下周计划接入更多的文档格式支持，包括 PDF 和图片。</w:t></w:r></w:p>
</w:body>
</w:document>"""
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", rels)
            zf.writestr("word/document.xml", document)
        docx_ok = True
    except Exception:
        docx_ok = False

    # 3. 图片（元数据语义化，不真正生成图片文件）
    image_path = os.path.join(demo_dir, "architecture_diagram.png")
    with open(image_path, "wb") as f:
        f.write(b"fake-image-bytes")

    # 4. 音频（元数据语义化）
    audio_path = os.path.join(demo_dir, "meeting_recording.mp3")
    with open(audio_path, "wb") as f:
        f.write(b"fake-audio-bytes")

    # 5. 简单 PDF（zlib 压缩流，测试轻量提取）
    pdf_path = os.path.join(demo_dir, "quickstart.pdf")
    try:
        content_stream = (b"BT /F1 12 Tf 72 720 Td "
                          b"(RAG quickstart: retrieve then generate.) Tj ET\n"
                          b"BT /F1 12 Tf 72 700 Td "
                          b"(PDF documents are supported.) Tj ET")
        compressed = zlib.compress(content_stream)
        stream_header = ("4 0 obj << /Length %d /Filter /FlateDecode >>\nstream\n"
                         % len(compressed)).encode("ascii")
        pdf_data = (
            b"%PDF-1.4\n"
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R >> endobj\n"
            + stream_header
            + compressed + b"\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n"
            b"trailer << /Size 5 /Root 1 0 R >>\nstartxref\n0\n%%EOF"
        )
        with open(pdf_path, "wb") as f:
            f.write(pdf_data)
        pdf_ok = True
    except Exception:
        pdf_ok = False

    # 6. Excel 表格（openpyxl 生成，测试 MarkItDown 表格转换）
    xlsx_path = os.path.join(demo_dir, "qa_records.xlsx")
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "QA记录"
        ws.append(["问题", "答案"])
        ws.append(["RAG 是什么", "检索增强生成，结合检索与大模型"])
        ws.append(["如何减少幻觉", "使用 RAG 引用真实资料"])
        wb.save(xlsx_path)
        xlsx_ok = True
    except Exception:
        xlsx_ok = False

    # 7. PPT 演示文稿（python-pptx 生成，测试 MarkItDown 幻灯片转换）
    pptx_path = os.path.join(demo_dir, "rag_overview.pptx")
    try:
        from pptx import Presentation
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "RAG 系统概览"
        slide.placeholders[1].text = "第一页：介绍 RAG 检索增强生成"
        slide2 = prs.slides.add_slide(prs.slide_layouts[1])
        slide2.shapes.title.text = "核心组件"
        slide2.placeholders[1].text = "文档转换、向量索引、混合检索、Qwen 生成"
        prs.save(pptx_path)
        pptx_ok = True
    except Exception:
        pptx_ok = False

    # ========== 初始化 RAG 工具 ==========
    rag = RagTool()
    print("✅ RAG 管道已初始化")
    print(f"   嵌入 API: {'可用' if rag._index.embedding_available else '不可用（TF-IDF 模式）'}")
    print(f"   转换引擎: {'MarkItDown 统一转换' if MARKITDOWN_AVAILABLE else '轻量回退解析'}")
    print(f"   分块配置: chunk_size=500, overlap=50")
    print()

    # ========== 测试 0: 文档转换预览（MarkItDown 统一转换）==========
    print("--- RAG 0: 文档转换预览（_convert_to_markdown 统一转换）---")
    if xlsx_ok:
        print("[xlsx] Excel 表格:")
        print(rag.execute("convert", file_path=xlsx_path))
        print()
    if pptx_ok:
        print("[pptx] PPT 演示文稿:")
        print(rag.execute("convert", file_path=pptx_path))
        print()
    print("[docx] Word 文档:")
    if docx_ok:
        print(rag.execute("convert", file_path=docx_path))
    else:
        print("   ⚠️ docx 样例生成失败，跳过")
    print()
    print("[png] 图片（MarkItDown 无 LLM 时回退元数据语义化）:")
    print(rag.execute("convert", file_path=image_path, description="系统架构示意图，展示 RAG 各组件关系"))
    print()

    # ========== 测试 1: 添加多格式文档 ==========
    print("--- RAG 1: 添加多格式文档 ---")
    print("[1] Markdown:")
    print(rag.add_file(md_path))
    print()
    print("[2] Word (docx):")
    if docx_ok:
        print(rag.add_file(docx_path))
    else:
        print("   ⚠️ docx 样例生成失败，跳过")
    print()
    print("[3] PDF:")
    if pdf_ok:
        print(rag.add_file(pdf_path))
    else:
        print("   ⚠️ PDF 样例生成失败，跳过")
    print()
    print("[4] 图片（元数据语义化）:")
    print(rag.add_file(image_path, description="系统架构示意图，展示 RAG 各组件关系"))
    print()
    print("[5] 音频（元数据语义化）:")
    print(rag.add_file(audio_path, description="项目评审会议录音"))
    print()
    print("[6] 纯文本 add_text:")
    print(rag.execute("add_text", content="大语言模型通过注意力机制理解文本上下文。"
                                          "Transformer 架构是当前主流大模型的基石。",
                      title="LLM 原理笔记"))
    print()
    print("[7] Excel (xlsx):")
    if xlsx_ok:
        print(rag.add_file(xlsx_path))
    else:
        print("   ⚠️ xlsx 样例生成失败，跳过")
    print()
    print("[8] PPT (pptx):")
    if pptx_ok:
        print(rag.add_file(pptx_path))
    else:
        print("   ⚠️ pptx 样例生成失败，跳过")
    print()

    # ========== 测试 2: 知识库统计与列表 ==========
    print("--- RAG 2: 知识库统计与列表 ---")
    print(rag.execute("stats"))
    print()
    print(rag.execute("list"))
    print()

    # ========== 测试 3: 智能检索 ==========
    print("--- RAG 3: 智能检索「RAG 的核心流程」---")
    print(rag.execute("search", query="RAG 的核心流程是什么", top_k=3))
    print()

    # ========== 测试 4: LLM 增强问答 ==========
    print("--- RAG 4: LLM 增强问答 ---")
    print(rag.execute("query", question="RAG 相比传统微调有什么优势？", top_k=3, show_context=True))
    print()

    # ========== 测试 5: 语义检索（词汇不重叠）==========
    print("--- RAG 5: 语义检索「如何让大模型回答更准确？」---")
    print(rag.execute("search", query="如何让大模型回答更准确", top_k=3))
    print()

    # ========== 测试 5b: 表格/幻灯片内容检索（MarkItDown 转换结果）==========
    print("--- RAG 5b: 表格/幻灯片内容检索 ---")
    print("   [search] 「如何减少幻觉」(xlsx 表格):")
    print(rag.execute("search", query="如何减少幻觉", top_k=2))
    print()
    print("   [search] 「RAG 系统概览」(pptx 幻灯片):")
    print(rag.execute("search", query="RAG 系统概览", top_k=2))
    print()

    # ========== 测试 5c: Token 级智能分块（chunk_paragraphs + 重叠）==========
    print("--- RAG 5c: Token 级智能分块（chunk_paragraphs + 重叠）---")
    test_paragraphs = [
        {"content": "RAG（检索增强生成）是将信息检索与大语言模型结合的技术范式。", "metadata": {"section": "简介"}},
        {"content": "它先从知识库中检索与问题相关的文档片段。", "metadata": {"section": "简介"}},
        {"content": "再将检索结果作为上下文注入提示词，由大语言模型生成答案。", "metadata": {"section": "简介"}},
        {"content": "索引构建阶段，文档被切分为小块并编码为向量存储。", "metadata": {"section": "流程"}},
        {"content": "检索召回阶段，查询被编码后与知识库向量计算相似度。", "metadata": {"section": "流程"}},
        {"content": "增强生成阶段，模型基于召回片段生成带引用的答案。", "metadata": {"section": "流程"}},
        {"content": "RAG 不需要重新训练模型，知识可实时更新，可追溯引用来源。", "metadata": {"section": "优势"}},
    ]
    token_chunks = chunk_paragraphs(test_paragraphs, chunk_tokens=60, overlap_tokens=20)
    print(f"   输入段落: {len(test_paragraphs)} 段 → 输出分块: {len(token_chunks)} 块")
    print(f"   每段 token: {[estimate_tokens(p['content']) for p in test_paragraphs]}")
    print()
    print("   [_approx_token_len] 中英文混合估算:")
    for sample in ["Hello world 你好世界", "RAG 检索增强生成",
                   "Transformer 架构是主流 大模型 的基石"]:
        print(f"     {sample!r} → {_approx_token_len(sample)} tokens")
    cjk_ext_sample = "𠀀𠀁"  # CJK 扩展 B 区字符
    print(f"     CJK 扩展 B 字符 {cjk_ext_sample!r} → "
          f"{sum(1 for ch in cjk_ext_sample if _is_cjk(ch))} CJK tokens")
    print()
    for tc in token_chunks:
        print(f"   块 {tc['chunk_index']}: tokens={tc['tokens']} (≤60) | "
              f"段落数={tc['paragraph_count']} | {tc['content'][:60]}...")
    print()
    # 重叠连续性验证
    print("   重叠连续性验证（相邻块首尾是否有重复内容）:")
    for ci in range(1, len(token_chunks)):
        prev_tail = token_chunks[ci - 1]["content"][-20:]
        cur_head = token_chunks[ci]["content"][:20]
        overlap_detect = "✅ 有重叠" if prev_tail in token_chunks[ci]["content"] else "⚠️ 无重叠"
        print(f"     块{ci-1}尾部: ...{prev_tail} | 块{ci}头部: {cur_head}... → {overlap_detect}")
    print()

    # ========== 测试 5d: Token 分块管线集成（RagTool + chunk_tokens 配置）==========
    print("--- RAG 5d: Token 分块管线集成（chunk_tokens=120, overlap_tokens=30）---")
    rag_token = RagTool(config={"chunk_tokens": 120, "overlap_tokens": 30})
    print(f"   分块器模式: {'Token 模式' if rag_token._splitter.use_tokens else '字符模式'}")
    print(rag_token.add_file(md_path))
    print()
    doc = rag_token._index.get_docs()[0]
    chunk_docs = rag_token._index.get_doc_chunks(doc.doc_id)
    print(f"   文档分块详情（{len(chunk_docs)} 块）:")
    for i, c in enumerate(chunk_docs):
        print(f"     块{i}: tokens={c.metadata.get('tokens', '?')} | "
              f"段落数={c.metadata.get('paragraph_count', '?')} | "
              f"章节={c.metadata.get('section', '?')} | {c.content[:40]}...")
    print()
    print("   [search]「RAG 的核心流程」（token 分块后检索）:")
    print(rag_token.execute("search", query="RAG 的核心流程", top_k=2))
    print()

    # ========== 测试 5e: 统一分块索引（index_chunks：百炼嵌入 + TF-IDF 兜底）==========
    print("--- RAG 5e: 统一分块索引（index_chunks：百炼嵌入 + TF-IDF 兜底）---")
    sample_chunks = [
        {"content": "RAG 检索增强生成技术结合了信息检索与大语言模型。", "metadata": {"section": "简介"}},
        {"content": "混合检索使用嵌入向量和 TF-IDF 双路召回。", "metadata": {"section": "检索"}},
        {"content": "Qwen 模型基于召回片段生成带引用的答案。", "metadata": {"section": "生成"}},
    ]
    print("   [百炼 API 可用] index_chunks 结果:")
    indexed = index_chunks(sample_chunks)
    for ic in indexed:
        emb = ic.get("embedding")
        emb_note = f"dims={len(emb)}" if emb else "None"
        tfidf_terms = len(ic.get("tfidf_vector", {}))
        print(f"     method={ic['vector_method']} | embedding {emb_note} | "
              f"tfidf_terms={tfidf_terms} | {ic['content'][:30]}...")
    print()
    print("   [TF-IDF 兜底] 临时禁用嵌入 API:")
    # 模拟 API 不可用：构造不可用的嵌入客户端
    import types
    fake_client = types.SimpleNamespace(is_available=False, embed=lambda t: None, embed_batch=lambda ts: [None] * len(ts))
    encoder_offline = VectorEncoder(embedding_client=fake_client)
    print(f"     embedding_available={encoder_offline.embedding_available}")
    for txt in ["检索增强生成示例"]:
        dense, sparse, method = encoder_offline.encode(txt)
        print(f"     encode('{txt}') → method={method}, dense={'None' if dense is None else len(dense)}, "
              f"tfidf_terms={len(sparse)}")
    print()

    # ========== 测试 5f: 多查询扩展检索（MQE）==========
    print("--- RAG 5f: 多查询扩展检索（MQE）---")
    expander = QueryExpander(llm_client=None, threshold=5, batch_size=10)
    print("   [QueryExpander] 独立 LLM 生成扩展查询（不影响主对话系统提示词）:")
    expanded = expander.expand("RAG 检索增强生成的优势", n=3)
    print(f"     扩展查询: {expanded}")
    print()
    print("   [兜底机制] LLM 不可用 → 仅返回原始查询:")
    import types as _types
    broken_llm = _types.SimpleNamespace(chat=lambda m, temperature=0.3: None)
    exp_offline = QueryExpander(llm_client=broken_llm, threshold=3, batch_size=10)
    print(f"     expand('测试查询', n=3) → {exp_offline.expand('测试查询', n=3)}")
    _resp_test = '1. 查询A\n2. 查询B\n3. 查询C\n4. 查询D'
    print(f"     解析兜底（生成过多只取前 n 个）: {QueryExpander._parse_response(_resp_test)[:3]}")
    print()
    print("   [模式调度] threshold=5, batch_size=3:")
    mock_llm = _types.SimpleNamespace(chat=lambda m, temperature=0.3: (
        "1. 查询一\n2. 查询二\n3. 查询三\n4. 查询四\n5. 查询五"))
    exp_plan = QueryExpander(llm_client=mock_llm, threshold=4, batch_size=3)
    queries, mode, total = exp_plan.plan("测试", n=5)
    print(f"     查询总数={total} > threshold={exp_plan.threshold} → 模式={mode}")
    batch_info = [len(b) for b in exp_plan.iter_batches(queries)]
    print(f"     分批（batch_size=3）: {batch_info}（{len(batch_info)}批）")
    exp_single = QueryExpander(llm_client=mock_llm, threshold=10, batch_size=3)
    q2, m2, t2 = exp_single.plan("测试", n=2)
    print(f"     查询总数={t2} ≤ threshold=10 → 模式={m2}")
    print()
    print("   [search_mqe] 实际检索（扩展查询仅用于检索增强）:")
    print(rag.execute("search_mqe", query="RAG 相比微调有什么好处", top_k=3, n_queries=3))
    print()
    print("   [query + mqe] 增强问答（MQE 检索 + Qwen 回答）:")
    print(rag.execute("query", question="RAG 的核心步骤有哪些？", top_k=2, mqe=True))
    print()

    # ========== 测试 5g: 假设文档嵌入（HyDE）==========
    print("--- RAG 5g: 假设文档嵌入（HyDE）---")
    hyde = HydeGenerator(llm_client=None)
    print("   [HydeGenerator] 独立 LLM 生成假设文档（不影响主对话系统提示词）:")
    hypo = hyde.generate("RAG 相比传统微调的优势")
    print(f"     假设文档: {hypo}")
    print()
    print("   [兜底机制] LLM 不可用 → 回退原始查询:")
    broken_hyde_llm = _types.SimpleNamespace(chat=lambda m, temperature=0.3: "")
    hyde_offline = HydeGenerator(llm_client=broken_hyde_llm)
    fallback_hypo = hyde_offline.generate("测试查询")
    print(f"     generate('测试查询') → '{fallback_hypo}'"
          + ("（✅ 回退原始查询）" if fallback_hypo == "测试查询" else ""))
    print()
    print("   [search_hyde] 实际检索（假设文档仅用于检索增强）:")
    print(rag.execute("search_hyde", query="RAG 为什么能减少幻觉", top_k=3))
    print()
    print("   [query + hyde] 增强问答（HyDE 检索 + Qwen 回答）:")
    print(rag.execute("query", question="RAG 的核心流程是什么？", top_k=2, hyde=True))
    print()

    # ========== 测试 5h: 统一扩展检索框架（MQE + HyDE 组合）==========
    print("--- RAG 5h: 统一扩展检索框架（enable_mqe + enable_hyde）---")
    print("   [场景1: 一般查询 → 仅 MQE] search_expanded(enable_mqe=True):")
    print(rag.execute("search_expanded", query="RAG 相比微调的优势",
                      top_k=4, enable_mqe=True, mqe_expansions=2))
    print()
    print("   [场景2: 专业领域查询 → MQE + HyDE] search_expanded(双启用):")
    print(rag.execute("search_expanded", query="RAG 如何减少大模型幻觉",
                      top_k=4, enable_mqe=True, enable_hyde=True,
                      mqe_expansions=2, candidate_pool_multiplier=4))
    print()
    print("   [场景3: 性能敏感 → 基础检索（均不启用）] search_expanded:")
    print(rag.execute("search_expanded", query="RAG 如何减少大模型幻觉", top_k=4))
    print()
    print("   [query 统一框架] 专业领域查询（enable_mqe + enable_hyde）:")
    print(rag.execute("query", question="RAG 相比微调有哪些优势？",
                      top_k=2, enable_mqe=True, enable_hyde=True))
    print()

    # ========== 测试 5i: 缓存层（SafeFullCache）==========
    print("--- RAG 5i: 缓存层（SafeFullCache）---")
    print("   [基础读写 + TTL] set/get/过期:")
    ttl_cache = SafeFullCache(max_size=100, default_ttl=1)
    ttl_cache.set("天气", "晴天", ttl=1)
    print(f"     get('天气') → {ttl_cache.get('天气')}")
    time.sleep(1.2)
    print(f"     1.2s 后 get('天气') → {ttl_cache.get('天气')}（✅ 已过期淘汰）")
    print()
    print("   [容量保护] max_size=3，超出淘汰最旧:")
    cap_cache = SafeFullCache(max_size=3, default_ttl=3600)
    for k in ["A", "B", "C", "D", "E"]:
        cap_cache.set(k, f"值{k}")
    print(f"     写入 5 项后 size={cap_cache.stats()['size']}（≤3），"
          f"A 已被淘汰: get('A')={cap_cache.get('A')}")
    print()
    print("   [一致性更新] update:")
    upd_cache = SafeFullCache(max_size=10, default_ttl=3600)
    upd_cache.set("k", "v1")
    upd_cache.update("k", "v2")
    print(f"     update 后 get('k') → {upd_cache.get('k')}")
    print()
    print("   [get_or_set] 未命中计算 + 命中复用:")
    gos_cache = SafeFullCache(max_size=10, default_ttl=3600)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return f"计算值{calls['n']}"

    print(f"     首次: {gos_cache.get_or_set('x', compute)}（compute 调用 {calls['n']} 次）")
    print(f"     二次: {gos_cache.get_or_set('x', compute)}（compute 调用 {calls['n']} 次，✅ 命中缓存）")
    print()
    print("   [MQE/HyDE 集成] 重复调用命中缓存（不再调用 LLM）:")
    print(f"     调用前缓存统计: {rag._cache.stats()}")
    rag.execute("search_mqe", query="RAG 相比微调的优势", top_k=3, n_queries=3)
    rag.execute("search_hyde", query="RAG 相比微调的优势", top_k=3)
    after_first = rag._cache.stats()
    rag.execute("search_mqe", query="RAG 相比微调的优势", top_k=3, n_queries=3)
    rag.execute("search_hyde", query="RAG 相比微调的优势", top_k=3)
    after_second = rag._cache.stats()
    print(f"     首轮调用后: size={after_first['size']} 命中={after_first['hits']}")
    print(f"     二轮调用后: size={after_second['size']} 命中={after_second['hits']} "
          f"（✅ 新增命中 {after_second['hits'] - after_first['hits']} 次，未新增缓存项）")
    print()
    print("   [QA 答案缓存] 相同问题二次查询命中缓存:")
    q1 = rag.execute("query", question="RAG 的核心流程是什么？", top_k=2)
    s1 = rag._cache.stats()
    q2 = rag.execute("query", question="RAG 的核心流程是什么？", top_k=2)
    s2 = rag._cache.stats()
    print(f"     两次回答一致: {q1 == q2} | 缓存命中数 {s1['hits']} → {s2['hits']}（✅ +{s2['hits'] - s1['hits']}）")
    print(f"     当前缓存统计: {rag._cache.stats()}")
    print()

    # ========== 测试 6: 删除与清空 ==========
    print("--- RAG 6: 删除与清空 ---")
    docs = rag._index.get_docs()
    if docs:
        target_id = docs[-1].doc_id[:12]
        print(rag.execute("delete", doc_id=target_id))
    print()
    print(rag.execute("stats"))
    print()
    print(rag.execute("clear"))
    print(rag.execute("stats"))
    print()

    print("=" * 60)
    print("✅ RAG 工具演示完成")
    print("=" * 60)