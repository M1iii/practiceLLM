"""Token 估算与智能分块。"""

import re
import math
from typing import Dict, List, Tuple, Any


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or
        0x3400 <= code <= 0x4DBF or
        0x20000 <= code <= 0x2A6DF or
        0x2A700 <= code <= 0x2B73F or
        0x2B740 <= code <= 0x2B81F or
        0x2B820 <= code <= 0x2CEAF or
        0xF900 <= code <= 0xFAFF
    )


def _approx_token_len(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk_tokens = len([t for t in text.split() if t])
    return cjk + non_cjk_tokens


def estimate_tokens(text: str) -> int:
    return _approx_token_len(text)


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


def chunk_paragraphs(paragraphs: List[Dict], chunk_tokens: int, overlap_tokens: int) -> List[Dict]:
    """基于 Token 数量的智能分块，同时构建重叠部分，保持信息连续性。"""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens 必须为正数")
    overlap_tokens = max(0, overlap_tokens)

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

    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0
    n = len(pieces)

    while i < n:
        para = pieces[i]
        tokens = para["tokens"]
        if cur_tokens + tokens <= chunk_tokens:
            cur.append(para)
            cur_tokens += tokens
            i += 1
            continue
        if cur:
            chunks.append(_finalize_chunk(cur, len(chunks)))
            cur, cur_tokens = _build_overlap(cur, overlap_tokens)
            while cur and cur_tokens + tokens > chunk_tokens:
                dropped = cur.pop(0)
                cur_tokens -= dropped["tokens"]
        else:
            chunks.append(_finalize_chunk([para], len(chunks)))
            i += 1

    if cur:
        chunks.append(_finalize_chunk(cur, len(chunks)))
    return chunks


def _finalize_chunk(items: List[Dict], index: int) -> Dict:
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
    overlap: List[Dict] = []
    tok = 0
    for p in reversed(items):
        if overlap and tok + p["tokens"] > overlap_tokens:
            break
        overlap.insert(0, p)
        tok += p["tokens"]
    return overlap, tok


def _split_oversized_paragraph(content: str, chunk_tokens: int) -> List[str]:
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
    lo, hi = 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


class TextSplitter:
    """智能文本分块器：支持字符模式（chunk_size）与 Token 模式（chunk_tokens）。"""

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
        if self.use_tokens:
            return chunk_paragraphs(paragraphs, self.chunk_tokens, self.overlap_tokens or 0)
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
        text = text.strip()
        if not text:
            return []
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n', text) if p.strip()]
        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            para_parts = self._split_long_paragraph(para)
            for part in para_parts:
                if not part:
                    continue
                if len(current) + len(part) + 1 <= self.chunk_size:
                    current = f"{current}\n{part}".strip() if current else part
                else:
                    if current:
                        chunks.append(current)
                    overlap_text = self._get_overlap(current)
                    current = f"{overlap_text}\n{part}".strip() if overlap_text else part
                    while len(current) > self.chunk_size:
                        chunks.append(current[:self.chunk_size])
                        current = current[self.chunk_size - self.overlap:] if self.overlap else ""
        if current.strip():
            chunks.append(current)
        return [c for c in chunks if c.strip()]

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        if len(paragraph) <= self.chunk_size:
            return [paragraph]
        sentences = re.split(r'(?<=[。！？；.!?;])\s*', paragraph)
        sentences = [s for s in sentences if s.strip()]
        if len(sentences) <= 1:
            return [paragraph]
        parts: List[str] = []
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 <= self.chunk_size:
                current = f"{current}{s}" if current else s
            else:
                if current:
                    parts.append(current)
                current = s
                while len(current) > self.chunk_size:
                    parts.append(current[:self.chunk_size])
                    current = current[self.chunk_size:]
        if current:
            parts.append(current)
        return parts

    @staticmethod
    def _get_overlap(text: str) -> str:
        if not text:
            return ""
        return text[-min(50, len(text)):]