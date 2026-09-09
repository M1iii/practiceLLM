"""多格式文档加载器：将不同格式的文件解析为文本块列表。"""

import os
import re
import zlib
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from src.tools.rag.rag_tool.models import EXT_DOC_TYPE, TEXT_EXTENSIONS, METADATA_EXTENSIONS


try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MarkItDown = None
    MARKITDOWN_AVAILABLE = False


class DocumentLoader:
    """多格式文档加载器：将不同格式的文件解析为文本块列表。"""

    def load(self, file_path: str, description: str = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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

        markdown = self._convert_to_markdown(file_path)
        if markdown:
            doc_meta["converter"] = "markitdown"
            blocks = self._split_markdown_blocks(markdown)
            for block in blocks:
                block["metadata"]["converter"] = "markitdown"
        else:
            doc_meta["converter"] = "fallback"
            blocks = self._load_fallback(file_path, ext, doc_type, description)
            for block in blocks:
                block["metadata"]["converter"] = "fallback"

        blocks = [b for b in blocks if b["content"] and b["content"].strip()]
        return doc_meta, blocks

    def _convert_to_markdown(self, path: str) -> str:
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
        if ext in TEXT_EXTENSIONS:
            content = self._read_text(file_path)
            return [{"content": content, "metadata": {"section": "全文"}}]
        if doc_type == "word":
            return self._parse_docx(file_path)
        if doc_type == "excel":
            return self._parse_xlsx(file_path)
        if doc_type == "ppt":
            return self._parse_pptx(file_path)
        if doc_type == "pdf":
            return self._parse_pdf(file_path)
        return self._parse_media(file_path, doc_type, description)

    @staticmethod
    def _split_markdown_blocks(markdown: str) -> List[Dict[str, Any]]:
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

    @staticmethod
    def _read_text(file_path: str) -> str:
        for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()

    @staticmethod
    def _parse_docx(file_path: str) -> List[Dict[str, Any]]:
        blocks = []
        try:
            with zipfile.ZipFile(file_path) as zf:
                xml_bytes = zf.read("word/document.xml")
            root = ET.fromstring(xml_bytes)
            for para in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
                texts = [t.text or "" for t in para.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")]
                text = "".join(texts).strip()
                if text:
                    blocks.append({"content": text, "metadata": {"section": "段落"}})
        except Exception:
            blocks = []
        return blocks

    @staticmethod
    def _parse_xlsx(file_path: str) -> List[Dict[str, Any]]:
        blocks = []
        try:
            with zipfile.ZipFile(file_path) as zf:
                shared = []
                if "xl/sharedStrings.xml" in zf.namelist():
                    sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                    for si in sroot:
                        text = "".join(t.text or "" for t in si.iter() if t.tag.endswith("}t") or t.tag == "t")
                        shared.append(text)
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

    @staticmethod
    def _parse_pptx(file_path: str) -> List[Dict[str, Any]]:
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

    def _parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
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
        try:
            blocks = self._extract_pdf_text_fallback(file_path)
            return blocks
        except Exception:
            return []

    @staticmethod
    def _extract_pdf_text_fallback(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "rb") as f:
            data = f.read()
        texts = []
        for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL):
            raw = m.group(1)
            try:
                decoded = zlib.decompress(raw)
            except Exception:
                continue
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

    def _parse_media(self, file_path: str, doc_type: str, description: str = None) -> List[Dict[str, Any]]:
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        name_without_ext = os.path.splitext(filename)[0]
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