"""RAG 工具演示脚本。"""

if __name__ == "__main__":
    import os
    import sys
    import tempfile
    import time
    import zlib
    import zipfile
    import types

    from src.tools.rag.rag_tool.tool import RagTool
    from src.tools.rag.rag_tool.loader import MARKITDOWN_AVAILABLE
    from src.tools.rag.rag_tool.splitter import chunk_paragraphs, estimate_tokens, _approx_token_len, _is_cjk
    from src.tools.rag.rag_tool.encoder import VectorEncoder, index_chunks
    from src.tools.rag.rag_tool.index import QueryExpander, HydeGenerator
    from src.core.cache import SafeFullCache

    print("=" * 60)
    print("RAG 工具演示（阿里云百炼）")
    print("=" * 60)
    print()

    demo_dir = os.path.join(tempfile.gettempdir(), "rag_demo")
    os.makedirs(demo_dir, exist_ok=True)

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

    image_path = os.path.join(demo_dir, "architecture_diagram.png")
    with open(image_path, "wb") as f:
        f.write(b"fake-image-bytes")

    audio_path = os.path.join(demo_dir, "meeting_recording.mp3")
    with open(audio_path, "wb") as f:
        f.write(b"fake-audio-bytes")

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

    rag = RagTool()
    print("RAG 管道已初始化")
    print(f"   嵌入 API: {'可用' if rag._index.embedding_available else '不可用（TF-IDF 模式）'}")
    print(f"   转换引擎: {'MarkItDown 统一转换' if MARKITDOWN_AVAILABLE else '轻量回退解析'}")
    print(f"   分块配置: chunk_size=500, overlap=50")
    print()

    print("--- RAG 0: 文档转换预览（MarkItDown 统一转换）---")
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
        print("   docx 样例生成失败，跳过")
    print()
    print("[png] 图片（MarkItDown 无 LLM 时回退元数据语义化）:")
    print(rag.execute("convert", file_path=image_path, description="系统架构示意图，展示 RAG 各组件关系"))
    print()

    print("--- RAG 1: 添加多格式文档 ---")
    print("[1] Markdown:")
    print(rag.add_file(md_path))
    print()
    print("[2] Word (docx):")
    if docx_ok:
        print(rag.add_file(docx_path))
    else:
        print("   docx 样例生成失败，跳过")
    print()
    print("[3] PDF:")
    if pdf_ok:
        print(rag.add_file(pdf_path))
    else:
        print("   PDF 样例生成失败，跳过")
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
        print("   xlsx 样例生成失败，跳过")
    print()
    print("[8] PPT (pptx):")
    if pptx_ok:
        print(rag.add_file(pptx_path))
    else:
        print("   pptx 样例生成失败，跳过")
    print()

    print("--- RAG 2: 知识库统计与列表 ---")
    print(rag.execute("stats"))
    print()
    print(rag.execute("list"))
    print()

    print("--- RAG 3: 智能检索「RAG 的核心流程」---")
    print(rag.execute("search", query="RAG 的核心流程是什么", top_k=3))
    print()

    print("--- RAG 4: LLM 增强问答 ---")
    print(rag.execute("query", question="RAG 相比传统微调有什么优势？", top_k=3, show_context=True))
    print()

    print("--- RAG 5: 语义检索（词汇不重叠）---")
    print(rag.execute("search", query="如何让大模型回答更准确", top_k=3))
    print()

    print("--- RAG 5b: 表格/幻灯片内容检索 ---")
    print("   [search] 「如何减少幻觉」(xlsx 表格):")
    print(rag.execute("search", query="如何减少幻觉", top_k=2))
    print()
    print("   [search] 「RAG 系统概览」(pptx 幻灯片):")
    print(rag.execute("search", query="RAG 系统概览", top_k=2))
    print()

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
    cjk_ext_sample = "𠀀𠀁"
    print(f"     CJK 扩展 B 字符 {cjk_ext_sample!r} → "
          f"{sum(1 for ch in cjk_ext_sample if _is_cjk(ch))} CJK tokens")
    print()
    for tc in token_chunks:
        print(f"   块 {tc['chunk_index']}: tokens={tc['tokens']} (≤60) | "
              f"段落数={tc['paragraph_count']} | {tc['content'][:60]}...")
    print()
    print("   重叠连续性验证（相邻块首尾是否有重复内容）:")
    for ci in range(1, len(token_chunks)):
        prev_tail = token_chunks[ci - 1]["content"][-20:]
        cur_head = token_chunks[ci]["content"][:20]
        overlap_detect = "有重叠" if prev_tail in token_chunks[ci]["content"] else "无重叠"
        print(f"     块{ci-1}尾部: ...{prev_tail} | 块{ci}头部: {cur_head}... → {overlap_detect}")
    print()

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
    fake_client = types.SimpleNamespace(is_available=False, embed=lambda t: None, embed_batch=lambda ts: [None] * len(ts))
    encoder_offline = VectorEncoder(embedding_client=fake_client)
    print(f"     embedding_available={encoder_offline.embedding_available}")
    for txt in ["检索增强生成示例"]:
        dense, sparse, method = encoder_offline.encode(txt)
        print(f"     encode('{txt}') → method={method}, dense={'None' if dense is None else len(dense)}, "
              f"tfidf_terms={len(sparse)}")
    print()

    print("--- RAG 5f: 多查询扩展检索（MQE）---")
    expander = QueryExpander(llm_client=None, threshold=5, batch_size=10)
    print("   [QueryExpander] 独立 LLM 生成扩展查询:")
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
    print("   [search_mqe] 实际检索:")
    print(rag.execute("search_mqe", query="RAG 相比微调有什么好处", top_k=3, n_queries=3))
    print()
    print("   [query + mqe] 增强问答:")
    print(rag.execute("query", question="RAG 的核心步骤有哪些？", top_k=2, mqe=True))
    print()

    print("--- RAG 5g: 假设文档嵌入（HyDE）---")
    hyde = HydeGenerator(llm_client=None)
    print("   [HydeGenerator] 独立 LLM 生成假设文档:")
    hypo = hyde.generate("RAG 相比传统微调的优势")
    print(f"     假设文档: {hypo}")
    print()
    print("   [兜底机制] LLM 不可用 → 回退原始查询:")
    broken_hyde_llm = _types.SimpleNamespace(chat=lambda m, temperature=0.3: "")
    hyde_offline = HydeGenerator(llm_client=broken_hyde_llm)
    fallback_hypo = hyde_offline.generate("测试查询")
    print(f"     generate('测试查询') → '{fallback_hypo}'"
          + ("（回退原始查询）" if fallback_hypo == "测试查询" else ""))
    print()
    print("   [search_hyde] 实际检索:")
    print(rag.execute("search_hyde", query="RAG 为什么能减少幻觉", top_k=3))
    print()
    print("   [query + hyde] 增强问答:")
    print(rag.execute("query", question="RAG 的核心流程是什么？", top_k=2, hyde=True))
    print()

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
    print("   [场景3: 性能敏感 → 基础检索] search_expanded:")
    print(rag.execute("search_expanded", query="RAG 如何减少大模型幻觉", top_k=4))
    print()
    print("   [query 统一框架] 专业领域查询（enable_mqe + enable_hyde）:")
    print(rag.execute("query", question="RAG 相比微调有哪些优势？",
                      top_k=2, enable_mqe=True, enable_hyde=True))
    print()

    print("--- RAG 5i: 缓存层（SafeFullCache）---")
    print("   [基础读写 + TTL] set/get/过期:")
    ttl_cache = SafeFullCache(max_size=100, default_ttl=1)
    ttl_cache.set("天气", "晴天", ttl=1)
    print(f"     get('天气') → {ttl_cache.get('天气')}")
    time.sleep(1.2)
    print(f"     1.2s 后 get('天气') → {ttl_cache.get('天气')}（已过期淘汰）")
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
    print(f"     二次: {gos_cache.get_or_set('x', compute)}（compute 调用 {calls['n']} 次，命中缓存）")
    print()
    print("   [MQE/HyDE 集成] 重复调用命中缓存:")
    print(f"     调用前缓存统计: {rag._cache.stats()}")
    rag.execute("search_mqe", query="RAG 相比微调的优势", top_k=3, n_queries=3)
    rag.execute("search_hyde", query="RAG 相比微调的优势", top_k=3)
    after_first = rag._cache.stats()
    rag.execute("search_mqe", query="RAG 相比微调的优势", top_k=3, n_queries=3)
    rag.execute("search_hyde", query="RAG 相比微调的优势", top_k=3)
    after_second = rag._cache.stats()
    print(f"     首轮调用后: size={after_first['size']} 命中={after_first['hits']}")
    print(f"     二轮调用后: size={after_second['size']} 命中={after_second['hits']} "
          f"（新增命中 {after_second['hits'] - after_first['hits']} 次，未新增缓存项）")
    print()
    print("   [QA 答案缓存] 相同问题二次查询命中缓存:")
    q1 = rag.execute("query", question="RAG 的核心流程是什么？", top_k=2)
    s1 = rag._cache.stats()
    q2 = rag.execute("query", question="RAG 的核心流程是什么？", top_k=2)
    s2 = rag._cache.stats()
    print(f"     两次回答一致: {q1 == q2} | 缓存命中数 {s1['hits']} → {s2['hits']}（+{s2['hits'] - s1['hits']}）")
    print(f"     当前缓存统计: {rag._cache.stats()}")
    print()

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
    print("RAG 工具演示完成")
    print("=" * 60)