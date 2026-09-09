"""NoteTool 结构化笔记工具演示脚本。"""

if __name__ == "__main__":
    import os
    import shutil
    import tempfile

    from src.tools.memory.note_tool import NoteTool

    print("=" * 60)
    print("NoteTool 结构化笔记工具演示")
    print("=" * 60)

    demo_dir = os.path.join(tempfile.gettempdir(), "note_tool_demo")
    shutil.rmtree(demo_dir, ignore_errors=True)
    tool = NoteTool(notes_dir=demo_dir)

    print("--- 1) create_note 创建笔记 ---")
    print(tool.execute("create_note", title="任务状态：构建 RAG 检索",
                       content="## 进行中\n- 完成向量索引\n- 下一步接入重排序",
                       note_type="task_state", tags=["rag", "进行中"]))
    print(tool.execute("create_note", title="结论：MQE 优于基础检索",
                       content="多查询扩展将最佳相关分从 0.479 提升到 0.645。",
                       note_type="conclusion", tags=["rag", "检索增强"]))
    print(tool.execute("create_note", title="阻塞：嵌入 API 限流",
                       content="text-embedding-v3 触发 QPS 限制，需加退避重试。",
                       note_type="blocker", tags=["api", "阻塞"]))
    print(tool.execute("create_note", title="行动：接入 GSSC 流水线",
                       content="将缓存层与监控日志接入上下文构建。",
                       note_type="action", tags=["gssc", "行动"]))
    note_ids = list(tool.index.keys())
    first_id = note_ids[0]
    print(f"   笔记 id 示例: {first_id}")

    print("\n--- 2) read_note 读取笔记（60s 缓存）---")
    print(tool.execute("read_note", note_id=first_id))
    tool.execute("read_note", note_id=first_id)
    print(f"   读取缓存（二次读取命中）: hits={tool._note_cache.stats()['hits']}")

    print("--- 3) update_note 更新笔记 ---")
    print(tool.execute("update_note", note_id=first_id,
                       content="## 进行中\n- 完成向量索引\n- 完成混合检索\n- 下一步接入重排序",
                       tags=["rag", "进行中", "v2"]))

    print("--- 4) search_notes 搜索（相关性+时间混合排序）---")
    print(tool.execute("search_notes", query="RAG 检索", limit=5))
    print("   [热门词缓存] 重复搜索同一 query:")
    print(tool.execute("search_notes", query="RAG 检索", limit=5))

    print("\n--- 5) list_notes 列出（类型过滤 + 分页）---")
    print(tool.execute("list_notes", limit=10))
    print(tool.execute("list_notes", note_type="conclusion", limit=5))

    print("--- 6) summary 摘要（类型/标签统计 + 最近 n 条）---")
    print(tool.execute("summary", recent=3))

    print("\n--- 6b) 缓存命中验证（删除前）---")
    tool.execute("list_notes", limit=10)
    tool.execute("summary", recent=3)
    print(f"   读取缓存: size={tool._note_cache.stats()['size']} hits={tool._note_cache.stats()['hits']}")
    print(f"   搜索缓存（热门词）: size={tool._search_cache.stats()['size']} "
          f"hits={tool._search_cache.stats()['hits']}")
    print(f"   列表缓存: size={tool._list_cache.stats()['size']} "
          f"hits={tool._list_cache.stats()['hits']}")
    print(f"   摘要缓存: size={tool._summary_cache.stats()['size']} "
          f"hits={tool._summary_cache.stats()['hits']}")

    print("\n--- 7) delete_note 删除笔记（清除相关缓存）---")
    print(tool.execute("delete_note", note_id=first_id))
    print(f"   删除后索引数: {len(tool.index)}")
    print(f"   删除后缓存: 读取 size={tool._note_cache.stats()['size']} | "
          f"搜索 size={tool._search_cache.stats()['size']} | "
          f"列表 size={tool._summary_cache.stats()['size']} | "
          f"摘要 size={tool._summary_cache.stats()['size']}（均清空）")

    print("\n--- 缓存状态 ---")
    print(f"   读取缓存: {tool._note_cache.stats()}")
    print(f"   搜索缓存: {tool._search_cache.stats()}")
    print(f"   列表缓存: {tool._list_cache.stats()}")
    print(f"   摘要缓存: {tool._summary_cache.stats()}")
    print()
    print("NoteTool 演示完成")