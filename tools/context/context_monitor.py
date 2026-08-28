"""上下文构建监控日志（ContextMonitor）。

记录每次上下文构建的统计信息，便于后续优化：
  - 基本信息：时间戳 / 查询 / 缓存命中标志
  - 规模指标：Gather 候选数、Select 选中数、来源分布
  - token 指标：预算、实际使用、token 使用率、系统指令占用
  - 质量指标：选中包平均综合分 / 相关性 / 新近度
  - 性能指标：各阶段耗时（ms）与总耗时
  - 压缩信息：是否发生压缩

输出：JSONL 日志文件（默认 logs/context_monitor.jsonl）+ 内存环形缓冲，
提供最近记录与聚合统计（平均值 / 分布 / 命中率 / 压缩率）。

使用方式:
    monitor = ContextMonitor(log_path="logs/context_monitor.jsonl")
    monitor.record({"query": "...", "tokens_used": 320, ...})
    monitor.get_recent(5)
    monitor.get_aggregate()
"""

import sys
import os
import json
import uuid
from collections import deque, Counter
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEFAULT_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "logs", "context_monitor.jsonl")


class ContextMonitor:
    """上下文构建监控：JSONL 落盘 + 内存环形缓冲 + 聚合统计。"""

    def __init__(self, log_path: Optional[str] = DEFAULT_LOG_PATH,
                 max_entries: int = 2000):
        self.log_path = log_path
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.max_entries = max(1, max_entries)
        self._entries: deque = deque(maxlen=self.max_entries)
        self._write_count = 0
        self._write_failures = 0

    # ------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------

    def record(self, stats: Dict[str, Any]) -> str:
        """记录一次上下文构建的统计信息，返回记录 ID。

        Args:
            stats: 统计字段（query / gather_count / selected_count /
                   tokens_used / token_utilization / ...）
        """
        entry = {
            "id": str(uuid.uuid4())[:8],
            "ts": datetime.now().isoformat(timespec="seconds"),
            **stats,
        }
        self._entries.append(entry)
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._write_count += 1
            except OSError:
                self._write_failures += 1
        return entry["id"]

    # ------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """返回最近 n 条记录（内存缓冲，最新在后）。"""
        return list(self._entries)[-max(0, n):]

    def get_aggregate(self) -> Dict[str, Any]:
        """聚合统计：平均值 / 命中率 / 压缩率 / 来源分布。"""
        if not self._entries:
            return {"entries": 0}
        n = len(self._entries)

        def avg(key: str, default: float = 0.0) -> float:
            vals = [e.get(key) for e in self._entries if e.get(key) is not None]
            return sum(vals) / len(vals) if vals else default

        # 来源分布汇总（Gather 阶段）
        sources: Counter = Counter()
        for e in self._entries:
            for src, count in (e.get("gather_sources") or {}).items():
                sources[src] += count

        return {
            "entries": n,
            "avg_gather_count": round(avg("gather_count"), 2),
            "avg_selected_count": round(avg("selected_count"), 2),
            "avg_tokens_used": round(avg("tokens_used"), 1),
            "avg_token_utilization": round(avg("token_utilization"), 4),
            "avg_score": round(avg("avg_score"), 4),
            "avg_total_time_ms": round(avg("total_time_ms"), 1),
            "cache_hit_rate": round(sum(1 for e in self._entries if e.get("cache_hit")) / n, 4),
            "compressed_rate": round(sum(1 for e in self._entries if e.get("compressed")) / n, 4),
            "source_distribution": dict(sources),
        }

    def stats(self) -> Dict[str, Any]:
        """监控器状态（内存 / 落盘统计）。"""
        return {
            "entries_in_memory": len(self._entries),
            "max_entries": self.max_entries,
            "log_path": self.log_path,
            "writes": self._write_count,
            "write_failures": self._write_failures,
        }


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📊 ContextMonitor 上下文构建监控演示")
    print("=" * 60)

    import tempfile
    monitor = ContextMonitor(
        log_path=os.path.join(tempfile.gettempdir(), "ctx_monitor_demo.jsonl"),
        max_entries=100)

    # 模拟记录 3 次上下文构建
    samples = [
        {"query": "RAG 检索增强生成", "cache_hit": False, "gather_count": 11,
         "gather_sources": {"system": 1, "rag": 2, "memory": 3, "history": 4, "custom": 1},
         "selected_count": 4, "system_tokens": 20, "tokens_used": 89,
         "max_tokens": 200, "token_utilization": 0.445, "compressed": False,
         "avg_score": 0.615, "total_time_ms": 32.5},
        {"query": "多查询扩展如何提升召回", "cache_hit": True, "gather_count": 11,
         "gather_sources": {"system": 1, "rag": 2, "memory": 3, "history": 4, "custom": 1},
         "selected_count": 4, "system_tokens": 20, "tokens_used": 89,
         "max_tokens": 200, "token_utilization": 0.445, "compressed": False,
         "avg_score": 0.615, "total_time_ms": 1.2},
        {"query": "超长查询触发压缩", "cache_hit": False, "gather_count": 20,
         "gather_sources": {"system": 1, "rag": 8, "memory": 6, "history": 5},
         "selected_count": 12, "system_tokens": 50, "tokens_used": 450,
         "max_tokens": 500, "token_utilization": 0.9, "compressed": True,
         "avg_score": 0.58, "total_time_ms": 210.0},
    ]
    for s in samples:
        rid = monitor.record(s)
        print(f"   已记录 {rid}: {s['query'][:18]}... tokens={s['tokens_used']}/{s['max_tokens']}")

    print()
    print("--- 最近 2 条记录 ---")
    for e in monitor.get_recent(2):
        print(f"   {e['ts']} | {e['query'][:16]}... | 选中{e['selected_count']} | "
              f"利用率{e['token_utilization']:.0%} | 缓存命中={e['cache_hit']}")

    print()
    print("--- 聚合统计 ---")
    agg = monitor.get_aggregate()
    for k, v in agg.items():
        print(f"   {k}: {v}")

    print()
    print("--- 监控器状态 ---")
    print(f"   {monitor.stats()}")
    print()
    print("✅ ContextMonitor 演示完成")
