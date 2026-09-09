"""SmartTerminal：TerminalTool × MemoryTool × NoteTool × ContextBuilder 协同封装。

让终端操作具备"记忆"与"上下文感知"能力：
  1. 执行前（build_context）：从笔记（NoteTool+ContextBuilder 类型加权）与
     记忆系统（MemoryTool）检索相关信息，结合会话状态与命令历史，
     经 ContextBuilder 组装为上下文注入提示词
  2. 执行中：委托 TerminalTool（白名单/沙箱/超时/输出限制安全机制）
  3. 执行后（自动沉淀）：
     - 记忆写入：每条命令 → MemoryTool 情景记忆（episodic）
     - 笔记沉淀：成功命令 → 会话 task_state 笔记；失败/超时 → blocker 笔记
     - 长时程协作：新会话可检索到历史阻塞与任务状态（跨会话记忆）

使用方式:
    smart = SmartTerminal(workspace_root="...")
    ctx = smart.build_context("继续优化检索")
    result = smart.execute("dir")          # 自动记忆 + 笔记沉淀
    smart.cd("tools")
"""

import sys
import os
import time
from collections import deque
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


from src.tools.terminal.terminal_tool import TerminalTool
from src.tools.memory.note_tool import NoteTool
from src.tools.memory.note_context_bridge import NoteContextBridge
from src.tools.context.context_builder import ContextPacket, ContextBuilder, ContextConfig
from src.tools.memory.modules import MemoryManager, MemoryEntry


class SmartTerminal:
    """协同终端：安全执行命令 + 记忆/笔记沉淀 + 上下文构建。"""

    def __init__(self, terminal: Optional[TerminalTool] = None,
                 memory: Optional[MemoryManager] = None,
                 notes: Optional[NoteTool] = None,
                 workspace_root: Optional[str] = None,
                 notes_dir: Optional[str] = None,
                 session_id: Optional[str] = None,
                 history_capacity: int = 20):
        self.terminal = terminal or TerminalTool(workspace_root=workspace_root)
        # 记忆系统（情景记忆 + 工作记忆）
        self.memory = memory or MemoryManager(config={
            "enabled_types": ["working", "episodic"],
            "working_capacity": 30,
            "episodic_db_path": ":memory:",
        })
        # 笔记工具 + 桥接（NoteTool × ContextBuilder）
        self.notes = notes or NoteTool(notes_dir=notes_dir)
        self.bridge = NoteContextBridge(note_tool=self.notes,
                                        default_limit=3)
        self.builder = ContextBuilder(ContextConfig(
            max_tokens=600, system_instruction_ratio=0.0,
            min_relevance=0.05, enable_compression=True,
            relevance_weight=0.7, recency_weight=0.3))

        self.session_id = session_id or f"session_{int(time.time())}"
        # 会话状态
        self._log: deque = deque(maxlen=history_capacity)   # (ts, cwd, command, status)
        self._blocker_log: deque = deque(maxlen=10)
        self.session_note_id: Optional[str] = None          # 会话 task_state 笔记
        self.blocker_note_id: Optional[str] = None          # 会话 blocker 笔记
        self._session_log_text = f"会话 {self.session_id} 执行记录（最新在前）:\n"

    # ------------------------------------------------------------
    # 命令执行（自动沉淀记忆与笔记）
    # ------------------------------------------------------------

    def execute(self, command: str, cwd: Optional[str] = None,
                timeout: Optional[float] = None) -> str:
        """执行命令（安全机制），并将结果沉淀到记忆与笔记。"""
        result = self.terminal.execute_command(command, cwd=cwd, timeout=timeout)
        status = self._classify(result)
        self._record(command, cwd, status, result)
        return result

    def cd(self, path: str) -> str:
        """目录导航（记录到会话日志）。"""
        result = self.terminal.handle_cd(path)
        self._record(f"cd {path}", None, self._classify(result), result)
        return result

    def _classify(self, result: str) -> str:
        """从执行结果判断状态。"""
        if "已拒绝执行" in result or result.startswith("🚫"):
            return "rejected"
        if "命令超时" in result:
            return "timeout"
        if "警告（退出码" in result or "执行异常" in result or result.startswith("❌"):
            return "failed"
        return "success"

    def _record(self, command: str, cwd: Optional[str], status: str, result: str):
        """执行后沉淀：记忆写入 + 笔记更新 + 会话日志。"""
        now = datetime.now().isoformat(timespec="seconds")
        cwd = cwd or self.terminal.cwd

        # 1. 记忆写入（情景记忆）
        summary = self._summarize(result)
        importance = {"failed": 0.8, "timeout": 0.8, "rejected": 0.5,
                      "success": 0.4}.get(status, 0.4)
        self.memory.add(MemoryEntry(
            memory_id=f"t_{int(time.time()*1000)}",
            content=f"终端命令[{status}]: {command}（cwd={cwd}）→ {summary}",
            memory_type="episodic",
            importance=importance,
            timestamp=now,
            session_id=self.session_id,
            metadata={"tool": "terminal", "status": status, "cwd": cwd},
        ))

        # 2. 会话日志
        self._log.appendleft((now, cwd, command, status))
        if status in ("failed", "timeout", "rejected"):
            self._blocker_log.appendleft((now, cwd, command, status, summary))

        # 3. 笔记沉淀
        if status == "success":
            self._upsert_session_note()
        elif status in ("failed", "timeout"):
            self._upsert_blocker_note()

    def _upsert_session_note(self):
        """更新会话 task_state 笔记（记录最近命令执行情况）。"""
        lines = [f"会话: {self.session_id}",
                 f"当前目录: {self.terminal.cwd}",
                 f"最近执行（最新在前，最多 {len(self._log)} 条）:"]
        for ts, cwd, cmd, st in list(self._log)[:10]:
            mark = {"success": "✅", "failed": "❌", "timeout": "⏱️",
                    "rejected": "🚫"}.get(st, "•")
            lines.append(f"- [{ts}] {mark} {st} | {cwd} | {cmd}")
        content = "\n".join(lines)

        if self.session_note_id is None:
            self.session_note_id = self.notes.create_note(
                title=f"终端会话状态 {self.session_id[:8]}",
                content=content, note_type="task_state",
                tags=["terminal", "会话"]).split("（")[0].replace("✅ 已创建笔记 ", "")
        else:
            self.notes.update_note(self.session_note_id, content=content)

    def _upsert_blocker_note(self):
        """更新会话 blocker 笔记（记录失败/超时命令）。"""
        lines = [f"会话: {self.session_id}",
                 f"阻塞记录（最新在前，最多 {len(self._blocker_log)} 条）:"]
        for ts, cwd, cmd, st, summary in list(self._blocker_log)[:10]:
            lines.append(f"- [{ts}] {st} | {cmd}（cwd={cwd}）→ {summary}")
        content = "\n".join(lines)

        if self.blocker_note_id is None:
            self.blocker_note_id = self.notes.create_note(
                title=f"终端阻塞记录 {self.session_id[:8]}",
                content=content, note_type="blocker",
                tags=["terminal", "阻塞"]).split("（")[0].replace("✅ 已创建笔记 ", "")
        else:
            self.notes.update_note(self.blocker_note_id, content=content)

    # ------------------------------------------------------------
    # 上下文构建（执行前：笔记 + 记忆 + 会话状态）
    # ------------------------------------------------------------

    def _summarize(self, result: str, limit: int = 80) -> str:
        """提取执行结果摘要。"""
        lines = [l for l in result.splitlines()
                 if l and not l.startswith("=") and not l.startswith("-----")
                 and not l.startswith("  ")]
        text = " | ".join(lines)
        return text[:limit] + ("…" if len(text) > limit else "")

    def _retrieve_memory(self, query: str, limit: int = 3) -> List[ContextPacket]:
        """从记忆系统检索相关记忆（episodic + working）。"""
        packets: List[ContextPacket] = []
        for mem_type in ("episodic", "working"):
            module = self.memory.get_module(mem_type)
            if module is None:
                continue
            try:
                for score, _, entry, _ in module.retrieve(query, limit=limit):
                    ts = None
                    try:
                        ts = datetime.fromisoformat(entry.timestamp).timestamp()
                    except (ValueError, TypeError):
                        ts = None
                    packets.append(ContextPacket(
                        content=entry.content,
                        timestamp=ts,
                        relevance_score=float(score),
                        metadata={"source": "memory", "memory_type": mem_type,
                                  "memory_id": entry.memory_id},
                    ))
            except Exception:
                continue
        return packets

    def build_context(self, query: str = "终端任务", limit: int = 6) -> str:
        """执行前构建上下文：会话状态 + 命令历史 + 相关笔记 + 相关记忆。

        全部经 ContextBuilder 组装（token 预算 / 相关性阈值 / 排序）。
        """
        packets: List[ContextPacket] = []

        # 1. 会话状态（固定高优先）
        packets.append(ContextPacket(
            content=(f"当前终端目录: {self.terminal.cwd}\n"
                     f"会话: {self.session_id} | 历史命令 {len(self._log)} 条"),
            timestamp=time.time(), relevance_score=0.6,
            metadata={"source": "terminal_state"}))

        # 2. 最近命令历史
        for ts, cwd, cmd, st in list(self._log)[:5]:
            packets.append(ContextPacket(
                content=f"[{st}] {cwd} | {cmd}",
                relevance_score=0.3,
                metadata={"source": "terminal_history", "status": st}))

        # 3. 相关笔记（NoteTool × ContextBuilder 类型加权）
        packets.extend(self.bridge.retrieve(query, limit=3))

        # 4. 相关记忆（MemoryTool）
        packets.extend(self._retrieve_memory(query, limit=2))

        return self.builder.build_text(packets)

    # ------------------------------------------------------------
    # 状态与统计
    # ------------------------------------------------------------

    def status(self) -> str:
        """协同终端状态。"""
        return (f"📊 SmartTerminal 状态\n"
                f"   会话: {self.session_id} | 当前目录: {self.terminal.cwd}\n"
                f"   命令历史: {len(self._log)} 条 | 阻塞记录: {len(self._blocker_log)} 条\n"
                f"   记忆条目: {self.memory.get_count()} | "
                f"笔记: {len(self.notes.index)} 条\n"
                f"   会话笔记: {self.session_note_id or '(未创建)'} | "
                f"阻塞笔记: {self.blocker_note_id or '(未创建)'}")

    def history_text(self, n: int = 5) -> str:
        """最近 n 条命令历史文本。"""
        lines = [f"- [{ts}] {st} | {cwd} | {cmd}" for ts, cwd, cmd, st in list(self._log)[:n]]
        return "\n".join(lines) if lines else "（暂无命令历史）"


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import shutil
    import tempfile

    print("=" * 60)
    print("🤝 SmartTerminal 协同终端演示（Terminal×Memory×Note×ContextBuilder）")
    print("=" * 60)

    sandbox = os.path.join(tempfile.gettempdir(), "smart_terminal_demo")
    notes_dir = os.path.join(sandbox, "notes")
    shutil.rmtree(sandbox, ignore_errors=True)
    os.makedirs(sandbox, exist_ok=True)
    with open(os.path.join(sandbox, "config.yaml"), "w", encoding="utf-8") as f:
        f.write("api_key: xxx\nmax_retries: 3\n")
    with open(os.path.join(sandbox, "readme.md"), "w", encoding="utf-8") as f:
        f.write("# 项目说明\nRAG 系统构建指南\n")

    is_win = os.name == "nt"
    ls_cmd = "dir" if is_win else "ls"
    type_cmd = "type" if is_win else "cat"
    pwd_cmd = "echo %cd%" if is_win else "pwd"

    smart = SmartTerminal(workspace_root=sandbox, notes_dir=notes_dir)

    # 1. 执行命令（成功 → 记忆 + 会话笔记）
    print("--- 1) 成功命令执行（自动沉淀记忆/笔记）---")
    print(smart.execute(pwd_cmd))
    print(smart.execute(ls_cmd))
    print(smart.execute(f"{type_cmd} config.yaml"))

    # 2. 失败命令（→ blocker 笔记 + 高重要性记忆）
    print("--- 2) 失败命令（→ blocker 笔记）---")
    print(smart.execute(f"{type_cmd} missing_file.txt"))

    # 3. 被拒绝命令（安全机制）
    print("--- 3) 危险命令被拒绝 ---")
    print(smart.execute("rm readme.md"))

    # 4. 目录导航
    print("--- 4) 目录导航 ---")
    print(smart.cd("tools"))           # 不存在 → 记录
    print(smart.cd(".."))              # 根目录向上 → 沙箱拒绝

    # 5. 执行前上下文构建（协同核心）
    print("--- 5) build_context 执行前上下文（笔记+记忆+会话状态）---")
    ctx = smart.build_context("继续优化 RAG 系统配置", limit=6)
    print(f"   上下文 tokens={smart.builder.estimate_tokens(ctx)}")
    for line in ctx.splitlines()[:14]:
        print(f"     {line[:66]}")

    # 6. 沉淀结果检查
    print("--- 6) 沉淀结果（记忆 + 笔记）---")
    print(f"   记忆条目: {smart.memory.get_count()}")
    for m in smart.memory.get_all()[-3:]:
        print(f"     [{m.memory_type}] {m.content[:44]}...")
    print(f"   笔记: {len(smart.notes.index)} 条")
    print(smart.notes.execute("summary", recent=5))

    # 7. 跨会话记忆（新会话检索历史 blocker / task_state 笔记）
    print("--- 7) 跨会话协作（新 SmartTerminal 检索沉淀的笔记）---")
    smart2 = SmartTerminal(workspace_root=sandbox, notes_dir=notes_dir,
                           session_id="session_new")
    ctx2 = smart2.build_context("终端任务", limit=4)
    print(f"   新会话上下文 tokens={smart2.builder.estimate_tokens(ctx2)}")
    for line in ctx2.splitlines()[:10]:
        print(f"     {line[:66]}")

    print()
    print(smart2.status())
    print()
    print("✅ SmartTerminal 演示完成")