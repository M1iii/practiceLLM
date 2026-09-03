"""安全终端工具（TerminalTool）：命令执行与目录导航。

安全机制：
  1. 命令白名单：只允许安全的只读命令，任何可能修改系统的操作立即拒绝
  2. 工作目录沙箱：只能访问指定工作目录及其子目录
  3. 超时控制：每个命令有执行时间限制，防止无限循环/资源耗尽
  4. 输出大小限制：限制输出大小，防止内存溢出

功能：
  - execute_command：cwd 感知执行，合并 stderr，非零退出码标记警告，
    超时/异常妥善处理（不崩溃）
  - handle_cd：目录导航（cd - 切换上一目录），处理 ~ ~/ - .. 特殊路径，
    沙箱检查 + 存在性检查（结果缓存）

使用方式:
    terminal = TerminalTool(workspace_root="d:/agent练习/practice")
    terminal.execute("execute", command="ls")
    terminal.execute("cd", path="tools")
"""

import sys
import os
import re
import time
import subprocess
from typing import List, Dict, Any, Optional, Tuple


from src.tools.framework.tool_system import Tool, ToolParameter, dual_protocol_execute
from src.core.cache import SafeFullCache   # 复用缓存层（路径检查结果缓存）

# 只读命令白名单（Windows + Unix 常用只读命令）
READONLY_COMMANDS = {
    # 目录与列表
    "ls", "dir", "pwd", "cd", "tree",
    # 文件读取
    "cat", "type", "head", "tail", "less", "more", "stat", "wc", "file", "fc", "comp",
    # 搜索
    "grep", "findstr", "find", "where", "which", "rg",
    # 系统只读信息
    "date", "time", "whoami", "hostname", "uname", "ver",
    "ipconfig", "ifconfig", "netstat", "ps", "tasklist", "systeminfo",
    "env", "printenv", "echo", "ping",
}

# 禁止词（子串匹配，覆盖所有修改类操作）
BLOCKLIST = (
    "rm ", "rmdir", "erase", "del ", "mkdir", "md ", "rd ", "touch", "mv ", "ren ",
    "move ", "copy ", "cp ", "xcopy", "robocopy", "chmod", "chown", "chgrp",
    "kill", "pkill", "taskkill", "sudo", "su ", "shutdown", "reboot", "format",
    "mkfs", "dd ", "mount", "umount", "passwd", "useradd", "usermod", "groupadd",
    "apt", "yum", "dnf", "brew", "pip", "pip3", "conda", "npm", "yarn", "pnpm",
    "cnpm", "npx", "go install", "cargo", "scp", "sftp", "wget", "curl", "iwr",
    "invoke-webrequest", "invoke-expression", "iex ", "start-process",
    "set-content", "add-content", "out-file", "remove-item", "new-item",
    "copy-item", "move-item", "clear-content", "net user", "net localgroup",
    "reg add", "reg delete", "schtasks", "powershell", "pwsh", "bash", "sh ", "zsh",
    "git clone", "git push", "git commit", "git add", "git rm", "git reset",
    "git checkout", "git stash", "git clean",
)

# 链式/重定向元字符：任何包含它们的命令直接拒绝（防止绕过白名单）
_METACHARS = re.compile(r'[><|;&`]|\$\(')


class TerminalTool(Tool):
    """安全终端工具：白名单 + 沙箱 + 超时 + 输出限制。"""

    def __init__(self, workspace_root: Optional[str] = None,
                 allow_cd: bool = True,
                 default_timeout: float = 15.0,
                 max_output_chars: int = 5000,
                 path_cache_ttl: int = 30):
        self.workspace_root = os.path.abspath(
            workspace_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        os.makedirs(self.workspace_root, exist_ok=True)
        self.cwd = self.workspace_root
        self._prev_dir: Optional[str] = None
        self.allow_cd = allow_cd
        self.default_timeout = max(0.5, float(default_timeout))
        self.max_output_chars = max(100, int(max_output_chars))
        # 路径检查结果缓存（30s）
        self._path_cache = SafeFullCache(max_size=200, default_ttl=path_cache_ttl)
        # 执行统计
        self.executed_count = 0
        self.rejected_count = 0
        self.timeout_count = 0

    # ------------------------------------------------------------
    # 安全机制
    # ------------------------------------------------------------

    def _in_sandbox(self, path: str) -> bool:
        """检查路径是否在工作目录沙箱内。"""
        path = os.path.abspath(path)
        root = os.path.abspath(self.workspace_root)
        return path == root or path.startswith(root + os.sep)

    def _validate_command(self, command: str) -> Tuple[bool, str]:
        """校验命令：白名单 + 禁词 + 元字符 + 路径沙箱。"""
        cmd = command.strip()
        if not cmd:
            return False, "空命令"
        if "\n" in cmd or "\r" in cmd:
            return False, "不允许换行"
        if _METACHARS.search(cmd):
            return False, "包含管道/重定向/链式操作符（> < | ; & ` $(）"

        low = cmd.lower()
        for word in BLOCKLIST:
            if word in low:
                return False, f"包含被禁止的操作 '{word.strip()}'"

        name = cmd.split()[0].lower()
        if name not in READONLY_COMMANDS:
            return False, f"命令 '{name}' 不在只读白名单"

        # 尽力而为的路径沙箱检查（跳过 flags，检查路径状参数）
        for token in cmd.split()[1:]:
            if token.startswith("-"):
                continue
            is_path = (os.sep in token or "/" in token or token in (".", "..")
                       or token.endswith((".txt", ".md", ".py", ".json", ".csv",
                                          ".log", ".xlsx", ".docx", ".png", ".jpg",
                                          ".yml", ".yaml", ".html", ".ini")))
            if not is_path:
                continue
            resolved = os.path.abspath(os.path.join(self.cwd, token))
            if not self._in_sandbox(resolved):
                return False, f"路径 '{token}' 超出工作目录沙箱"
        return True, "ok"

    # ------------------------------------------------------------
    # 命令执行
    # ------------------------------------------------------------

    def execute_command(self, command: str, cwd: Optional[str] = None,
                        timeout: Optional[float] = None,
                        max_output: Optional[int] = None) -> str:
        """执行只读命令（cwd 感知、合并 stderr、退出码检查、容错）。"""
        # 1. 白名单校验（立即拒绝）
        ok, reason = self._validate_command(command)
        if not ok:
            self.rejected_count += 1
            return (f"🚫 已拒绝执行（安全策略）: {reason}\n"
                    f"   命令: {command}")

        # 2. cd 命令路由到 handle_cd
        stripped = command.strip()
        if stripped.lower() == "cd" or stripped.lower().startswith("cd "):
            return self.handle_cd(stripped[2:].strip() if len(stripped) > 2 else "",
                                  cwd=cwd)

        # 3. 工作目录沙箱（cwd 必须在沙箱内）
        target = os.path.abspath(cwd or self.cwd)
        if not self._in_sandbox(target):
            return f"🚫 目标目录 '{target}' 超出工作目录沙箱"
        if not os.path.isdir(target):
            return f"❌ 目录不存在: {target}"

        # 4. 执行（超时控制 + 容错）
        time_limit = timeout if timeout is not None else self.default_timeout
        try:
            start = time.perf_counter()
            proc = subprocess.run(
                command, shell=True, cwd=target,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=time_limit)
            elapsed = time.perf_counter() - start
        except subprocess.TimeoutExpired as e:
            self.timeout_count += 1
            partial = (e.stdout or "")
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            # 部分输出同样受大小限制（防止内存/输出过大）
            limit = max_output or self.max_output_chars
            shown = len(partial)
            truncated = False
            if len(partial) > limit:
                partial = partial[:limit] + "\n…[部分输出已截断]"
                truncated = True
            return (f"⏱️ 命令超时（>{time_limit:g}s 已终止）"
                    f"（已输出 {shown} 字符{'，已截断' if truncated else ''}）\n"
                    f"   命令: {command}\n----- 部分输出 -----\n{partial}")
        except Exception as e:
            return (f"⚠️ 命令执行异常（已妥善处理，智能体不中断）: {type(e).__name__}: {e}\n"
                    f"   命令: {command}")

        # 5. 合并标准输出与标准错误
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        merged = stdout
        if stderr:
            merged = f"{stdout}\n[stderr]\n{stderr}" if stdout else f"[stderr]\n{stderr}"

        # 6. 输出大小限制
        limit = max_output or self.max_output_chars
        truncated = False
        if len(merged) > limit:
            merged = merged[:limit] + "\n…[输出已截断，完整输出过大]"
            truncated = True

        # 7. 返回码检查
        self.executed_count += 1
        if proc.returncode == 0:
            status = "成功"
        else:
            status = f"⚠️ 警告（退出码 {proc.returncode}）"

        return (f"===== 命令执行 =====\n"
                f"命令: {command}\n"
                f"目录: {target}\n"
                f"状态: {status} | 耗时: {elapsed:.2f}s\n"
                f"输出: {len(stdout)} 字符{'（已截断）' if truncated else ''}\n"
                f"----- 输出 -----\n{merged}")

    # ------------------------------------------------------------
    # 目录导航
    # ------------------------------------------------------------

    def handle_cd(self, path: str, cwd: Optional[str] = None) -> str:
        """目录导航：cd - 切换上一目录；处理 ~ ~/ - .. 特殊路径。

        沙箱检查 + 目录存在性检查（结果缓存）。
        """
        # 若不允许 cd 或 cd 无参数 → 直接返回（不执行）
        if not self.allow_cd:
            return "cd 未执行（allow_cd=False，已禁用目录导航）"
        if not path:
            return "cd 未执行（无参数，用法: cd <目录> | cd - | cd ~ | cd ..）"

        current = os.path.abspath(cwd or self.cwd)

        # 特殊路径处理：- 上一个目录
        if path == "-":
            if self._prev_dir is None:
                return "❌ 没有上一个目录（首次 cd 无法使用 cd -）"
            target = self._prev_dir
        else:
            # ~ / ~/ → 沙箱根目录（沙箱内"家目录"）
            p = path
            if p == "~":
                p = ""
            elif p.startswith("~/"):
                p = p[2:]
            target = os.path.abspath(os.path.join(current, p))   # 支持 .. 等相对路径

        # 沙箱检查：不能跳出工作目录
        if not self._in_sandbox(target):
            return f"🚫 目标目录 '{path}' 超出工作目录沙箱（仅可访问 {self.workspace_root} 内）"

        # 目录存在性检查（缓存结果 30s）
        cache_key = f"cd:{target}"
        cached = self._path_cache.get(cache_key)
        if cached is None:
            exists = os.path.isdir(target)
            self._path_cache.set(cache_key, exists)
        else:
            exists = cached
        if not exists:
            return f"❌ 目录不存在: {target}"

        # 切换：记录上一目录
        self._prev_dir = current
        self.cwd = target
        return f"✅ 已切换到 {target}"

    # ------------------------------------------------------------
    # Tool 接口
    # ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "TerminalTool"

    @property
    def description(self) -> str:
        return (
            "安全终端工具：白名单只读命令 + 工作目录沙箱 + 超时/输出限制，"
            "支持 cwd 感知命令执行与目录导航（cd - / ~ / ..）。"
            "适用：查看文件/目录列表与内容（ls/cat/type）、系统只读信息（pwd/date/whoami）、"
            "代码搜索（grep/rg/findstr）。"
            "不适用：任何文件修改操作（rm/mkdir/mv/chmod 等会被拒绝）、"
            "需管理员权限或交互式命令。"
            "注意：命令必须在沙箱根目录内执行，修改类操作立即拒绝。"
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="command", type="string",
                          description="要执行的只读命令（如 ls/pwd/type/cat）", required=False),
            ToolParameter(name="path", type="string",
                          description="cd 目标目录（支持 - ~ ~/ ..）", required=False),
            ToolParameter(name="cwd", type="string",
                          description="命令执行目录（默认当前目录，必须在沙箱内）", required=False),
            ToolParameter(name="timeout", type="number",
                          description="超时秒数（默认 15s）", required=False, default=15),
        ]

    def execute(self, action, **kwargs):
        """统一入口：兼容旧协议 execute(action, **kwargs) 与新协议 execute(args)。"""
        return dual_protocol_execute(self, action, **kwargs)

    def run(self, args: Dict[str, Any]) -> str:
        action = (args.get("action") or "execute").lower()
        if action == "cd":
            return self.handle_cd(args.get("path", ""))
        if action == "status":
            return (f"📊 TerminalTool 状态\n"
                    f"   沙箱根目录: {self.workspace_root}\n"
                    f"   当前目录: {self.cwd}\n"
                    f"   执行: {self.executed_count} 次 | 拒绝: {self.rejected_count} 次 | "
                    f"超时: {self.timeout_count} 次\n"
                    f"   白名单命令: {len(READONLY_COMMANDS)} 个 | allow_cd: {self.allow_cd}")
        if action == "execute":
            command = (args.get("command") or "").strip()
            if not command:
                return "❌ execute 需要提供 command"
            return self.execute_command(
                command, cwd=args.get("cwd"),
                timeout=args.get("timeout"))
        return f"❌ 未知操作: '{action}'，当前支持: execute, cd, status"


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import shutil
    import tempfile

    print("=" * 60)
    print("🛡️ TerminalTool 安全终端工具演示")
    print("=" * 60)

    # 沙箱工作目录
    sandbox = os.path.join(tempfile.gettempdir(), "terminal_tool_demo")
    shutil.rmtree(sandbox, ignore_errors=True)
    os.makedirs(sandbox)
    os.makedirs(os.path.join(sandbox, "sub"), exist_ok=True)
    with open(os.path.join(sandbox, "data.txt"), "w", encoding="utf-8") as f:
        f.write("RAG 检索增强生成示例数据\nMQE 与 HyDE 检索策略\n")
    with open(os.path.join(sandbox, "notes.md"), "w", encoding="utf-8") as f:
        f.write("超长内容。" * 1500)   # 触发输出截断
    with open(os.path.join(sandbox, "sub", "inner.txt"), "w", encoding="utf-8") as f:
        f.write("子目录文件\n")

    shell = TerminalTool(workspace_root=sandbox)
    is_win = os.name == "nt"
    ls_cmd = "dir" if is_win else "ls"
    type_cmd = "type" if is_win else "cat"
    pwd_cmd = "echo %cd%" if is_win else "pwd"   # Windows 下用 cmd 内建打印当前目录

    # 1. 安全机制：白名单拒绝
    print("--- 1) 安全机制（白名单 / 禁词 / 元字符）---")
    for bad in ["rm data.txt", "mkdir newdir", "type data.txt > copy.txt",
                "ls | wc -l", "git push origin main", "powershell Remove-Item *"]:
        print(shell.execute_command(bad))
    print()

    # 2. 只读命令执行
    print("--- 2) execute_command 只读命令执行 ---")
    print(shell.execute_command(pwd_cmd))
    print(shell.execute_command(ls_cmd))
    print(shell.execute_command(f"{type_cmd} data.txt"))

    # 2b. cwd 参数感知
    print("--- 2b) cwd 参数（在子目录中执行）---")
    print(shell.execute_command(ls_cmd, cwd=os.path.join(sandbox, "sub")))
    print(shell.execute_command(pwd_cmd, cwd=os.path.join(sandbox, "sub")))

    # 3. 非零退出码（警告）
    print("--- 3) 非零退出码标记警告 ---")
    print(shell.execute_command(f"{type_cmd} nonexistent_file.txt"))

    # 4. 超时控制（超大文件读取，timeout=1s）
    print("--- 4) 超时控制（timeout=1s 读取超大文件）---")
    with open(os.path.join(sandbox, "big.md"), "w", encoding="utf-8") as f:
        f.write("x" * (50 * 1024 * 1024))   # 50MB，读取必然超过 1s
    print(shell.execute_command(f"{type_cmd} big.md", timeout=1.0))

    # 5. 输出大小限制
    print("--- 5) 输出大小限制（max_output=300）---")
    print(shell.execute_command(f"{type_cmd} notes.md", max_output=300))

    # 6. 目录导航
    print("--- 6) handle_cd 目录导航 ---")
    print(shell.handle_cd("sub"))
    print(shell.execute_command("pwd"))
    print(shell.handle_cd("-"))                 # 返回上一目录
    print(shell.handle_cd("~"))                 # 沙箱根
    print(shell.handle_cd(".."))                # 根目录向上 → 沙箱外拒绝
    print(shell.handle_cd("nonexistent"))       # 目录不存在
    print(shell.handle_cd("../../.."))          # 沙箱外 → 拒绝
    print(shell.handle_cd(""))                  # 无参数 → 返回
    print(shell.handle_cd(os.path.join(tempfile.gettempdir(), "outside")))

    # 7. 状态统计
    print("--- 7) 状态统计 ---")
    print(shell.execute("status"))
    print()
    print("✅ TerminalTool 演示完成")