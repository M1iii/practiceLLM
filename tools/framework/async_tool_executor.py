"""
异步工具执行器
==============
基于 ThreadPoolExecutor + asyncio 实现工具的异步与并行执行。
适配项目现有的 ToolRegistry（execute(name, args_dict) 接口）。

使用方式:
    import asyncio
    from tools.framework.async_tool_executor import AsyncToolExecutor
    from tools.search.search_registry import create_search_registry

    registry = create_search_registry()
    executor = AsyncToolExecutor(registry)

    tasks = [
        {"tool_name": "AdvancedSearch", "args": {"query": "Python"}},
        {"tool_name": "Calculator", "args": {"expression": "2 ** 10"}},
    ]
    results = asyncio.run(executor.execute_tools_parallel(tasks))
"""

import sys
import os
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.framework.tool_system import ToolRegistry


class AsyncToolExecutor:
    """
    异步工具执行器。
    通过 ThreadPoolExecutor 将同步的工具调用包装为异步任务，
    支持单个工具异步执行和多个工具并行执行。
    """

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        """
        Args:
            registry: 工具注册表
            max_workers: 线程池最大工作线程数
        """
        self.registry = registry
        self.max_workers = max_workers
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    @property
    def executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """惰性初始化线程池。"""
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            )
        return self._executor

    # ============================================================
    # 单工具异步执行
    # ============================================================

    async def execute_tool_async(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        异步执行单个工具。

        Args:
            tool_name: 工具名称
            args: 工具参数字典
        Returns:
            工具执行结果字符串
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.registry.execute,
            tool_name,
            args,
        )
        return result

    # ============================================================
    # 多工具并行执行
    # ============================================================

    async def execute_tools_parallel(
        self,
        tasks: List[Dict[str, Any]],
        return_details: bool = False,
    ) -> List[Any]:
        """
        并行执行多个工具任务。

        Args:
            tasks: 任务列表，每个任务为 {"tool_name": str, "args": dict}
            return_details: 为 True 时返回包含状态和结果的详情列表
        Returns:
            结果列表（return_details=False）或详情列表（return_details=True）
        """
        print(f"🚀 开始并行执行 {len(tasks)} 个工具任务")

        async_tasks = [
            self._execute_with_status(task, i)
            for i, task in enumerate(tasks)
        ]
        details = await asyncio.gather(*async_tasks)

        success_count = sum(1 for d in details if d["success"])
        fail_count = len(details) - success_count
        print(f"✅ 并行执行完成: {success_count} 成功, {fail_count} 失败")

        if return_details:
            return details
        return [d["result"] for d in details]

    async def _execute_with_status(self, task: Dict[str, Any], index: int) -> Dict[str, Any]:
        """执行单个任务并包装状态信息。"""
        tool_name = task["tool_name"]
        args = task.get("args", {})

        try:
            result = await self.execute_tool_async(tool_name, args)
            preview = result[:80] + "..." if len(result) > 80 else result
            print(f"  📌 任务 {index + 1} [{tool_name}] ✅ {preview}")
            return {"success": True, "result": result, "error": None, "tool_name": tool_name}
        except Exception as e:
            print(f"  📌 任务 {index + 1} [{tool_name}] ❌ {e}")
            return {"success": False, "result": None, "error": str(e), "tool_name": tool_name}

    # ============================================================
    # 多工具顺序执行（异步）
    # ============================================================

    async def execute_tools_sequence(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[str]:
        """
        异步顺序执行多个工具任务（非并行）。
        适用于步骤间有依赖关系的场景。

        Args:
            tasks: 任务列表，每个任务为 {"tool_name": str, "args": dict}
        Returns:
            每个任务的执行结果列表
        """
        print(f"🔗 开始顺序执行 {len(tasks)} 个工具任务")
        results = []

        for i, task in enumerate(tasks):
            tool_name = task["tool_name"]
            args = task.get("args", {})

            print(f"  📌 步骤 {i + 1}/{len(tasks)}: {tool_name}")
            try:
                result = await self.execute_tool_async(tool_name, args)
                results.append(result)
                preview = result[:80] + "..." if len(result) > 80 else result
                print(f"     ✅ {preview}")
            except Exception as e:
                print(f"     ❌ {e}")
                results.append(f"错误: {e}")

        print(f"✅ 顺序执行完成")
        return results

    # ============================================================
    # 资源清理
    # ============================================================

    def shutdown(self):
        """关闭线程池，释放资源。"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            print("🔒 异步执行器线程池已关闭")

    def __del__(self):
        self.shutdown()


# ============================================================
# 演示
# ============================================================

async def demo():
    from tools.search.search_registry import create_search_registry

    registry = create_search_registry()
    executor = AsyncToolExecutor(registry, max_workers=4)

    # --- 测试1: 并行执行多个搜索 ---
    print("=" * 60)
    print("测试1: 并行执行多个搜索任务")
    print("=" * 60)
    search_tasks = [
        {"tool_name": "AdvancedSearch", "args": {"query": "AI Agent 框架", "count": 3}},
        {"tool_name": "AdvancedSearch", "args": {"query": "Python 异步编程", "count": 3}},
    ]
    results = await executor.execute_tools_parallel(search_tasks)
    for i, r in enumerate(results):
        print(f"\n任务 {i + 1} 结果预览:\n{r[:200]}...")
    print()

    # --- 测试2: 并行执行混合任务（搜索 + 计算 + 时间） ---
    print("=" * 60)
    print("测试2: 并行执行混合任务")
    print("=" * 60)
    mixed_tasks = [
        {"tool_name": "AdvancedSearch", "args": {"query": "DeepSeek V4", "count": 2}},
        {"tool_name": "Calculator", "args": {"expression": "(15 + 27) * 3"}},
        {"tool_name": "Calculator", "args": {"expression": "2 ** 10"}},
        {"tool_name": "Time", "args": {}},
    ]
    details = await executor.execute_tools_parallel(mixed_tasks, return_details=True)
    for i, d in enumerate(details):
        status = "✅" if d["success"] else "❌"
        print(f"  任务 {i + 1} [{d['tool_name']}] {status}: {str(d['result'])[:100]}")
    print()

    # --- 测试3: 顺序执行 ---
    print("=" * 60)
    print("测试3: 顺序执行（模拟链式依赖）")
    print("=" * 60)
    seq_tasks = [
        {"tool_name": "Calculator", "args": {"expression": "100 * 2"}},
        {"tool_name": "Time", "args": {}},
    ]
    results = await executor.execute_tools_sequence(seq_tasks)
    for i, r in enumerate(results):
        print(f"  步骤 {i + 1} 结果: {r}")
    print()

    executor.shutdown()


if __name__ == "__main__":
    asyncio.run(demo())
