"""
VLMediaTool: 多模态媒体分析工具（图像、视频、混合输入）。

基于 qwen-vl-plus 模型，支持：
  - 图像分析：单张/多张图片的内容描述、识别、问答
  - 视频分析：自动提取关键帧，分析视频场景、动作、事件
  - 混合输入：文本 + 图片组合分析

纯文本分析由本地 LLM 处理，不经过 VL 模型。

用法：
    from src.tools.vl_media.vl_media_tool import VLMediaTool

    tool = VLMediaTool()
    # 分析图片
    result = tool.run({
        "action": "analyze_image",
        "file_path": "path/to/image.jpg",
        "prompt": "这张图片里有什么？",
    })
    # 分析视频
    result = tool.run({
        "action": "analyze_video",
        "file_path": "path/to/video.mp4",
        "prompt": "视频中发生了什么？",
    })
"""

import os
import logging
from typing import List, Optional

from src.tools.framework.tool_system import Tool, ToolParameter
from src.tools.framework.tool_response import ToolResponse
from src.tools.vl_media.vl_client import VLClient

logger = logging.getLogger(__name__)


class VLMediaTool(Tool):
    """多模态媒体分析工具（基于 qwen-vl-plus）。

    支持 action:
      - analyze_image: 分析单张/多张图片
      - analyze_video: 分析视频（自动提取关键帧）
      - analyze_mixed: 分析混合输入（文本 + 图片）

    纯文本内容由本地 LLM 处理，不经过此工具。
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self._model = config.get("vl_model")
        self._api_key = config.get("vl_api_key")
        self._base_url = config.get("vl_base_url")
        self._temperature = config.get("temperature", 0.3)
        self._max_tokens = config.get("max_tokens", 4096)
        self._client: VLClient | None = None

    @property
    def name(self) -> str:
        return "VLMediaTool"

    @property
    def description(self) -> str:
        return (
            "多模态媒体分析工具，支持分析图片、视频和混合输入（文本+图片）。"
            "基于 qwen-vl-plus 视觉语言模型，能够理解图像内容、识别场景和物体、"
            "分析视频中的动作和事件。"
            "适用：图片内容分析、视频内容理解、图文混合分析场景。"
            "不适用：纯文本问答（请用本地 LLM）、音频分析（不支持）。"
            "注意：需要配置 DASHSCOPE_API_KEY，视频文件过大时请先压缩。"
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型",
                enum=["analyze_image", "analyze_video", "analyze_mixed"],
                required=True,
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="图片或视频文件路径（analyze_image/analyze_video 需要）",
                required=False,
            ),
            ToolParameter(
                name="file_paths",
                type="array",
                description="多个文件路径列表（analyze_image 多张图片、analyze_mixed 的图片列表）",
                required=False,
                items=ToolParameter(
                    name="path",
                    type="string",
                    description="文件路径",
                    required=True,
                ),
            ),
            ToolParameter(
                name="prompt",
                type="string",
                description="分析提示词，描述你希望工具关注什么",
                required=False,
            ),
            ToolParameter(
                name="texts",
                type="array",
                description="文本上下文列表（analyze_mixed 使用，作为额外文本信息）",
                required=False,
                items=ToolParameter(
                    name="text",
                    type="string",
                    description="文本内容",
                    required=True,
                ),
            ),
            ToolParameter(
                name="system_prompt",
                type="string",
                description="系统提示词，设定分析角色和行为",
                required=False,
            ),
            ToolParameter(
                name="max_frames",
                type="number",
                description="视频分析时最大提取帧数（默认 10，最大 20）",
                required=False,
            ),
        ]

    def _lazy_init_client(self):
        if self._client is None:
            self._client = VLClient(
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

    def run(self, args: dict) -> ToolResponse:
        action = args.get("action")
        file_path = args.get("file_path")
        file_paths = args.get("file_paths", [])
        prompt = args.get("prompt")
        texts = args.get("texts", [])
        system_prompt = args.get("system_prompt")
        max_frames = args.get("max_frames", 10)

        try:
            if action == "analyze_image":
                return self._handle_image(file_path, file_paths, prompt, system_prompt)
            elif action == "analyze_video":
                return self._handle_video(file_path, prompt, system_prompt, max_frames)
            elif action == "analyze_mixed":
                return self._handle_mixed(texts, file_paths, file_path, prompt, system_prompt)
            else:
                return ToolResponse.error(f"不支持的 action: {action}")

        except Exception as e:
            logger.exception("VLMediaTool 执行出错")
            return ToolResponse.error(f"执行出错: {e}")

    def _handle_image(
        self,
        file_path: Optional[str],
        file_paths: List[str],
        prompt: Optional[str],
        system_prompt: Optional[str],
    ) -> ToolResponse:
        paths = list(file_paths)
        if file_path:
            paths.append(file_path)

        if not paths:
            return ToolResponse.error("analyze_image 需要提供 file_path 或 file_paths 参数")

        self._lazy_init_client()

        if len(paths) == 1:
            result = self._client.analyze_image(paths[0], prompt=prompt, system_prompt=system_prompt)
        else:
            result = self._client.analyze_images(paths, prompt=prompt, system_prompt=system_prompt)

        return ToolResponse.success(output=result, data={"file_paths": paths})

    def _handle_video(
        self,
        file_path: Optional[str],
        prompt: Optional[str],
        system_prompt: Optional[str],
        max_frames: int,
    ) -> ToolResponse:
        if not file_path:
            return ToolResponse.error("analyze_video 需要提供 file_path 参数")

        self._lazy_init_client()
        result = self._client.analyze_video(
            file_path, prompt=prompt, system_prompt=system_prompt, max_frames=min(max_frames, 20),
        )
        return ToolResponse.success(output=result, data={"file_path": file_path})

    def _handle_mixed(
        self,
        texts: List[str],
        file_paths: List[str],
        file_path: Optional[str],
        prompt: Optional[str],
        system_prompt: Optional[str],
    ) -> ToolResponse:
        paths = list(file_paths)
        if file_path:
            paths.append(file_path)

        if not paths:
            return ToolResponse.error("analyze_mixed 需要提供至少一张图片（file_path 或 file_paths）")

        self._lazy_init_client()
        result = self._client.analyze_mixed(
            texts=texts or None,
            image_paths=paths,
            prompt=prompt,
            system_prompt=system_prompt,
        )
        return ToolResponse.success(output=result, data={"texts": texts, "image_paths": paths})